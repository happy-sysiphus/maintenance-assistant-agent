"""
B-21 (2026-11-02): does training the LightGBM half on PRE-fault windows only change the early warning? The user's
definition doc (`docs/에러코드와 센서값 의미 EDA(클로드) 일부.docx` §5) says the 10 minutes after the fault (file end -
600 s, the terminal code; 63/72 files exactly) are reset / restart data and should not be trained on. The production
recipe labels the file's last 3600 s as positive (train.py: label = ttf_s <= 3600), so ~1/6 of the positives are
post-fault windows (terminal code, reset, restart), and error-state windows sit on the positive side only. That is
not a serving leak (those windows never precede a failure online, and the terminal-code rule takes them) but a target
mismatch: the supervised half may learn "restart / error state looks like a failure", which could be where part of
the early warning's +14.6 points comes from (A-5: E01 / E03 are start-up failures).

PRE-REGISTERED before any B-21 score was computed (2026-11-02):
    variant     V1 = the production recipe (B-11 fit_production: IsolationForest + LightGBM, --exclude-non-welding,
                model seed 42) with the LightGBM positives restricted to windows ending BEFORE the fault (lead > 0;
                fault = the first own terminal code, results/B-17/cache/meta.csv). IsolationForest, negatives and
                features unchanged.
    baseline    BASE = the B-19 OOF scores of the same folds (production recipe, all last-hour positives)
    runs        the 8 B-19 / B-20 runs: fold seeds 42, 1, 2, 3 x file / physical folds; each gun's hit / chance
                averaged over runs, gun bootstrap 2000, paired for differences (results/A-5/overlap.py helpers)
    rule        the B-20 production early warning: model_ng_k1 (one window >= threshold, no gate, warm-up never
                alarms), threshold = a quantile of each run's normal OOF scores (lead > 1 day, no error, no warm-up)
    metrics     B-18 exactly: hit = alarm in a window ending 10-60 min before T, chance = the same at T - 1..6 days,
                false alarm = run onsets > 60 min before T per gun-day
    primary     D = excess(V1 at q_m) - excess(BASE at 0.975); q_m = the highest B-18 grid quantile whose pooled V1
                false alarms per gun-day >= BASE's at 0.975 (V1 spends at least as many alarms)
    adopt V1 if (1) D CI lower bound > 0, (2) test (8 guns; each run's V1 threshold at q_m on V1 test scores) mean
                excess >= BASE's (the production recipe refitted the same way on all 64 guns, threshold at 0.975) -
                direction only, (3) CV auroc_pre_rule (windows > 10 min before the failure vs normal) mean over the
                8 runs not below BASE's by more than 0.01
    otherwise   production stays as it is (no retrain)
    reading     D CI upper bound < 0 -> part of the early warning comes from post-fault-like windows (restart / error
                state); recorded as an explanation of the signal, not acted on
    reported, not decided on: V1 at 0.975 as it is, both rules over B-18's quantile grid (false alarms vs excess),
    by class, the test with the production bundle's scores (B-20's reference), and the all-positives auroc (it is
    expected to fall: V1 no longer learns the terminal state the rule covers)

    python results/B-21/target_align.py smoke           -> code-path check on 8 guns, 2 folds (nothing kept)
    python results/B-21/target_align.py run [seed ...]  -> cache/scores_cv_<scheme>_s<seed>.parquet, target_align_cv.json
    python results/B-21/target_align.py test            -> cache/scores_test_{base,v1}.parquet (fits on all 64 guns)
    python results/B-21/target_align.py evaluate        -> target_align.json (+ .log via shell)

Inputs: results/B-2/cache (production windows), results/B-19 (folds, OOF caches), results/B-17/cache/meta.csv,
results/B-11/cache/scores_test.parquet. Run one sub-command at a time (~1 GB free RAM). Never writes outside
results/B-21/.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (".py", "results/B-11", "results/B-19", "results/B-18", "results/A-5"):
    sys.path.insert(0, os.path.join(ROOT, *p.split("/")))
import early_warning as EW  # noqa: E402
import horizon as H  # noqa: E402
import overlap as OV  # noqa: E402
import physical_cv as PC  # noqa: E402
import train as T  # noqa: E402

CACHE = os.path.join(HERE, "cache")
B19 = os.path.join(ROOT, "results", "B-19")
SEEDS = (42, 1, 2, 3)
RUNS = OV.RUNS                       # (scheme, seed) x 8, same order as B-20 / overlap
Q = OV.Q                             # 0.975, B-20
QS = OV.QS
PRE_RULE_TOL = 0.01


def log(msg):
    print(msg, flush=True)


def add_lead(w):
    meta = pd.read_csv(EW.META, parse_dates=["fault_time"]).set_index("file")
    w["lead"] = (w["file"].map(meta["fault_time"]) - (w["time"] + pd.Timedelta(seconds=60))).dt.total_seconds()
    return w


def fit(tr, cols, variant):
    """H.fit_production, except that V1 keeps only the positives that end before the fault."""
    normal = (tr["label"] == 0) & (tr["error_active"] == 0) & (tr["non_welding"] == 0)
    pos = (tr["label"] == 1) & (tr["non_welding"] == 0)
    n_all = int(pos.sum())
    if variant == "v1":
        pos &= tr["lead"] > 0
    fit_, pos_ = tr[normal], tr[pos]
    model = T.build_model("ensemble", H.SEED, cols)
    model.fit(fit_[cols].to_numpy(np.float32), pos_[cols].to_numpy(np.float32), fit_["file"].to_numpy(),
              pos_["file"].to_numpy(), fit_["ttf_s"].to_numpy())
    return model, float(np.quantile(model.fit_scores_, H.Q)), n_all, len(pos_)


def score(model, thr, v, cols, cal, fold):
    v = v.copy()
    v["score_ens"] = H.score_parts(model, v[cols].to_numpy(np.float32))["ens"]
    v["alarm"] = H.alarms(v, model, cols, thr, cal)
    v["fold"] = fold
    return v[PC.SLIM]


def oof(w, cal, cols, folds):
    out = []
    for k, va in enumerate(folds):
        t0 = time.time()
        model, thr, n_all, n_kept = fit(w[~w["file"].isin(va)], cols, "v1")
        out.append(score(model, thr, w[w["file"].isin(va)], cols, cal[cal["file"].isin(va)], k))
        log(f"  fold {k + 1}/{len(folds)}: {len(va)} files, positives {n_kept}/{n_all} kept, threshold {thr:.4f}, "
            f"{time.time() - t0:.0f}s")
        del model
    return pd.concat(out, ignore_index=True)


def load_train():
    w, cal, cols = H.load()
    return add_lead(w["train"]), cal["train"], add_lead(w["test"]), cal["test"], cols


def smoke():
    wt, ct, _, _, cols = load_train()
    files = sorted(wt["file"].unique())[::9][:8]
    folds = T.cv_folds(files, 2, 42)
    d = oof(wt[wt["file"].isin(files)], ct, cols, folds)
    log(f"smoke metrics: { {k: v['mean'] for k, v in PC.metrics(d).items() if isinstance(v, dict)} }")
    tmp = os.path.join(CACHE, "smoke.parquet")
    os.makedirs(CACHE, exist_ok=True)
    d.to_parquet(tmp)
    fr = EW.Frame(EW.load("cv", tmp))
    r = OV.per_gun(fr, fr.condition("model_ng", OV.normal_thr(fr, Q)))
    log(f"smoke early warning: guns {len(fr.guns)}, hits {np.nansum(r['real'])}, fa {r['fa'].sum()}")
    os.remove(tmp)


def run(seeds):
    os.makedirs(CACHE, exist_ok=True)
    wt, ct, _, _, cols = load_train()
    phys = PC.physical_ids()
    files = sorted(wt["file"].unique())
    dst = os.path.join(HERE, "target_align_cv.json")
    res = json.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else {}
    for seed in seeds:
        ff = T.cv_folds(files, PC.K, seed)
        for scheme, folds in (("file", ff), ("physical", PC.physical_folds(ff, phys))):
            key = f"{scheme}_s{seed}"
            path = os.path.join(CACHE, f"scores_cv_{key}.parquet")
            if key in res and os.path.exists(path):
                log(f"{key}: done, skipped")
                continue
            t0 = time.time()
            log(f"{key}: fold sizes {[len(x) for x in folds]}")
            d = oof(wt, ct, cols, folds)
            d.to_parquet(path)
            res[key] = PC.metrics(d)
            m = res[key]
            log(f"{key}: auroc {m['auroc']['mean']:.3f}  pre_rule {m['auroc_pre_rule']['mean']:.3f}  gun "
                f"{m['auroc_gun_mean']['mean']:.3f}  alarm@normal {m['alarm_rate_normal']['mean']:.4f}  "
                f"({time.time() - t0:.0f}s)")
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            del d


def test():
    os.makedirs(CACHE, exist_ok=True)
    wt, _, we, ce, cols = load_train()
    for variant in ("base", "v1"):
        t0 = time.time()
        model, thr, n_all, n_kept = fit(wt, cols, variant)
        score(model, thr, we, cols, ce, -1).to_parquet(os.path.join(CACHE, f"scores_test_{variant}.parquet"))
        log(f"test {variant}: positives {n_kept}/{n_all}, threshold {thr:.4f}, {time.time() - t0:.0f}s")
        del model


def test_row(fr, thr, run):
    fa, hits = fr.evaluate(fr.condition("model_ng", thr), 1)
    t = EW.summarise(fa, hits, fr.days)
    return {"run": run, "fa_per_day": round(t["fa_per_day"], 2), "hits": int(np.nansum(hits["real"])),
            "chance": round(t["chance"], 3), "excess": round(t["excess"], 3),
            "hit_guns": [str(g) for g, h in zip(fr.guns, hits["real"]) if h == 1]}


def test_summary(rows):
    return {"runs": rows, "hits_mean": round(float(np.mean([r["hits"] for r in rows])), 2),
            "excess_mean": round(float(np.mean([r["excess"] for r in rows])), 3),
            "fa_per_day_mean": round(float(np.mean([r["fa_per_day"] for r in rows])), 2)}


def cv_metric(path, keys, name):
    res = json.load(open(path, encoding="utf-8"))
    return float(np.mean([keys(res, sc, s)[name]["mean"] for sc, s in RUNS]))


def evaluate():
    te = {"base": EW.Frame(EW.load("test", os.path.join(CACHE, "scores_test_base.parquet"))),
          "v1": EW.Frame(EW.load("test", os.path.join(CACHE, "scores_test_v1.parquet"))),
          "bundle": EW.Frame(EW.load("test"))}
    per = {"BASE": [], "V1_975": []}
    grid = {v: {q: [] for q in QS} for v in ("BASE", "V1")}
    thr_v = {}
    test = {"BASE": [], "BASE_bundle": []}
    guns = None
    for sc, s in RUNS:
        run = f"{sc}_s{s}"
        for name, path in (("BASE", os.path.join(B19, "cache", f"scores_cv_{run}.parquet")),
                           ("V1", os.path.join(CACHE, f"scores_cv_{run}.parquet"))):
            cv = EW.Frame(EW.load("cv", path))
            guns = list(cv.guns) if guns is None else guns
            assert list(cv.guns) == guns
            ts = OV.normal_thr(cv, QS)
            for q, t in zip(QS, ts):
                grid[name][q].append(OV.per_gun(cv, cv.condition("model_ng", t)))
            t975 = float(OV.normal_thr(cv, Q))
            if name == "BASE":
                per["BASE"].append(OV.per_gun(cv, cv.condition("model_ng", t975)))
                test["BASE"].append(test_row(te["base"], t975, run))
                test["BASE_bundle"].append(test_row(te["bundle"], t975, run))
            else:
                per["V1_975"].append(OV.per_gun(cv, cv.condition("model_ng", t975)))
                thr_v[run] = dict(zip(QS, ts))
            log(f"{run} {name}: thr@0.975 {t975:.4f}")
            del cv
    out = {"criteria": __doc__.split("PRE-REGISTERED")[1].split("    python")[0].strip(), "pooled": {}, "test": {}}
    out["pooled"]["BASE"] = OV.summary(per["BASE"])
    out["pooled"]["V1_975"] = OV.summary(per["V1_975"])
    fa_b = out["pooled"]["BASE"]["fa_per_day"]
    curve = {v: {q: OV.summary(g[q]) for q in QS} for v, g in grid.items()}
    ok = [q for q in QS if curve["V1"][q]["fa_per_day"] >= fa_b]
    q_m = max(ok) if ok else min(QS)
    per["V1_m"] = grid["V1"][q_m]
    out["pooled"]["V1_matched"] = OV.summary(per["V1_m"]) | {"q": round(float(q_m), 3)}
    out["curve"] = {v: {f"{q:.3f}": c for q, c in cv.items()} for v, cv in curve.items()}
    out["D_primary"] = OV.paired(per["V1_m"], per["BASE"])
    out["D_at_0975"] = OV.paired(per["V1_975"], per["BASE"])
    test["V1_matched"] = [test_row(te["v1"], thr_v[f"{sc}_s{s}"][q_m], f"{sc}_s{s}") for sc, s in RUNS]
    test["V1_975"] = [test_row(te["v1"], thr_v[f"{sc}_s{s}"][Q], f"{sc}_s{s}") for sc, s in RUNS]
    out["test"] = {k: test_summary(v) for k, v in test.items()}
    pc = lambda res, sc, s: res[str(s)][sc]["all"]  # noqa: E731
    me = lambda res, sc, s: res[f"{sc}_s{s}"]  # noqa: E731
    out["cv_auroc"] = {k: {"BASE": round(cv_metric(os.path.join(B19, "physical_cv.json"), pc, k), 4),
                           "V1": round(cv_metric(os.path.join(HERE, "target_align_cv.json"), me, k), 4)}
                       for k in ("auroc", "auroc_pre_rule", "auroc_gun_mean", "alarm_rate_normal")}
    cls = pd.read_csv(EW.META).set_index("file")["class"].reindex(guns).to_numpy()
    out["by_class"] = {}
    for name, runs in (("BASE", per["BASE"]), ("V1_matched", per["V1_m"])):
        real, ch, _ = OV.stack(runs)
        out["by_class"][name] = {c: {"recall": round(float(np.nanmean(real[cls == c])), 3),
                                     "chance": round(float(np.nanmean(ch[cls == c])), 3),
                                     "excess": round(float(np.nanmean(real[cls == c]) - np.nanmean(ch[cls == c])), 3)}
                                 for c in sorted(set(cls))}
    d = out["D_primary"]
    pre = out["cv_auroc"]["auroc_pre_rule"]
    checks = {"D_ci_above_0": d["ci"][0] > 0,
              "test_not_worse": out["test"]["V1_matched"]["excess_mean"] >= out["test"]["BASE"]["excess_mean"],
              "pre_rule_kept": pre["V1"] >= pre["BASE"] - PRE_RULE_TOL}
    out["checks"] = checks
    out["adopt_v1"] = bool(all(checks.values()))
    out["post_fault_reliance"] = bool(d["ci"][1] < 0)
    log("\npooled over 8 runs (64 CV guns): fa/gun-day  recall  chance  excess [95% CI]")
    for name, v in out["pooled"].items():
        log(f"  {name:10s} {v['fa_per_day']:6.2f}  {v['recall']:.3f}  {v['chance']:.3f}  {v['excess']:+.3f} "
            f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" + (f"  (q {v['q']})" if "q" in v else ""))
    log(f"D primary (V1 matched - BASE) {d}")
    log(f"D at 0.975 (V1 - BASE)        {out['D_at_0975']}")
    log("curve (q: BASE fa/excess | V1 fa/excess): " + "  ".join(
        f"{q:.3f}: {curve['BASE'][q]['fa_per_day']}/{curve['BASE'][q]['excess']:+.3f} | "
        f"{curve['V1'][q]['fa_per_day']}/{curve['V1'][q]['excess']:+.3f}" for q in QS if q >= 0.9))
    log("test (8 guns, run mean): " + json.dumps({k: {kk: v[kk] for kk in ("hits_mean", "excess_mean", "fa_per_day_mean")}
                                                   for k, v in out["test"].items()}))
    log(f"CV window AUROC (mean of 8 runs): {out['cv_auroc']}")
    log(f"by class: {json.dumps(out['by_class'])}")
    log(f"checks {checks} -> adopt V1: {out['adopt_v1']}; post-fault reliance: {out['post_fault_reliance']}")
    with open(os.path.join(HERE, "target_align.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log("saved target_align.json")


def main():
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "smoke":
        smoke()
    elif cmd == "run":
        run([int(a) for a in args] or list(SEEDS))
    elif cmd == "test":
        test()
    elif cmd == "evaluate":
        evaluate()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
