"""
B-19 (2026-11-01): how much do the CV numbers move when the folds respect PHYSICAL guns? A-4 found that 79 files are
72 guns ((c7, c8, c9) + continuous counters; history.md §40): four within-train groups are split across folds by the
seed-42 file-level folds every reported CV used, two of them the same week recorded twice (E03_10 / E01_3,
E01_16 / E03_3). A split pair puts near-identical windows on both sides of a fold.

Design: same production recipe as B-11 (IsolationForest + LightGBM ensemble on the B-2 60 s windows, OOF CDF score,
per-fold threshold), same model seed. Folds:
    file      train.cv_folds(files, 4, seed) - what train.py --cv and every experiment used
    physical  the same folds, then every file of a multi-file physical gun moved into the fold of its first member
              (sorted name) - the minimal change, so the difference is the leakage, not a new random split
Seeds 42 (the reported one) and more if time allows; metrics per fold as reported (auroc = last hour vs normal,
auroc_pre_rule = > 10 min before the failure vs normal, gun-mean AUROC, normal alarm rate), also split into the moved /
duplicate files vs the rest. The OOF scores of every run are saved (slim) for re-running B-18 on them, and
`summary` pools B-18's nested early-warning result over seeds: the per-run table, and each gun's hit / chance averaged
over runs with a gun bootstrap (one OOF run's CI misses the fold / re-fit noise - history.md §41).

    python results/B-19/physical_cv.py [seed ...]     -> results/B-19/physical_cv.json, cache/scores_cv_<scheme>_s<seed>.parquet
    python results/B-18/early_warning.py --cv-scores results/B-19/cache/scores_cv_<scheme>_s<seed>.parquet --tag <scheme>_s<seed>
    python results/B-19/physical_cv.py summary        -> results/B-19/summary.json (CV metrics + B-18 nested, per seed and mean)

Inputs: results/B-2/cache (production windows), results/A-4/cache/static_*.parquet (gun constants).
Never writes outside results/B-19/.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, ".py"))
sys.path.insert(0, os.path.join(ROOT, "results", "B-11"))
import horizon as H  # noqa: E402
import train as T  # noqa: E402

CACHE = os.path.join(HERE, "cache")
K = 4
SLIM = ["file", "time", "label", "ttf_s", "error_active", "terminal_any", "non_welding", "warmup", "score_ens", "alarm",
        "fold"]                                         # what B-18 early_warning.py reads


def log(msg):
    print(msg, flush=True)


def physical_ids():
    st = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(ROOT, "results", "A-4", "cache", "static_*.parquet"))])
    key = st[["c7", "c8", "c9"]].round(4).astype(str).agg("|".join, axis=1)
    return dict(zip(st["file"], key))


def physical_folds(folds, phys):
    where = {f: i for i, fs in enumerate(folds) for f in fs}
    files = sorted(where)
    first = {}
    for f in files:
        first.setdefault(phys[f], where[f])
    out = [[] for _ in folds]
    for f in files:
        out[first[phys[f]]].append(f)
    return [sorted(x) for x in out]


def oof(w, cal, cols, folds):
    out = []
    for k, va in enumerate(folds):
        t0 = time.time()
        model, thr = H.fit_production(w[~w["file"].isin(va)], cols)
        v = w[w["file"].isin(va)].copy()
        v["score_ens"] = H.score_parts(model, v[cols].to_numpy(np.float32))["ens"]
        v["alarm"] = H.alarms(v, model, cols, thr, cal[cal["file"].isin(va)])
        v["fold"] = k
        out.append(v)
        log(f"  fold {k + 1}/{K}: {len(va)} files, threshold {thr:.4f}, {time.time() - t0:.0f}s")
    return pd.concat(out, ignore_index=True)


def metrics(d, files=None):
    if files is not None:
        d = d[d["file"].isin(files)]
    s = d["score_ens"].to_numpy()
    pre = (d["label"] == 1).to_numpy()
    normal = ((d["label"] == 0) & (d["error_active"] == 0)).to_numpy()
    pre_rule = pre & (d["ttf_s"].to_numpy() > T.RULE_LEAD_S)
    out = {"auroc": [], "auroc_pre_rule": [], "auroc_gun_mean": [], "alarm_rate_normal": []}
    for k in sorted(d["fold"].unique()):
        f = (d["fold"] == k).to_numpy()
        out["auroc"].append(H.auc(s, pre & f, normal & f))
        out["auroc_pre_rule"].append(H.auc(s, pre_rule & f, normal & f))
        g = H.gun_aucs(d[f], d.loc[f, "score_ens"].to_numpy(), pre[f], normal[f])
        out["auroc_gun_mean"].append(float(np.nanmean(list(g.values()))) if isinstance(g, dict) else float(np.nanmean(g)))
        out["alarm_rate_normal"].append(float(d.loc[f & normal, "alarm"].mean()))
    pooled = {"pooled_auroc_pre_rule": H.auc(s, pre_rule, normal)}
    return {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "folds": [float(x) for x in v]}
            for k, v in out.items()} | pooled


def pooled(runs, n=2000):
    """Average each gun's nested hit / chance over runs (same 64 failures, different folds / re-fits), then bootstrap
    guns: the CI is the failure-sampling noise left after the fold and re-fit noise is averaged out."""
    files = sorted(runs[0]["file"])
    get = {k: np.array([[dict(zip(r["file"], r[k]))[f] for f in files] for r in runs], dtype=float)
           for k in ("real", "chance", "fa", "days")}
    real, plc = np.nanmean(get["real"], axis=0), np.nanmean(get["chance"], axis=0)
    rng = np.random.default_rng(42)
    ex = [np.nanmean(real[i]) - np.nanmean(plc[i]) for i in (rng.integers(0, len(files), len(files)) for _ in range(n))]
    return {"runs": len(runs), "fa_per_day": round(float(get["fa"].sum() / get["days"].sum()), 2),
            "recall": round(float(np.nanmean(real)), 3), "chance": round(float(np.nanmean(plc)), 3),
            "excess": round(float(np.nanmean(real) - np.nanmean(plc)), 3),
            "ci": [round(float(np.nanpercentile(ex, q)), 3) for q in (2.5, 97.5)]}


def summary():
    res = json.load(open(os.path.join(HERE, "physical_cv.json"), encoding="utf-8"))
    b18 = os.path.join(ROOT, "results", "B-18")
    out = {"cv": {}, "early_warning_nested": {}, "early_warning_pooled": {}}
    per_gun = {}
    for scheme in ("file", "physical"):
        rows = {s: r[scheme]["all"] for s, r in res.items()}
        out["cv"][scheme] = {k: {"per_seed": {s: round(m[k]["mean"], 3) for s, m in rows.items()},
                                 "mean": round(float(np.mean([m[k]["mean"] for m in rows.values()])), 3),
                                 "sd_over_seeds": round(float(np.std([m[k]["mean"] for m in rows.values()])), 3)}
                             for k in ("auroc", "auroc_pre_rule", "auroc_gun_mean", "alarm_rate_normal")}
        ew = {}
        for s in res:
            p = os.path.join(b18, f"early_warning_{scheme}_s{s}.json")
            if os.path.exists(p):
                ew[s] = json.load(open(p, encoding="utf-8"))["nested"]
                for b, v in ew[s].items():
                    if "per_gun" in v:
                        per_gun.setdefault(scheme, {}).setdefault(b, []).append(v["per_gun"])
        out["early_warning_nested"][scheme] = {
            b: {"per_seed": {s: {k: round(v[b][k], 3) for k in ("fa_per_day", "recall", "chance", "excess")} | {
                "ci": [round(x, 3) for x in v[b]["ci"]]} for s, v in ew.items()},
                "mean_excess": round(float(np.mean([v[b]["excess"] for v in ew.values()])), 3),
                "mean_recall": round(float(np.mean([v[b]["recall"] for v in ew.values()])), 3),
                "mean_chance": round(float(np.mean([v[b]["chance"] for v in ew.values()])), 3),
                "mean_fa": round(float(np.mean([v[b]["fa_per_day"] for v in ew.values()])), 2),
                "seeds_ci_above_0": int(sum(v[b]["ci"][0] > 0 for v in ew.values()))}
            for b in (next(iter(ew.values())).keys() if ew else [])}
    for scheme in ("file", "physical"):
        c = out["cv"][scheme]
        log(f"{scheme:8s} CV over seeds {list(res)}: auroc {c['auroc']['mean']} (sd {c['auroc']['sd_over_seeds']})  pre_rule "
            f"{c['auroc_pre_rule']['mean']} (sd {c['auroc_pre_rule']['sd_over_seeds']}) {c['auroc_pre_rule']['per_seed']}")
        for b, v in out["early_warning_nested"][scheme].items():
            log(f"   budget {b:>4}: fa {v['mean_fa']}  recall {v['mean_recall']}  chance {v['mean_chance']}  excess "
                f"{v['mean_excess']:+.3f}  (CI above 0 in {v['seeds_ci_above_0']}/{len(v['per_seed'])} seeds)  "
                + "  ".join(f"s{s} {x['excess']:+.3f}" for s, x in v["per_seed"].items()))
    per_gun["both"] = {b: per_gun.get("file", {}).get(b, []) + per_gun.get("physical", {}).get(b, [])
                       for b in per_gun.get("file", {})}
    for scheme in ("file", "physical", "both"):
        out["early_warning_pooled"][scheme] = {b: pooled(v) for b, v in per_gun.get(scheme, {}).items() if v}
        for b, v in out["early_warning_pooled"][scheme].items():
            log(f"pooled {scheme:8s} budget {b:>4}: {v['runs']} runs  fa {v['fa_per_day']}  recall {v['recall']}  chance "
                f"{v['chance']}  excess {v['excess']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log("saved summary.json")


def main():
    if sys.argv[1:] == ["summary"]:
        return summary()
    seeds = [int(a) for a in sys.argv[1:]] or [42]
    os.makedirs(CACHE, exist_ok=True)
    w, cal, cols = H.load()
    wt, ct = w["train"], cal["train"]
    phys = physical_ids()
    files = sorted(wt["file"].unique())
    dup = sorted(f for f in files if sum(phys[g] == phys[f] for g in files) > 1)
    log(f"{len(files)} train files, {len(set(phys[f] for f in files))} physical guns; duplicate files {dup}")
    dst = os.path.join(HERE, "physical_cv.json")
    res = json.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else {}
    for seed in seeds:
        ff = T.cv_folds(files, K, seed)
        pf = physical_folds(ff, phys)
        moved = sorted(f for f in files if [f in x for x in ff] != [f in x for x in pf])
        log(f"seed {seed}: moved {moved}")
        r = {"moved": moved, "duplicates": dup}
        for scheme, folds in (("file", ff), ("physical", pf)):
            t0 = time.time()
            log(f" {scheme} folds: sizes {[len(x) for x in folds]}")
            d = oof(wt, ct, cols, folds)
            d[SLIM].to_parquet(os.path.join(CACHE, f"scores_cv_{scheme}_s{seed}.parquet"))
            r[scheme] = {"folds": folds, "all": metrics(d), "duplicates": metrics(d, dup),
                         "others": metrics(d, [f for f in files if f not in dup])}
            m = r[scheme]["all"]
            log(f" {scheme}: auroc {m['auroc']['mean']:.3f} +- {m['auroc']['sd']:.3f}  pre_rule "
                f"{m['auroc_pre_rule']['mean']:.3f} +- {m['auroc_pre_rule']['sd']:.3f}  gun {m['auroc_gun_mean']['mean']:.3f}"
                f"  alarm@normal {m['alarm_rate_normal']['mean']:.4f}  | duplicates pre_rule "
                f"{r[scheme]['duplicates']['pooled_auroc_pre_rule']:.3f}, others "
                f"{r[scheme]['others']['pooled_auroc_pre_rule']:.3f}  ({time.time() - t0:.0f}s)")
            del d
        res[str(seed)] = r
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=lambda o: None if isinstance(o, float) and o != o else str(o))
        log(f"saved {dst}")


if __name__ == "__main__":
    main()
