"""
B-11 (2026-10-03): how well does the model see the failure at each horizon (3 min ... 24 h before it), and WHY does
the separation change with the horizon? Gun-level 4-fold CV of the production recipe + the 8 test guns.

    horizons  0-3m, 3-10m, 10-30m, 30-60m, 1-3h, 3-6h, 6-12h, 12-24h before the failure (ttf_s of the window end)
    negatives error-free, non-warm-up windows > 24 h before the failure of the same held-out guns

Per horizon it reports (a) the production ensemble and its two halves (IsolationForest / LightGBM CDF), pooled and
per-gun AUROC and the alarm rate, (b) the same on "clean" windows (no error state, welding) - what is left once the
terminal-code state and cap dressing are taken out, (c) the window STATE mix (error / terminal code / non-welding /
weld duty), (d) a horizon-specific LightGBM trained in the same 4 folds (positives = that horizon) - the learnable upper
bound of the 46 features at that horizon, and (e) a PLACEBO for all of it: the same numbers with a pseudo failure
anchored A hours earlier (ttf' = ttf - A, only windows before the anchor). Scores are causal (6 h warm-up from the file
start, 10 min rolling), so a placebo anchor sees the same elapsed-time drift and gun-norm ageing as the real failure;
real - placebo is the failure-specific part. (f) per-feature within-gun effect sizes (real vs placebo) and how many
guns agree on the sign.

    python results/B-11/horizon.py            # ~13 min, writes results/B-11/horizon_cv.json (+ cache/scores_*.parquet)
    python results/B-11/horizon.py followup   # horizon x window state, activity, code onset -> horizon_followup.json

Inputs: results/B-2/cache/w60_*.parquet + calib60_*.parquet (production windows: 2026-09-29 preprocessing, 6 h
gun-centring), models/baseline_ensemble.joblib (test guns). Never writes outside results/B-11/.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import train as T  # noqa: E402

warnings.filterwarnings("ignore")
B2 = os.path.join(ROOT, "results", "B-2", "cache")
SEED, K, GATE, Q = 42, 4, 0.5, 0.99
M = 60
BINS = [(0, 3 * M), (3 * M, 10 * M), (10 * M, 30 * M), (30 * M, 60 * M), (3600, 3 * 3600), (3 * 3600, 6 * 3600),
        (6 * 3600, 12 * 3600), (12 * 3600, 24 * 3600)]
LABELS = ["0-3m", "3-10m", "10-30m", "30-60m", "1-3h", "3-6h", "6-12h", "12-24h"]
NEG_MIN_S = 24 * 3600
PLACEBO_H = list(range(36, 121, 6))   # 15 pseudo-failure anchors (h before the real failure) for the per-gun null
PLACEBO_MAIN_H = 72                   # the one anchor the horizon-specific LightGBM is refitted for
SUP_NEG_SUB = 120_000


def log(msg):
    print(msg, flush=True)


def load():
    meta = json.load(open(os.path.join(B2, "meta60.json"), encoding="utf-8"))
    feat = [c for c in meta["feature_cols"] if c != "c19"]
    cols = T.model_input_columns(feat)
    w = {s: pd.read_parquet(os.path.join(B2, f"w60_{s}.parquet")) for s in ("train", "test")}
    cal = {s: pd.read_parquet(os.path.join(B2, f"calib60_{s}.parquet")) for s in ("train", "test")}
    for d in (*w.values(), *cal.values()):
        d.drop(columns=[c for c in d.columns if c.startswith("c19_")], inplace=True)
    return w, cal, cols


# ------------------------------------------------------------------ scoring
def fit_production(tr, cols):
    """train.py fit_and_eval with --exclude-non-welding (ensemble = iforest + lgbm, OOF CDF + threshold)."""
    normal = (tr["label"] == 0) & (tr["error_active"] == 0) & (tr["non_welding"] == 0)
    fit = tr[normal]
    pos = tr[(tr["label"] == 1) & (tr["non_welding"] == 0)]
    model = T.build_model("ensemble", SEED, cols)
    model.fit(fit[cols].to_numpy(np.float32), pos[cols].to_numpy(np.float32), fit["file"].to_numpy(),
              pos["file"].to_numpy(), fit["ttf_s"].to_numpy())
    return model, float(np.quantile(model.fit_scores_, Q))


def score_parts(model, X):
    """ensemble score + the CDF of each half (same scale as the ensemble)."""
    s_if = -model.iforest.score_samples(model._xf(X))
    s_lg = model.sup["lgbm"].predict_proba(X)[:, 1]
    c_if, c_lg = model._cdf(model.grids["iforest"], s_if), model._cdf(model.grids["lgbm"], s_lg)
    return {"ens": (1 - model.lgbm_weight) * c_if + model.lgbm_weight * c_lg, "iforest": c_if, "lgbm": c_lg}


def alarms(w, model, cols, thr, cal):
    gun_thr = T.gun_thresholds(model, cols, cal, thr, Q, GATE)
    t = T.window_thresholds(w, thr, gun_thr)
    return T.alarm_mask(w["score_ens"].to_numpy(), t, w["non_welding"].to_numpy(), GATE, w["warmup"].to_numpy())


def oof_scores(w, cal, cols):
    files = sorted(w["file"].unique())
    folds = T.cv_folds(files, K, SEED)
    out = []
    for k, va in enumerate(folds):
        t0 = time.time()
        model, thr = fit_production(w[~w["file"].isin(va)], cols)
        v = w[w["file"].isin(va)].copy()
        for name, s in score_parts(model, v[cols].to_numpy(np.float32)).items():
            v[f"score_{name}"] = s
        v["alarm"] = alarms(v, model, cols, thr, cal[cal["file"].isin(va)])
        v["fold"] = k
        out.append(v)
        log(f"fold {k + 1}/{K}: {len(va)} guns, threshold {thr:.4f}, {time.time() - t0:.0f}s")
    return pd.concat(out, ignore_index=True), folds


def test_scores(w, cal, cols):
    b = T.load_bundle(os.path.join(ROOT, "models", "baseline_ensemble.joblib"))
    assert b["model_cols"] == cols, "bundle features differ from the cache"
    w = w.copy()
    for name, s in score_parts(b["model"], w[cols].to_numpy(np.float32)).items():
        w[f"score_{name}"] = s
    w["alarm"] = alarms(w, b["model"], cols, b["threshold"], cal)
    w["fold"] = -1
    return w


# ------------------------------------------------------------------ horizon tables
def sets(w, shift_s=0):
    """ttf relative to the (pseudo) failure; masks of each horizon bin and of the negatives."""
    ttf = w["ttf_s"].to_numpy() - shift_s
    valid = ttf > 0
    neg = valid & (ttf > NEG_MIN_S) & (w["error_active"].to_numpy() == 0) & (w["warmup"].to_numpy() == 0)
    bins = [valid & (ttf > lo) & (ttf <= hi) & (w["warmup"].to_numpy() == 0) for lo, hi in BINS]
    return bins, neg


def clean_mask(w):
    return (w["error_active"].to_numpy() == 0) & (w["non_welding"].to_numpy() <= GATE)


def auc(s, pos, neg):
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    return float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], np.r_[s[pos], s[neg]]))


def gun_aucs(w, score, pos, neg):
    out = {}
    for f, idx in w.groupby("file").indices.items():
        m = np.zeros(len(w), bool)
        m[idx] = True
        if (pos & m).sum() >= 2 and (neg & m).sum() >= 20:
            out[f] = auc(score, pos & m, neg & m)
    return out


def horizon_table(w, shift_s=0, per_gun=True):
    bins, neg = sets(w, shift_s)
    clean = clean_mask(w)
    rows = []
    for lab, pos in zip(LABELS, bins):
        r = {"bin": lab, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
             "error_state": float(w["error_active"].to_numpy()[pos].mean()) if pos.any() else np.nan,
             "terminal_code": float(w["terminal_any"].to_numpy()[pos].mean()) if pos.any() else np.nan,
             "non_welding_held": float((w["non_welding"].to_numpy()[pos] > GATE).mean()) if pos.any() else np.nan,
             "weld_duty": float(w["weld_duty_10min_mean"].to_numpy()[pos].mean()) if pos.any() else np.nan,
             "alarm_rate": float(w["alarm"].to_numpy()[pos].mean()) if pos.any() else np.nan}
        for name in ("ens", "iforest", "lgbm"):
            s = w[f"score_{name}"].to_numpy()
            r[f"auc_{name}"] = auc(s, pos, neg)
            r[f"auc_{name}_clean"] = auc(s, pos & clean, neg & clean)
        if per_gun:
            g = gun_aucs(w, w["score_ens"].to_numpy(), pos, neg)
            r["auc_gun"] = g
            r["auc_gun_mean"] = float(np.nanmean(list(g.values()))) if g else np.nan
            gc = gun_aucs(w, w["score_ens"].to_numpy(), pos & clean, neg & clean)
            r["auc_gun_clean"] = gc
            r["auc_gun_clean_mean"] = float(np.nanmean(list(gc.values()))) if gc else np.nan
        rows.append(r)
    neg_alarm = float(w["alarm"].to_numpy()[neg].mean())
    return rows, neg_alarm


def placebo_null(w):
    """Per horizon: pooled AUROC and per-gun AUROC under each pseudo-failure anchor."""
    out = {}
    for a in PLACEBO_H:
        rows, neg_alarm = horizon_table(w, a * 3600)
        out[a] = {"rows": rows, "neg_alarm": neg_alarm}
    return out


def gun_excess(real_rows, null, key="auc_gun"):
    """Per horizon: guns whose real per-gun AUROC beats every placebo anchor (p < 1/16), vs the expectation."""
    res = []
    for i, lab in enumerate(LABELS):
        real = real_rows[i][key]
        beats, n = 0, 0
        for f, v in real.items():
            nv = [null[a]["rows"][i][key].get(f) for a in PLACEBO_H]
            nv = [x for x in nv if x is not None and x == x]
            if len(nv) >= 8 and v == v:
                n += 1
                beats += v > max(nv)
        res.append({"bin": lab, "guns": n, "beat_all_placebos": beats,
                    "expected": round(n / (len(PLACEBO_H) + 1), 1)})
    return res


# ------------------------------------------------------------------ horizon-specific upper bound
def horizon_lgbm(w, cols, folds, shift_s=0):
    """4-fold LightGBM per horizon: positives = that horizon's windows (clean), negatives = clean > 24 h."""
    bins, neg = sets(w, shift_s)
    clean = clean_mask(w)
    rng = np.random.default_rng(SEED)
    X = w[cols].to_numpy(np.float32)
    files = w["file"].to_numpy()
    res = []
    for lab, pos in zip(LABELS, bins):
        pos, ng = pos & clean, neg & clean
        oof = np.full(len(w), np.nan)
        for va in folds:
            te = np.isin(files, va)
            ni = np.flatnonzero(ng & ~te)
            if len(ni) > SUP_NEG_SUB:
                ni = rng.choice(ni, SUP_NEG_SUB, replace=False)
            pi = np.flatnonzero(pos & ~te)
            clf = T.lgbm_classifier(len(ni), len(pi), SEED)
            clf.set_params(n_estimators=200, n_jobs=4)
            clf.fit(np.vstack([X[ni], X[pi]]), np.r_[np.zeros(len(ni)), np.ones(len(pi))])
            m = te & (pos | ng)
            oof[m] = clf.predict_proba(X[m])[:, 1]
        g = gun_aucs(w, oof, pos, ng)
        res.append({"bin": lab, "n_pos": int(pos.sum()), "auc": auc(oof, pos, ng),
                    "auc_gun_mean": float(np.nanmean(list(g.values()))) if g else np.nan})
        log(f"  horizon LGBM{' placebo' if shift_s else ''} {lab}: AUROC {res[-1]['auc']:.3f} "
            f"(per gun {res[-1]['auc_gun_mean']:.3f}, {res[-1]['n_pos']} pos)")
    return res


# ------------------------------------------------------------------ feature effect sizes
def effects(w, cols, shift_s=0):
    """Per horizon and feature: median over guns of (mean of the horizon's clean windows - median of the gun's clean
    negatives) / robust sd of those negatives, and the share of guns with the median's sign."""
    bins, neg = sets(w, shift_s)
    clean = clean_mask(w)
    feats = [c for c in cols if not c.startswith(("hour_", "c7_", "c8_", "c9_"))]
    X = w[feats].to_numpy(np.float64)
    out = {}
    for lab, pos in zip(LABELS, bins):
        per = []
        for f, idx in w.groupby("file").indices.items():
            p, n = idx[(pos & clean)[idx]], idx[(neg & clean)[idx]]
            if len(p) < 2 or len(n) < 50:
                continue
            q75, q25 = np.percentile(X[n], [75, 25], axis=0)
            sd = np.maximum((q75 - q25) / 1.349, X[n].std(axis=0) * 0.1 + 1e-6)
            per.append((X[p].mean(axis=0) - np.median(X[n], axis=0)) / sd)
        per = np.array(per)
        med = np.median(per, axis=0)
        agree = (np.sign(per) == np.sign(med)).mean(axis=0)
        out[lab] = {"n_guns": len(per), "median_d": dict(zip(feats, med.round(3).tolist())),
                    "sign_agreement": dict(zip(feats, agree.round(3).tolist()))}
    return out


def top_effects(eff, real_lab, n=6):
    d = eff[real_lab]["median_d"]
    return sorted(d.items(), key=lambda kv: -abs(kv[1]))[:n]


# ------------------------------------------------------------------ main
def fmt(x):
    return "  nan" if x != x else f"{x:.3f}"


CACHE = os.path.join(HERE, "cache")


def scored():
    """OOF (cv) and bundle (test) scored windows, cached in results/B-11/cache/ (git-ignored)."""
    w, cal, cols = load()
    folds = T.cv_folds(sorted(w["train"]["file"].unique()), K, SEED)
    paths = {s: os.path.join(CACHE, f"scores_{s}.parquet") for s in ("cv", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        return pd.read_parquet(paths["cv"]), pd.read_parquet(paths["test"]), cols, folds
    log(f"train {len(w['train']):,} windows / {w['train']['file'].nunique()} guns, test {len(w['test']):,} / "
        f"{w['test']['file'].nunique()} guns, {len(cols)} features")
    cv, folds = oof_scores(w["train"], cal["train"], cols)
    te = test_scores(w["test"], cal["test"], cols)
    os.makedirs(CACHE, exist_ok=True)
    cv.to_parquet(paths["cv"])
    te.to_parquet(paths["test"])
    return cv, te, cols, folds


def followup():
    """Why the last 10 min separate and 10 min+ does not: split each horizon by window state and compare activity.
    States: terminal = a terminal code in the window, other_err = another error code, after_err = error-free window
    with an error code in the last 10 min (error_share_10min > 0), quiet = none of these. Negatives: quiet windows
    > 24 h before the failure (same population as the 'quiet' positives)."""
    cv, te, cols, _ = scored()
    out = {}
    for tag, w in (("cv", cv), ("test", te)):
        bins, neg = sets(w)
        err, term = w["error_active"].to_numpy() > 0, w["terminal_any"].to_numpy() > 0
        after = ~err & (w["error_share_10min_mean"].to_numpy() > 0)
        quiet = ~err & ~after
        weld = w["non_welding"].to_numpy() <= GATE
        state = {"terminal": term, "other_err": err & ~term, "after_err": after, "quiet": quiet}
        nq = neg & quiet & weld
        s = w["score_ens"].to_numpy()
        duty, welds = w["weld_duty_10min_mean"].to_numpy(), w["welds_10min_mean"].to_numpy()
        rows = []
        log(f"\n== {tag}: horizon x window state (share of windows | ensemble AUROC vs quiet welding > 24 h) ==")
        log(f"negatives: {nq.sum():,} windows, weld duty {duty[nq].mean():.2f}, held by gate "
            f"{(~weld[neg & quiet]).mean():.2f}, alarm {w['alarm'].to_numpy()[nq].mean():.4f}")
        log("bin      terminal      other_err     after_err     quiet         | quiet: duty  held  alarm")
        for lab, pos in zip(LABELS, bins):
            r = {"bin": lab, "n": int(pos.sum())}
            cells = []
            for k, m in state.items():
                share = float(m[pos].mean()) if pos.any() else np.nan
                a = auc(s, pos & m & weld, nq) if (pos & m & weld).sum() >= 5 else np.nan
                r[k] = {"share": share, "auc": a}
                cells.append(f"{share:.2f} {fmt(a)}")
            q = pos & quiet
            r["quiet_duty"] = float(duty[q].mean()) if q.any() else np.nan
            r["quiet_held"] = float((~weld[q]).mean()) if q.any() else np.nan
            r["quiet_alarm"] = float(w["alarm"].to_numpy()[q & weld].mean()) if (q & weld).any() else np.nan
            rows.append(r)
            log(f"{lab:7s} " + "    ".join(cells) + f" | {r['quiet_duty']:.2f}  {r['quiet_held']:.2f}  "
                f"{r['quiet_alarm']:.3f}")
        out[tag] = {"neg_duty": float(duty[nq].mean()), "neg_welds10": float(welds[nq].mean()), "rows": rows}

    # when do the codes start? first window (from the end backwards) of the last terminal / other-code episode
    w = pd.concat([cv, te], ignore_index=True)
    onset = {"terminal": [], "other_err_last_1h": []}
    for f, g in w.groupby("file"):
        g = g.sort_values("ttf_s")
        t = g["terminal_any"].to_numpy() > 0
        if t[: 20].any():  # a terminal code within the last ~20 windows
            i = np.flatnonzero(t[:20])[0]
            j = i
            while j + 1 < len(t) and t[j + 1]:
                j += 1
            onset["terminal"].append(float(g["ttf_s"].to_numpy()[j] / 60))
        o = (g["error_active"].to_numpy() > 0) & ~t & (g["ttf_s"].to_numpy() <= 3600)
        onset["other_err_last_1h"].append(bool(o.any()))
    tm = np.array(onset["terminal"])
    out["terminal_onset_min"] = {"guns": len(tm), "median": float(np.median(tm)), "q10_q90": np.quantile(tm, [.1, .9]).tolist(),
                                 "share_gt_15min": float((tm > 15).mean())}
    out["guns_with_other_code_last_1h"] = float(np.mean(onset["other_err_last_1h"]))
    log(f"\nterminal-code episode start before the failure: {len(tm)} guns, median {np.median(tm):.1f} min, "
        f"q10-q90 {np.quantile(tm, .1):.1f}-{np.quantile(tm, .9):.1f} min, > 15 min {np.mean(tm > 15):.0%}; "
        f"guns with another error code in the last 1 h {out['guns_with_other_code_last_1h']:.0%}")
    dst = os.path.join(HERE, "horizon_followup.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"saved {dst}")


def main():
    t0 = time.time()
    cv, te, cols, folds = scored()

    # sanity: the production auroc_pre_rule definition on the OOF scores
    pre = (cv["label"] == 1).to_numpy()
    normal = ((cv["label"] == 0) & (cv["error_active"] == 0)).to_numpy()
    sel = (pre & (cv["ttf_s"].to_numpy() > T.RULE_LEAD_S)) | normal
    fold_pre = [auc(cv["score_ens"].to_numpy(), pre & sel & (cv["fold"] == k).to_numpy(),
                    normal & (cv["fold"] == k).to_numpy()) for k in range(K)]
    log(f"sanity: auroc_pre_rule per fold {np.round(fold_pre, 3).tolist()} mean {np.mean(fold_pre):.3f} "
        f"(train.py --cv 4: 0.644 +- 0.060)")

    out = {"bins": LABELS, "neg_min_h": NEG_MIN_S / 3600, "placebo_anchors_h": PLACEBO_H,
           "sanity_auroc_pre_rule_folds": fold_pre}
    for tag, d in (("cv", cv), ("test", te)):
        rows, neg_alarm = horizon_table(d)
        null = placebo_null(d)
        prow = null[PLACEBO_MAIN_H]["rows"]
        out[tag] = {"rows": rows, "neg_alarm_rate": neg_alarm,
                    "placebo": {a: {"neg_alarm_rate": v["neg_alarm"],
                                    "rows": [{k: r[k] for k in r if not k.startswith("auc_gun") or k.endswith("mean")}
                                             for r in v["rows"]]} for a, v in null.items()},
                    "gun_excess": gun_excess(rows, null), "gun_excess_clean": gun_excess(rows, null, "auc_gun_clean")}
        log(f"\n== {tag}: horizon table (negatives: error-free > 24 h, alarm rate {neg_alarm:.4f}) ==")
        log("bin      n_pos  err  term  held duty | ens   IF    LGBM  gun  | clean ens gun | alarm | placebo72 ens gun "
            "| placebo pooled ens median [q05,q95] over 15 anchors")
        for i, r in enumerate(rows):
            pv = np.array([null[a]["rows"][i]["auc_ens"] for a in PLACEBO_H], float)
            log(f"{r['bin']:7s} {r['n_pos']:6d} {r['error_state']:.2f} {r['terminal_code']:.2f} "
                f"{r['non_welding_held']:.2f} {r['weld_duty']:.2f} | {fmt(r['auc_ens'])} {fmt(r['auc_iforest'])} "
                f"{fmt(r['auc_lgbm'])} {fmt(r['auc_gun_mean'])} | {fmt(r['auc_ens_clean'])} "
                f"{fmt(r['auc_gun_clean_mean'])} | {r['alarm_rate']:.3f} | {fmt(prow[i]['auc_ens'])} "
                f"{fmt(prow[i]['auc_gun_mean'])} | {np.nanmedian(pv):.3f} [{np.nanquantile(pv, .05):.3f}, "
                f"{np.nanquantile(pv, .95):.3f}]")
        log("guns beating all 15 placebo anchors (all windows / clean): " + ", ".join(
            f"{a['bin']} {a['beat_all_placebos']}/{a['guns']} ({b['beat_all_placebos']}/{b['guns']}, "
            f"exp {a['expected']})" for a, b in zip(out[tag]["gun_excess"], out[tag]["gun_excess_clean"])))

    log("\n== horizon-specific LightGBM (4-fold, clean windows; real vs placebo anchor 72 h) ==")
    out["horizon_lgbm"] = {"real": horizon_lgbm(cv, cols, folds),
                           "placebo72": horizon_lgbm(cv, cols, folds, PLACEBO_MAIN_H * 3600)}

    log("\n== within-gun feature effects (clean windows, 64 CV guns + 8 test) ==")
    allw = pd.concat([cv, te], ignore_index=True)
    eff, effp = effects(allw, cols), effects(allw, cols, PLACEBO_MAIN_H * 3600)
    out["effects"], out["effects_placebo72"] = eff, effp
    for lab in LABELS:
        real = top_effects(eff, lab)
        pmax = max(abs(v) for v in effp[lab]["median_d"].values())
        log(f"{lab:7s} ({eff[lab]['n_guns']} guns): " + ", ".join(
            f"{k} {v:+.2f} ({eff[lab]['sign_agreement'][k]:.0%})" for k, v in real)
            + f" | placebo max |d| {pmax:.2f}")

    # per class (CV): ensemble pooled AUROC per horizon, real vs placebo 72 h
    out["per_class"] = {}
    for c in T.CLASSES:
        d = cv[cv["class"] == c]
        rr, _ = horizon_table(d, per_gun=False)
        pp, _ = horizon_table(d, PLACEBO_MAIN_H * 3600, per_gun=False)
        out["per_class"][c] = {"real": [r["auc_ens"] for r in rr], "placebo72": [r["auc_ens"] for r in pp],
                               "real_clean": [r["auc_ens_clean"] for r in rr]}
        log(f"{c}: real " + " ".join(fmt(r["auc_ens"]) for r in rr) + " | clean "
            + " ".join(fmt(r["auc_ens_clean"]) for r in rr) + " | placebo72 " + " ".join(fmt(r["auc_ens"]) for r in pp))

    out["seconds"] = round(time.time() - t0)
    dst = os.path.join(HERE, "horizon_cv.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log(f"saved {dst} in {out['seconds']}s")


if __name__ == "__main__":
    followup() if sys.argv[1:] == ["followup"] else main()
