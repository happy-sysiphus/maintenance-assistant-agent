"""
B-14 (2026-10-06): is predictive maintenance (hours ahead) feasible with this data set at all? Three checks the
earlier experiments (B-2 .. B-13) did not make. Each is a sub-command; run them one at a time (~1 GB free RAM).

    power    DETECTION LIMIT. Inject a step of d within-gun sd (0, 0.25, 0.5, 1.0) into k features (1 or 5) of the
             clean pre-failure windows of one horizon (1-3 h, 3-6 h, 6-24 h) and train the horizon-specific LightGBM
             (4-fold by gun, as B-11). The smallest d it picks up is the size of signal this pipeline would have found
             - the real data sit at d = 0, so "no signal" means "no signal of at least that size".
    events   EVENT-ANCHORED PRE-SIGNAL. Use every code onset as an event instead of the file end: terminal-code onsets
             (hits + false triggers), false triggers alone (they happen mid-file, so elapsed time does not confound
             them) and other-code onsets (E003 ...). Windows 10 min-6 h before an event vs windows > 24 h from the
             next event: production score AUROC + horizon-specific LightGBM, against pseudo events (same guns, shifted
             24-48 h). Plus a "watch" rule (other-code onset -> failure within 10-60 min?) and the hit / false
             trigger split from the preceding hour.
    raw      1 Hz DISTRIBUTION FEATURES BY HORIZON. B-10's raw60 cache (quantiles, ranges, press intervals of 9
             signals, welding rows only), gun-centred on the warm-up, through the B-11 horizon bins: horizon-specific
             LightGBM real vs placebo (anchor 72 h), base 46 vs base + raw, and within-gun effect sizes vs placebo.

    python results/B-14/feasibility.py power|events|raw     -> results/B-14/<cmd>.json (+ <cmd>.log via shell)

Inputs: results/B-11/cache/scores_{cv,test}.parquet (production windows + out-of-fold / bundle scores),
results/B-10/cache/raw60_*.parquet. Never writes outside results/B-14/.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, ".py"))
sys.path.insert(0, os.path.join(ROOT, "results", "B-11"))
import horizon as H  # noqa: E402
import train as T  # noqa: E402

warnings.filterwarnings("ignore")
B11 = os.path.join(ROOT, "results", "B-11", "cache")
B10 = os.path.join(ROOT, "results", "B-10", "cache")
SEED = 42
HR = 3600
META = ["file", "time", "ttf_s", "label", "error_active", "terminal_any", "terminal_idx", "non_welding", "warmup",
        "class", "score_ens"]
NEG_SUB = 120_000


def log(msg):
    print(msg, flush=True)


def save(name, out):
    dst = os.path.join(HERE, f"{name}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log(f"saved {dst}")


def feature_cols():
    meta = json.load(open(os.path.join(ROOT, "results", "B-2", "cache", "meta60.json"), encoding="utf-8"))
    return T.model_input_columns([c for c in meta["feature_cols"] if c != "c19"])


def load(splits=("cv",), extra=()):
    """Scored production windows, sorted by (file, time), floats as float32."""
    cols = feature_cols()
    parts = []
    for s in splits:
        x = pd.read_parquet(os.path.join(B11, f"scores_{s}.parquet"), columns=META + cols + list(extra))
        x["split"] = s
        parts.append(x)
    w = pd.concat(parts, ignore_index=True).sort_values(["file", "time"]).reset_index(drop=True)
    f64 = w.select_dtypes("float64").columns
    w[f64] = w[f64].astype(np.float32)
    w["file"] = w["file"].astype(str)
    return w, cols


def gun_folds(files):
    """train.cv_folds (class-stratified) for the E0x guns, test_N guns dealt round-robin on top."""
    names = sorted(set(files))
    folds = T.cv_folds([f for f in names if not f.startswith("test_")], H.K, SEED)
    for i, f in enumerate(f for f in names if f.startswith("test_")):
        folds[i % H.K].append(f)
    return folds


def clean(w):
    return ((w["error_active"].to_numpy() == 0) & (w["non_welding"].to_numpy() <= H.GATE)
            & (w["warmup"].to_numpy() == 0))


def bin_lgbm(X, files, pos, neg, folds, seed=SEED):
    """4-fold (by gun) LightGBM, positives vs negatives -> pooled and per-gun-mean out-of-fold AUROC."""
    rng = np.random.default_rng(seed)
    oof = np.full(len(files), np.nan)
    for va in folds:
        te = np.isin(files, va)
        ni, pi = np.flatnonzero(neg & ~te), np.flatnonzero(pos & ~te)
        if len(pi) < 5:
            continue
        if len(ni) > NEG_SUB:
            ni = rng.choice(ni, NEG_SUB, replace=False)
        clf = T.lgbm_classifier(len(ni), len(pi), seed)
        clf.set_params(n_estimators=200, n_jobs=4)
        clf.fit(np.vstack([X[ni], X[pi]]), np.r_[np.zeros(len(ni)), np.ones(len(pi))])
        m = te & (pos | neg)
        oof[m] = clf.predict_proba(X[m])[:, 1]
    ok = ~np.isnan(oof)
    pooled = H.auc(oof, pos & ok, neg & ok)
    g = []
    for f in np.unique(files):
        m = (files == f) & ok
        if (pos & m).sum() >= 2 and (neg & m).sum() >= 20:
            g.append(H.auc(oof, pos & m, neg & m))
    return {"auc": pooled, "auc_gun_mean": float(np.nanmean(g)) if g else np.nan, "n_pos": int(pos.sum()),
            "n_guns": len(g)}


def gun_sd(X, files, neg):
    """Per-gun robust sd of each column over the gun's negatives -> (n_rows, n_cols) array aligned with X."""
    out = np.ones_like(X)
    for f in np.unique(files):
        rows = files == f
        n = rows & neg
        if n.sum() < 20:
            continue
        q75, q25 = np.percentile(X[n], [75, 25], axis=0)
        out[rows] = np.maximum((q75 - q25) / 1.349, X[n].std(axis=0) * 0.1 + 1e-6)
    return out


# ------------------------------------------------------------------ power
POWER_BINS = {"1-3h": (HR, 3 * HR), "3-6h": (3 * HR, 6 * HR), "6-24h": (6 * HR, 24 * HR)}
INJECT = {1: ["c5_mean"], 5: ["c1_mean", "c2_mean", "c3_mean", "c5_mean", "c6_mean"]}
DS = [0.25, 0.5, 1.0]


def power():
    t0 = time.time()
    w, cols = load()
    files = w["file"].to_numpy()
    folds = gun_folds(files)
    X0 = w[cols].to_numpy(np.float32)
    ttf, ok = w["ttf_s"].to_numpy(), clean(w)
    neg = ok & (ttf > 24 * HR)
    sd = gun_sd(X0, files, neg)
    out = {"bins": list(POWER_BINS), "d": [0.0] + DS, "inject": INJECT, "rows": []}
    log(f"{len(w):,} windows, {len(set(files))} guns, negatives {neg.sum():,}")
    for b, (lo, hi) in POWER_BINS.items():
        pos = ok & (ttf > lo) & (ttf <= hi)
        base = bin_lgbm(X0, files, pos, neg, folds)
        tp = ttf - 72 * HR
        plc = bin_lgbm(X0, files, ok & (tp > lo) & (tp <= hi), ok & (tp > 24 * HR), folds)
        out["rows"].append({"bin": b, "k": 0, "d": 0.0, **base, "placebo72": plc["auc"]})
        log(f"{b}: d=0 AUROC {base['auc']:.3f} (per gun {base['auc_gun_mean']:.3f}, {base['n_pos']} pos) | "
            f"placebo72 {plc['auc']:.3f}")
        for k, feats in INJECT.items():
            ci = [cols.index(c) for c in feats]
            for d in DS:
                X = X0.copy()
                X[np.ix_(np.flatnonzero(pos), ci)] += d * sd[np.ix_(np.flatnonzero(pos), ci)]
                r = bin_lgbm(X, files, pos, neg, folds)
                del X
                out["rows"].append({"bin": b, "k": k, "d": d, **r, "gain": r["auc"] - base["auc"]})
                log(f"  k={k} d={d:.2f}: AUROC {r['auc']:.3f} (+{r['auc'] - base['auc']:.3f}) "
                    f"per gun {r['auc_gun_mean']:.3f}  {time.time() - t0:.0f}s")
    out["seconds"] = round(time.time() - t0)
    save("power", out)


# ------------------------------------------------------------------ events
EV_BINS = {"10-60m": (600, HR), "1-3h": (HR, 3 * HR), "3-6h": (3 * HR, 6 * HR)}
N_PSEUDO = 20


def onsets(flag, ttf):
    """Chronological per-window flag -> ttf of the episode starts, 30 min cooldown (train.rule_triggers)."""
    return ttf[T.rule_triggers(flag, ttf)]


def gun_events(w):
    """file -> dict of event ttf arrays (seconds before the file's failure)."""
    ev = {}
    for f, idx in w.groupby("file").indices.items():
        ttf = w["ttf_s"].to_numpy()[idx]
        term = w["terminal_any"].to_numpy()[idx] > 0
        other = (w["error_active"].to_numpy()[idx] > 0) & ~term
        t_on = onsets(term, ttf)
        ev[f] = {"terminal_all": t_on, "terminal_false": t_on[t_on > T.RULE_EVAL_WINDOW_S],
                 "terminal_hit": t_on[t_on <= T.RULE_EVAL_WINDOW_S], "other_codes": onsets(other, ttf)}
    return ev


def time_to_next(ttf, ev_ttf):
    """For windows at ttf (seconds before failure), seconds until the next event (events have smaller ttf)."""
    if len(ev_ttf) == 0:
        return np.full(len(ttf), np.inf)
    e = np.sort(ev_ttf)  # ascending ttf = latest event first
    i = np.searchsorted(e, ttf, side="right") - 1  # largest event ttf <= window ttf
    out = np.full(len(ttf), np.inf)
    ok = i >= 0
    out[ok] = ttf[ok] - e[i[ok]]
    return out


def time_since_any(ttf, all_ev):
    if len(all_ev) == 0:
        return np.full(len(ttf), np.inf)
    e = np.sort(all_ev)
    i = np.searchsorted(e, ttf, side="left")  # smallest event ttf >= window ttf = the last event before it
    out = np.full(len(ttf), np.inf)
    ok = i < len(e)
    out[ok] = e[i[ok]] - ttf[ok]
    return out


def event_sets(w, ev, kind, shift=None, rng=None):
    """tte (time to next event of `kind`) and tsa (time since any code event) per window; shift moves every event
    earlier by a random 24-48 h (pseudo events), kept inside the file and > 6 h from the real ones."""
    tte = np.full(len(w), np.inf)
    tsa = np.full(len(w), np.inf)
    ttf_all = w["ttf_s"].to_numpy()
    for f, idx in w.groupby("file").indices.items():
        e = ev[f][kind]
        real_all = np.concatenate([ev[f]["terminal_all"], ev[f]["other_codes"]])
        if shift is not None and len(e):
            span = ttf_all[idx].max()
            p = e + rng.uniform(24 * HR, 48 * HR, len(e))
            far = np.array([np.min(np.abs(real_all - x)) > 6 * HR if len(real_all) else True for x in p])
            e = p[(p < span - 6 * HR) & far]
        tte[idx] = time_to_next(ttf_all[idx], e)
        tsa[idx] = time_since_any(ttf_all[idx], real_all)
    return tte, tsa


def events():
    t0 = time.time()
    w, cols = load(("cv", "test"))
    files = w["file"].to_numpy()
    folds = gun_folds(files)
    X = w[cols].to_numpy(np.float32)
    s = w["score_ens"].to_numpy()
    ok = clean(w)
    ev = gun_events(w)
    counts = {k: int(sum(len(v[k]) for v in ev.values())) for k in ("terminal_all", "terminal_false", "terminal_hit",
                                                                       "other_codes")}
    log(f"{len(w):,} windows, {len(set(files))} guns; events {counts}")
    out = {"counts": counts, "bins": list(EV_BINS), "kinds": {}}
    rng = np.random.default_rng(SEED)
    for kind in ("terminal_all", "terminal_false", "other_codes"):
        tte, tsa = event_sets(w, ev, kind)
        base = ok & (tsa > HR)
        neg = base & (tte > 24 * HR)
        pseudo = [event_sets(w, ev, kind, shift=True, rng=rng) for _ in range(N_PSEUDO)]
        rows = []
        for b, (lo, hi) in EV_BINS.items():
            pos = base & (tte > lo) & (tte <= hi)
            r = {"bin": b, "n_pos": int(pos.sum()), "score_auc": H.auc(s, pos, neg)}
            pv = []
            for p_tte, p_tsa in pseudo:
                pb = ok & (p_tsa > HR)
                pv.append(H.auc(s, pb & (p_tte > lo) & (p_tte <= hi), pb & (p_tte > 24 * HR)))
            pv = np.array(pv, float)
            r["score_pseudo_median"], r["score_pseudo_q95"] = float(np.nanmedian(pv)), float(np.nanquantile(pv, .95))
            r["lgbm"] = bin_lgbm(X, files, pos, neg, folds)
            p_tte, p_tsa = pseudo[0]
            pb = ok & (p_tsa > HR)
            r["lgbm_pseudo"] = bin_lgbm(X, files, pb & (p_tte > lo) & (p_tte <= hi), pb & (p_tte > 24 * HR), folds)
            rows.append(r)
            log(f"{kind:15s} {b:6s} pos {r['n_pos']:6d} | score {r['score_auc']:.3f} (pseudo {r['score_pseudo_median']:.3f},"
                f" q95 {r['score_pseudo_q95']:.3f}) | LGBM {r['lgbm']['auc']:.3f} (pseudo {r['lgbm_pseudo']['auc']:.3f})"
                f"  {time.time() - t0:.0f}s")
        out["kinds"][kind] = rows

    # watch rule: an other-code onset (30 min cooldown, no terminal code yet) -> failure 10-60 min later?
    wr = {"guns": 0, "alerts": 0, "alerts_hit": 0, "guns_hit": 0, "days": 0.0}
    for f, e in ev.items():
        a = e["other_codes"]
        span_d = w.loc[w["file"] == f, "ttf_s"].max() / 86400
        hit = (a > 600) & (a <= HR)
        wr["guns"] += 1
        wr["alerts"] += int(len(a))
        wr["alerts_hit"] += int(hit.sum())
        wr["guns_hit"] += int(hit.any())
        wr["days"] += float(span_d)
    wr["alerts_per_gun_day"] = wr["alerts"] / wr["days"]
    wr["precision"] = wr["alerts_hit"] / max(wr["alerts"], 1)
    wr["recall_guns"] = wr["guns_hit"] / wr["guns"]
    wr["precision_by_chance"] = 50 / (wr["days"] / wr["guns"] * 1440)  # a 50 min target window in a ~7 day file
    out["watch_rule"] = wr
    log(f"watch rule (other-code onset -> failure in 10-60 min): {wr['alerts']} alerts, "
        f"{wr['alerts_per_gun_day']:.2f}/gun/day, precision {wr['precision']:.3f} (chance {wr['precision_by_chance']:.4f}),"
        f" guns with a hit {wr['guns_hit']}/{wr['guns']}")

    # hit vs false terminal onset from the hour before it (B-5: 0.62); elapsed time shown as the confound
    feats, y, grp, elapsed = [], [], [], []
    for f, idx in w.groupby("file").indices.items():
        ttf = w["ttf_s"].to_numpy()[idx]
        for e in ev[f]["terminal_all"]:
            m = (ttf > e) & (ttf <= e + HR)
            if m.sum() < 10:
                continue
            ii = idx[m]
            other = ((w["error_active"].to_numpy()[ii] > 0) & (w["terminal_any"].to_numpy()[ii] == 0)).mean()
            feats.append(np.r_[X[ii].mean(0), s[ii].mean(), s[ii].max(), other])
            y.append(int(e <= T.RULE_EVAL_WINDOW_S))
            grp.append(f)
            elapsed.append(ttf.max() - e)
    F, y, grp, elapsed = np.array(feats), np.array(y), np.array(grp), np.array(elapsed)[:, None]
    gfolds = gun_folds(grp)

    def lr_auc(Z):
        p = np.full(len(y), np.nan)
        for va in gfolds:
            te = np.isin(grp, va)
            if len(set(y[~te])) < 2:
                continue
            sc = StandardScaler().fit(Z[~te])
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Z[~te]), y[~te])
            p[te] = clf.predict_proba(sc.transform(Z[te]))[:, 1]
        k = ~np.isnan(p)
        return H.auc(p, (y == 1) & k, (y == 0) & k)

    out["hit_vs_false"] = {"n_hit": int(y.sum()), "n_false": int((y == 0).sum()), "auc_features": lr_auc(F),
                           "auc_elapsed_time_only": lr_auc(elapsed)}
    log(f"hit vs false terminal onset (preceding 1 h, LR, 4-fold by gun): {y.sum()} hits / {(y == 0).sum()} false, "
        f"AUROC features {out['hit_vs_false']['auc_features']:.3f}, elapsed time alone "
        f"{out['hit_vs_false']['auc_elapsed_time_only']:.3f}")
    out["seconds"] = round(time.time() - t0)
    save("events", out)


# ------------------------------------------------------------------ raw
RAW_BINS = ["10-30m", "30-60m", "1-3h", "3-6h", "6-12h", "12-24h"]


def raw():
    t0 = time.time()
    w, cols = load()
    r = pd.read_parquet(os.path.join(B10, "raw60_train.parquet")).drop(columns=["seg"])
    r["file"] = r["file"].astype(str)
    rcols = [c for c in r.columns if c not in ("file", "time")]
    r[rcols] = r[rcols].astype(np.float32)
    r = r.drop_duplicates(["file", "time"], keep="last")
    w = w.merge(r, on=["file", "time"], how="left")
    del r
    # gun-centre on the warm-up (as the production features), carry welding gaps forward, then 0
    w[rcols] = w.groupby("file", sort=False)[rcols].ffill()
    wu = w[w["warmup"] > 0].groupby("file")[rcols].mean()
    w[rcols] = (w[rcols] - wu.reindex(w["file"]).to_numpy()).fillna(0.0).astype(np.float32)
    log(f"{len(w):,} windows, {len(rcols)} raw features, {time.time() - t0:.0f}s")
    files = w["file"].to_numpy()
    folds = gun_folds(files)
    ok = clean(w)
    # one matrix, two views (base | raw): the frame keeps only what sets() / effects() read
    Xc = np.hstack([w[cols].to_numpy(np.float32), w[rcols].to_numpy(np.float32)])
    Xb, Xr = Xc[:, :len(cols)], Xc[:, len(cols):]
    eff_cols = rcols
    w = w.drop(columns=[c for c in cols if c != "weld_duty_10min_mean"])
    out = {"raw_features": rcols, "rows": []}
    for shift in (0, 72 * HR):
        bins, neg = H.sets(w, shift)
        for lab, pos in zip(H.LABELS, bins):
            if lab not in RAW_BINS:
                continue
            row = {"bin": lab, "placebo": bool(shift),
                   "raw": bin_lgbm(Xr, files, pos & ok, neg & ok, folds),
                   "base": bin_lgbm(Xb, files, pos & ok, neg & ok, folds)}
            row["base_raw"] = bin_lgbm(Xc, files, pos & ok, neg & ok, folds)
            out["rows"].append(row)
            log(f"{'placebo72' if shift else 'real':9s} {lab:7s} raw {row['raw']['auc']:.3f}  base {row['base']['auc']:.3f}"
                f"  base+raw {row['base_raw']['auc']:.3f}  ({row['raw']['n_pos']} pos)  {time.time() - t0:.0f}s")
    del Xc, Xb, Xr
    eff, effp = H.effects(w, eff_cols), H.effects(w, eff_cols, 72 * HR)
    out["effects"], out["effects_placebo72"] = eff, effp
    for lab in RAW_BINS:
        top = sorted(eff[lab]["median_d"].items(), key=lambda kv: -abs(kv[1]))[:5]
        log(f"effects {lab:7s}: " + ", ".join(
            f"{k} {v:+.2f}/plc {effp[lab]['median_d'][k]:+.2f} ({eff[lab]['sign_agreement'][k]:.0%})" for k, v in top))
    out["seconds"] = round(time.time() - t0)
    save("raw", out)


def events_control():
    """Is the other-code pre-signal (events: LightGBM 0.76-0.88) about the codes, or about the file end? Other codes
    pile up in the last hours before the failure (B-11: 15 % of windows 10-60 min out vs 3 %), so 'before an event'
    may just mean 'late in the file'. Controls: (1) elapsed time alone as the score, within gun; (2) only events
    > 30 h before the failure (and windows > 30 h out), where the file end is far."""
    t0 = time.time()
    w, cols = load(("cv", "test"))
    files = w["file"].to_numpy()
    folds = gun_folds(files)
    X = w[cols].to_numpy(np.float32)
    ttf = w["ttf_s"].to_numpy()
    elapsed = np.zeros(len(w))
    for _, idx in w.groupby("file").indices.items():
        elapsed[idx] = ttf[idx].max() - ttf[idx]
    ok = clean(w)
    ev = gun_events(w)
    out = {}
    for tag, min_ttf in (("all", 0), ("far_from_failure", 30 * HR)):
        ev2 = {f: {**e, "other_codes": e["other_codes"][e["other_codes"] > min_ttf]} for f, e in ev.items()}
        tte, tsa = event_sets(w, ev2, "other_codes")
        base = ok & (tsa > HR) & (ttf > min_ttf)
        neg = base & (tte > 24 * HR)
        rows = []
        for b, (lo, hi) in EV_BINS.items():
            pos = base & (tte > lo) & (tte <= hi)
            g = [H.auc(elapsed, pos & (files == f), neg & (files == f)) for f in np.unique(files)
                 if (pos & (files == f)).sum() >= 2 and (neg & (files == f)).sum() >= 20]
            r = {"bin": b, "n_events": int(sum(len(e["other_codes"]) for e in ev2.values())), "n_pos": int(pos.sum()),
                 "elapsed_auc_pooled": H.auc(elapsed, pos, neg), "elapsed_auc_gun_mean": float(np.nanmean(g)) if g else np.nan,
                 "lgbm": bin_lgbm(X, files, pos, neg, folds)}
            rows.append(r)
            log(f"other_codes [{tag}] {b:6s} events {r['n_events']} pos {r['n_pos']:6d} | elapsed-time AUROC "
                f"{r['elapsed_auc_pooled']:.3f} (per gun {r['elapsed_auc_gun_mean']:.3f}) | LGBM {r['lgbm']['auc']:.3f} "
                f"(per gun {r['lgbm']['auc_gun_mean']:.3f}, {r['lgbm']['n_guns']} guns)  {time.time() - t0:.0f}s")
        out[tag] = rows
    save("events_control", out)


def events_quiet():
    """Other-code onsets far from the failure stay predictable (events_control: LightGBM 0.87-0.91, elapsed time
    0.47-0.50). Is that a precursor, or the previous episode's aftermath (codes come in bursts: an onset > 1 h ago
    can be an episode that ended minutes ago)? Keep only windows with no error state at all in the preceding Q hours,
    on both sides, and show what the model uses (gain, all-gun fit, 1-3 h)."""
    t0 = time.time()
    w, cols = load(("cv", "test"))
    files = w["file"].to_numpy()
    folds = gun_folds(files)
    X = w[cols].to_numpy(np.float32)
    ttf = w["ttf_s"].to_numpy()
    err = w["error_active"].to_numpy() > 0
    since_err = np.full(len(w), np.inf)
    for _, idx in w.groupby("file").indices.items():
        since_err[idx] = time_since_any(ttf[idx], ttf[idx][err[idx]])  # the window itself is error-free (clean)
    ok = clean(w)
    ev = gun_events(w)
    ev2 = {f: {**e, "other_codes": e["other_codes"][e["other_codes"] > 30 * HR]} for f, e in ev.items()}
    tte, tsa = event_sets(w, ev2, "other_codes")
    out = {}
    for q in (0, HR, 6 * HR):
        base = ok & (tsa > HR) & (ttf > 30 * HR) & (since_err > q)
        neg = base & (tte > 24 * HR)
        rows = []
        for b, (lo, hi) in EV_BINS.items():
            pos = base & (tte > lo) & (tte <= hi)
            r = {"bin": b, "quiet_h": q / HR, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                 "lgbm": bin_lgbm(X, files, pos, neg, folds)}
            rows.append(r)
            log(f"other_codes far, no error in prior {q / HR:.0f} h, {b:6s}: pos {r['n_pos']:6d} neg {r['n_neg']:7d} | "
                f"LGBM {r['lgbm']['auc']:.3f} (per gun {r['lgbm']['auc_gun_mean']:.3f}, {r['lgbm']['n_guns']} guns)"
                f"  {time.time() - t0:.0f}s")
        out[f"quiet_{q // HR}h"] = rows
    # what does the model use? all-gun fit on the 1-3 h bin without the quiet filter
    base = ok & (tsa > HR) & (ttf > 30 * HR)
    pos, neg = base & (tte > HR) & (tte <= 3 * HR), base & (tte > 24 * HR)
    ni, pi = np.flatnonzero(neg), np.flatnonzero(pos)
    if len(ni) > NEG_SUB:
        ni = np.random.default_rng(SEED).choice(ni, NEG_SUB, replace=False)
    clf = T.lgbm_classifier(len(ni), len(pi), SEED).set_params(n_estimators=200, n_jobs=4, importance_type="gain")
    clf.fit(np.vstack([X[ni], X[pi]]), np.r_[np.zeros(len(ni)), np.ones(len(pi))])
    imp = sorted(zip(cols, clf.feature_importances_ / clf.feature_importances_.sum()), key=lambda kv: -kv[1])[:10]
    out["gain_top10_1_3h"] = [(k, round(float(v), 3)) for k, v in imp]
    log("gain top 10 (1-3 h): " + ", ".join(f"{k} {v:.2f}" for k, v in imp))
    save("events_quiet", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"power": power, "events": events, "events_control": events_control, "events_quiet": events_quiet,
     "raw": raw}.get(cmd, lambda: sys.exit(__doc__))()
