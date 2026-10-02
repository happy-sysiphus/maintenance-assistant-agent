"""
B-6 (2026-09-28): the paper's own task, and whether an error-code history adds to ours.

The paper (Wang et al., Sci Data 2024, Table 12) forecasts the next N = 60 s and classifies that horizon as a
working state: normal or one of the four faults. It reports average accuracy 68.62 %, recall normal 0.4431 (56 %
of normal states flagged as error), recall error 0.7431, per-fault recall E01 0.358 / E02 0.195 / E03 0.239 /
E04 0.184. Our own metrics (AUROC of "failure within 1 h") are a different question, so this script puts
our pipeline on the paper's protocol:

    sample  = one 60 s window at time t (the same per-gun-centred windows as train.py, experiments.py cache)
    target  = state of the window at t + h (h = 1 -> the paper's 60 s): E01..E04 if that class's terminal code
              (E012 / E016 / E028 / E029) is present in it, else normal. Samples whose t + h window is missing
              (gap / end of file) are dropped.
    input   = only what exists at t: the 46 honest window features (c19 dropped) and, in the "+codes" setting,
              the error-code history (share of every code in the window and over the past 10 / 60 / 360 windows,
              minutes since each terminal code and since any error, capped at 6 h, length of the current terminal
              state). The "sensor" setting drops every error-derived input (error_share_10min too).
    model   = LightGBM, 5 classes, balanced class weights. Gun-level 4-fold over the 64 train guns (train.cv_folds,
              same seed as B-5), then a model on all 64 guns scores the 8 test guns.

A state that lasts for hours makes this task easy for a copy of the present: the PERSISTENCE baseline (next
state = current state) is reported next to every model, and the metrics are also given for the ONSET subset -
current window normal, target window in a fault state - where a prediction is actually needed (threshold-free:
`onset_auroc` = AUROC of P(fault) over every sample whose current window is normal).

Second question (`fail1h`): does the code history lift the 1-h failure task that the production model answers?
Same gun-level 4-fold as B-5 (experiments.cv_run, LightGBM on `label`), with and without the code features;
the gate is auroc_pre_rule (techspec B-9).

    python .py/paper_task.py cache          # per-window error-code shares -> results/B-6/cache (~3 min)
    python .py/paper_task.py run            # paper task h = 1, 10, 60 windows + fail1h -> results/B-6/*.json
    python .py/paper_task.py summary        # markdown tables of the JSON results
"""
import argparse
import glob
import json
import os
import sys
import time

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as X  # noqa: E402
import train as T  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")
OUT = os.path.join(T.PROJECT_ROOT, "results", "B-6")
CACHE = os.path.join(OUT, "cache")
WINDOW = 60
SEED = 42
CODES = ["E001", "E003", "E006", "E007", "E009", "E010", "E011", "E012", "E016", "E028", "E029"]
SPANS = (10, 60, 360)  # windows: 10 min, 1 h, 6 h
CAP_MIN = 360  # "minutes since" stops at 6 h: longer lookbacks grow with file age = time to failure (c19-style leak)
STATES = ["normal"] + T.CLASSES
PAPER = {"accuracy": 0.6862, "recall_normal": 0.4431, "recall_error": 0.7431,
         "recall_class": {"E01": 0.3578, "E02": 0.1945, "E03": 0.2385, "E04": 0.1843}}
HYBRID_Q = 0.999  # onset budget: ~0.1 % of normal->normal windows may be called a fault
ERROR_DERIVED = ("error_share_10min_mean", "error_share_10min_std")


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ cache
def code_windows(df):
    """1 Hz rows -> share of every error code per (segment, 60 s window), indexed like train.window_features."""
    d = pd.get_dummies(df["error_code"].astype(str)).reindex(columns=CODES, fill_value=0).astype("float32")
    g = d.groupby([df["segment_id"], pd.Grouper(freq=f"{WINDOW}s")]).mean().droplevel(0)
    g.columns = [f"code_{c}" for c in CODES]
    g["file"] = df["file"].iloc[0]
    return g[~g.index.duplicated(keep="last")]


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    data = os.path.join(T.PROJECT_ROOT, "preprocessed")
    for split, fs in (("train", sorted(glob.glob(os.path.join(data, "E0*.parquet")))),
                      ("test", T.test_files_in(os.path.join(data, "test")))):
        parts = []
        for i, f in enumerate(fs, 1):
            parts.append(code_windows(pd.read_parquet(f, columns=["file", "segment_id", "error_code"])))
            log(f"[{split} {i}/{len(fs)}] {os.path.basename(f)}")
        x = pd.concat(parts)
        x.index.name = "time"
        x.reset_index().to_parquet(os.path.join(CACHE, f"codes{WINDOW}_{split}.parquet"))
    log(f"cached error-code shares in {CACHE}")


# --------------------------------------------------------- features / targets
def code_features(w, codes):
    """Join the code shares onto the windows of one split and derive the history features (per file,
    chronological, past only)."""
    w = w.reset_index().merge(codes, on=["file", "time"], how="left")
    cc = [f"code_{c}" for c in CODES]
    w[cc] = w[cc].fillna(0.0)
    w = w.sort_values(["file", "time"]).reset_index(drop=True)
    g = w.groupby("file", sort=False)
    new = {}
    for c in cc:
        for k in SPANS:
            new[f"{c}_r{k}"] = g[c].transform(lambda s, k=k: s.rolling(k, min_periods=1).mean())
    tmin = (w["time"] - w["time"].min()).dt.total_seconds() / 60.0
    for c in [f"code_{x}" for x in T.TERMINAL_CODES] + ["error_active"]:
        seen = tmin.where(w[c] > 0)
        last = seen.groupby(w["file"]).ffill()
        new[f"min_since_{c}"] = (tmin - last).clip(upper=CAP_MIN).fillna(CAP_MIN)
    state = (w["terminal_idx"] > 0).astype(int)
    run_id = (state != g["terminal_idx"].shift().gt(0).astype(int)).cumsum()
    new["terminal_run_len"] = (state.groupby([w["file"], run_id]).cumcount() + 1).where(state > 0, 0).clip(upper=CAP_MIN)
    return pd.concat([w, pd.DataFrame(new)], axis=1)


def code_cols():
    cc = [f"code_{c}" for c in CODES]
    return (["error_active"] + cc + [f"{c}_r{k}" for c in cc for k in SPANS]
            + [f"min_since_code_{x}" for x in T.TERMINAL_CODES] + ["min_since_error_active", "terminal_run_len"])


def add_target(w, h):
    """y = terminal_idx (0 normal, 1..4 = E01..E04) of the window h steps later in the same file."""
    nxt = w.groupby("file", sort=False)[["time", "terminal_idx"]].shift(-h)
    ok = (nxt["time"] - w["time"]).dt.total_seconds() == h * WINDOW
    out = w[ok.to_numpy()].copy()
    out["y"] = nxt.loc[ok, "terminal_idx"].astype(int).to_numpy()
    out["now"] = out["terminal_idx"].astype(int)
    return out


# ------------------------------------------------------------------ model / metrics
def fit_predict(tr, te, cols):
    import lightgbm as lgb

    m = lgb.LGBMClassifier(objective="multiclass", num_class=5, n_estimators=300, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                           random_state=SEED, n_jobs=8, verbose=-1)
    # a class absent from a fold's training guns would make LightGBM emit fewer columns
    m.fit(tr[cols].to_numpy(np.float32), tr["y"].to_numpy())
    p = np.zeros((len(te), 5))
    p[:, m.classes_] = m.predict_proba(te[cols].to_numpy(np.float32))
    return p, m


def onset_threshold(m, tr, cols, q=HYBRID_Q):
    """P(fault) threshold = q-quantile over the training samples that stay normal (normal now, normal next).
    In-sample, so the realised false-alarm rate on held-out guns is above 1 - q (history.md §14); the onset AUROCs
    do not depend on it."""
    base = tr[(tr["now"] == 0) & (tr["y"] == 0)]
    pf = np.zeros((len(base), 5))
    pf[:, m.classes_] = m.predict_proba(base[cols].to_numpy(np.float32))
    return float(np.quantile(1 - pf[:, 0], q))


def hybrid_predict(p, now, thr):
    """State + onset: a fault state that is on now is carried (persistence = the terminal-code rule's view);
    from a normal state the model predicts a fault only when P(fault) clears the false-alarm budget thr,
    and then the most probable fault class."""
    now = np.asarray(now)
    onset = (now == 0) & (1 - p[:, 0] > thr)
    return np.where(now > 0, now, np.where(onset, p[:, 1:].argmax(1) + 1, 0))


def state_metrics(y, pred, p_fault=None, now=None):
    y, pred = np.asarray(y), np.asarray(pred)
    err, perr = y > 0, pred > 0
    r = {"n": int(len(y)), "n_fault": int(err.sum()),
         "accuracy": float((y == pred).mean()),
         "recall_normal": float((~perr[~err]).mean()) if (~err).any() else np.nan,
         "recall_error": float(perr[err].mean()) if err.any() else np.nan,
         "recall_class": {c: (float((pred[y == i] == i).mean()) if (y == i).any() else None)
                          for i, c in enumerate(T.CLASSES, 1)},
         "n_class": {c: int((y == i).sum()) for i, c in enumerate(T.CLASSES, 1)},
         "macro_f1": float(f1_score(y, pred, labels=range(5), average="macro", zero_division=0))}
    r["false_alarm_rate"] = 1 - r["recall_normal"]
    rc = [v for v in r["recall_class"].values() if v is not None]
    r["balanced_accuracy"] = float(np.mean([r["recall_normal"]] + rc))
    if now is not None:
        on = (np.asarray(now) == 0) & err
        r["onset"] = {"n": int(on.sum()), "recall_error": float(perr[on].mean()) if on.any() else None,
                      "recall_class": float((pred[on] == y[on]).mean()) if on.any() else None}
        if p_fault is not None:
            base = np.asarray(now) == 0
            yy = err[base]
            r["onset"]["auroc"] = float(roc_auc_score(yy, p_fault[base])) if 0 < yy.sum() < len(yy) else None
            # onset recall at a fixed false-alarm budget: threshold = 99th pct of P(fault) on normal->normal samples
            thr = float(np.quantile(p_fault[base & ~err], 0.99))
            r["onset"]["recall_at_fa1pct"] = float((p_fault[on] > thr).mean()) if on.any() else None
    return r


def summarize(rs):
    """mean / sd over folds of the headline numbers."""
    def pick(r):
        return {"accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"], "recall_normal": r["recall_normal"],
                "recall_error": r["recall_error"], "macro_f1": r["macro_f1"],
                **{f"recall_{c}": r["recall_class"][c] for c in T.CLASSES},
                "onset_recall_error": r["onset"]["recall_error"], "onset_auroc": r["onset"].get("auroc"),
                "onset_recall_at_fa1pct": r["onset"].get("recall_at_fa1pct")}
    rows = [pick(r) for r in rs]
    keys = rows[0].keys()
    return {"mean": {k: float(np.nanmean([np.nan if x[k] is None else x[k] for x in rows])) for k in keys},
            "sd": {k: float(np.nanstd([np.nan if x[k] is None else x[k] for x in rows])) for k in keys}}


# ------------------------------------------------------------------ experiments
def load_frames():
    D = X.load(WINDOW)
    ctr = pd.read_parquet(os.path.join(CACHE, f"codes{WINDOW}_train.parquet"))
    cte = pd.read_parquet(os.path.join(CACHE, f"codes{WINDOW}_test.parquet"))
    return D, code_features(D["all"], ctr), code_features(D["te"], cte), ctr, cte


def paper_task(D, tr_all, te_all, h):
    sensor = [c for c in D["model_cols"] if c not in ("c19_mean", "c19_std") and c not in ERROR_DERIVED]
    settings = {"sensor": sensor, "sensor+codes": sensor + list(ERROR_DERIVED) + code_cols()}
    a, t = add_target(tr_all, h), add_target(te_all, h)
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
    folds = T.cv_folds(sorted(glob.glob(os.path.join(T.PROJECT_ROOT, "preprocessed", "E0*.parquet"))), 4, SEED)
    res = {"name": f"paper_h{h}", "horizon_s": h * WINDOW, "paper_table12": PAPER, "n_train_samples": int(len(a)),
           "n_test_samples": int(len(t)), "target_share": {s: float((a["y"] == i).mean()) for i, s in enumerate(STATES)}}
    # persistence baseline
    per_cv = [state_metrics(a.loc[a["file"].isin({stem(f) for f in va}), "y"], a.loc[a["file"].isin({stem(f) for f in va}), "now"],
                            now=a.loc[a["file"].isin({stem(f) for f in va}), "now"]) for va in folds]
    res["persistence"] = {"cv": summarize(per_cv), "test": state_metrics(t["y"], t["now"], now=t["now"])}
    for name, cols in settings.items():
        t0, cv, hy = time.time(), [], []
        for va in folds:
            vs = {stem(f) for f in va}
            tr, va_ = a[~a["file"].isin(vs)], a[a["file"].isin(vs)]
            p, m = fit_predict(tr, va_, cols)
            cv.append(state_metrics(va_["y"], p.argmax(1), 1 - p[:, 0], va_["now"]))
            hy.append(state_metrics(va_["y"], hybrid_predict(p, va_["now"], onset_threshold(m, tr, cols)), 1 - p[:, 0], va_["now"]))
        p, m = fit_predict(a, t, cols)
        res[f"hybrid:{name}"] = {"cv": summarize(hy), "cv_folds": hy, "onset_q": HYBRID_Q,
                                 "test": state_metrics(t["y"], hybrid_predict(p, t["now"], onset_threshold(m, a, cols)),
                                                       1 - p[:, 0], t["now"])}
        gain = m.booster_.feature_importance("gain")
        top = sorted(zip(cols, gain / (gain.sum() or 1)), key=lambda x: -x[1])[:12]
        res[name] = {"n_features": len(cols), "cv": summarize(cv), "cv_folds": cv,
                     "test": state_metrics(t["y"], p.argmax(1), 1 - p[:, 0], t["now"]),
                     "top_features_gain_share": [(c, round(float(g), 3)) for c, g in top], "fit_s": round(time.time() - t0, 1)}
        cvm, te = res[name]["cv"]["mean"], res[name]["test"]
        log(f"== h={h:>3} {name:<13} cv acc {cvm['accuracy']:.4f} bal {cvm['balanced_accuracy']:.3f} "
            f"rec N {cvm['recall_normal']:.3f} E {cvm['recall_error']:.3f} onset AUROC {cvm['onset_auroc']:.3f} "
            f"| test acc {te['accuracy']:.4f} bal {te['balanced_accuracy']:.3f} rec N {te['recall_normal']:.3f} "
            f"E {te['recall_error']:.3f} onset AUROC {te['onset'].get('auroc') or float('nan'):.3f}")
    for name in settings:
        hm, ht = res[f"hybrid:{name}"]["cv"]["mean"], res[f"hybrid:{name}"]["test"]
        log(f"== h={h:>3} hybrid:{name:<13} cv acc {hm['accuracy']:.4f} bal {hm['balanced_accuracy']:.3f} rec N "
            f"{hm['recall_normal']:.4f} onset rec {hm['onset_recall_error']:.3f} | test acc {ht['accuracy']:.4f} bal "
            f"{ht['balanced_accuracy']:.3f} rec N {ht['recall_normal']:.4f} onset rec {ht['onset']['recall_error']:.3f}")
    p = res["persistence"]
    log(f"== h={h:>3} persistence   cv acc {p['cv']['mean']['accuracy']:.4f} bal {p['cv']['mean']['balanced_accuracy']:.3f} "
        f"| test acc {p['test']['accuracy']:.4f} bal {p['test']['balanced_accuracy']:.3f}")
    save(res)
    return res


def fail1h(D, tr_all, te_all, ctr, cte):
    """The production question (failure within 1 h), B-5 gun-level 4-fold LightGBM, with / without code history."""
    D2 = dict(D)
    for key, codes in (("all", ctr), ("cal_all", ctr), ("te", cte), ("cte", cte)):
        f = code_features(D[key], codes).set_index("time")
        D2[key] = f
    base = [c for c in D["model_cols"] if c not in ("c19_mean", "c19_std")]
    out = {}
    for name, cols in (("fail1h_base", base), ("fail1h_codes", base + code_cols())):
        r = X.cv_run(name, D2, cols, "lgbm", note="B-6: LightGBM on label (1 h), gun-level 4-fold")
        out[name] = r
        os.replace(os.path.join(X.OUT3, f"{name}.json"), os.path.join(OUT, f"{name}.json"))
    return out


def save(res):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{res['name']}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def summary():
    rows = ["| horizon | model | split | accuracy | recall normal | recall error | E01 | E02 | E03 | E04 | onset AUROC | "
            "onset recall @FA1% |", "|" + "---|" * 12]
    rows.append(f"| 60 s | paper Table 12 | test | {PAPER['accuracy']:.3f} | {PAPER['recall_normal']:.3f} | "
                f"{PAPER['recall_error']:.3f} | " + " | ".join(f"{PAPER['recall_class'][c]:.3f}" for c in T.CLASSES) + " | - | - |")
    for f in sorted(glob.glob(os.path.join(OUT, "paper_h*.json")), key=lambda p: int(p.split("_h")[1][:-5])):
        r = json.load(open(f, encoding="utf-8"))
        for name in ("persistence", "sensor", "sensor+codes", "hybrid:sensor", "hybrid:sensor+codes"):
            for split in ("cv", "test"):
                if split == "cv":
                    m = r[name]["cv"]["mean"]
                    vals = [m["accuracy"], m["recall_normal"], m["recall_error"]] + [m[f"recall_{c}"] for c in T.CLASSES] + \
                           [m.get("onset_auroc"), m.get("onset_recall_at_fa1pct")]
                else:
                    m = r[name]["test"]
                    vals = [m["accuracy"], m["recall_normal"], m["recall_error"]] + [m["recall_class"][c] for c in T.CLASSES] + \
                           [m["onset"].get("auroc"), m["onset"].get("recall_at_fa1pct")]
                rows.append(f"| {r['horizon_s']} s | {name} | {split} | " + " | ".join(fmt(v) for v in vals) + " |")
    print("\n".join(rows))
    for n in ("fail1h_base", "fail1h_codes"):
        p = os.path.join(OUT, f"{n}.json")
        if os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            v, t = r["val_folds"], r["test_over_folds"]
            print(f"{n}: cv AUROC {v['mean']['auroc']:.3f}+-{v['sd']['auroc']:.3f} pre-rule {v['mean']['auroc_pre_rule']:.3f}"
                  f"+-{v['sd']['auroc_pre_rule']:.3f} | test AUROC {t['mean']['auroc']:.3f} pre-rule {t['mean']['auroc_pre_rule']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    r = sub.add_parser("run")
    r.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 60], help="in 60 s windows")
    r.add_argument("--skip-fail1h", action="store_true")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "cache":
        build_cache()
    elif a.cmd == "run":
        D, tr_all, te_all, ctr, cte = load_frames()
        for h in a.horizons:
            paper_task(D, tr_all, te_all, h)
        if not a.skip_fail1h:
            fail1h(D, tr_all, te_all, ctr, cte)
        summary()
    else:
        summary()


if __name__ == "__main__":
    main()
