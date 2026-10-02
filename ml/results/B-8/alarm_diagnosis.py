"""B-8 diagnosis (history.md §17; needs `python .py/experiments.py cache --window 60`): where / when do the high-alarm guns alarm on normal windows? (4-fold, production IF setting)"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import experiments as E  # noqa: E402
import train as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache")  # per-window scores (git-ignored)
os.makedirs(OUT, exist_ok=True)
D = E.without(E.load(60), ["c19"])
cols = D["model_cols"]
allw, cal = D["all"], D["cal_all"]
stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
folds = T.cv_folds(sorted(glob.glob(os.path.join(E.DATA, "E0*.parquet"))), 4, E.SEED)
rows = []
for i, va in enumerate(folds, 1):
    va_s = {stem(f) for f in va}
    tr = allw[~allw["file"].isin(va_s)]
    m = E.normal_mask(tr)
    X = tr.loc[m, cols].to_numpy(dtype=np.float32)
    model = T.build_model("iforest", E.SEED).fit(X)
    thr = float(np.quantile(T.anomaly_score(model, X), E.Q))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    va_w = allw[allw["file"].isin(va_s)].copy()
    gthr = T.gun_thresholds(model, cols, cal[cal["file"].isin(va_s)], thr, E.Q, E.GATE)
    s = T.anomaly_score(model, va_w[cols].to_numpy(dtype=np.float32))
    th = T.window_thresholds(va_w, thr, gthr)
    va_w["score"], va_w["thr"] = s, th
    va_w["alarm"] = T.alarm_mask(s, th, va_w["non_welding"].values, E.GATE)
    va_w["fold"] = i
    z = (va_w[cols].to_numpy(dtype=np.float32) - mu) / sd
    va_w = pd.concat([va_w, pd.DataFrame(z, index=va_w.index, columns=[f"z_{c}" for c in cols])], axis=1)
    rows.append(va_w)
    print(f"fold {i}: thr {thr:.4f}", flush=True)
W = pd.concat(rows)
W["normal"] = (W["label"] == 0) & (W["error_active"] == 0)
W["day"] = W.groupby("file")["ttf_s"].transform(lambda t: (t.max() - t) // 86400).astype(int)
N = W[W["normal"]]
rate = N.groupby("file")["alarm"].mean().sort_values(ascending=False)
print("per-gun normal alarm rate: mean %.4f max %.4f  >5%%: %s" % (rate.mean(), rate.max(), list(rate[rate > 0.05].round(3).items())))
W[["file", "fold", "day", "score", "thr", "alarm", "normal", "warmup", "non_welding", "label", "ttf_s"]
  + [f"z_{c}" for c in cols]].to_parquet(os.path.join(OUT, "windows.parquet"))

report = {"rate": rate.round(4).to_dict()}
for f in list(rate.index[:6]) + list(rate.index[-2:]):
    n = N[N["file"] == f]
    by_day = n.groupby("day")["alarm"].agg(["mean", "size"]).round(3)
    by_warm = n.groupby("warmup")["alarm"].mean().round(3).to_dict()
    a = n[n["alarm"]]
    zc = [f"z_{c}" for c in cols]
    top_alarm = a[zc].mean().abs().sort_values(ascending=False).head(6) if len(a) else pd.Series(dtype=float)
    top_all = n[zc].mean().abs().sort_values(ascending=False).head(6)
    hour = n.groupby(n.index.hour)["alarm"].mean()
    report[f] = {"rate": float(rate[f]), "by_day": by_day["mean"].to_dict(), "by_warmup": by_warm,
                 "gun_thr": float(n["thr"].iloc[-1]), "median_score": float(n["score"].median()),
                 "top_z_in_alarms": {k[2:]: round(float(a[k].mean()), 2) for k in top_alarm.index},
                 "top_z_all_normal": {k[2:]: round(float(n[k].mean()), 2) for k in top_all.index},
                 "alarm_hours_top": hour.sort_values(ascending=False).head(4).round(3).to_dict()}
    print(f"\n== {f}  rate {rate[f]:.3f}  thr {n['thr'].iloc[-1]:.4f}  median score {n['score'].median():.4f}")
    print("  by day:", by_day["mean"].to_dict(), " warmup:", by_warm)
    print("  z in alarms:", report[f]["top_z_in_alarms"])
    print("  z all normal:", report[f]["top_z_all_normal"])
    print("  alarm by hour (top):", report[f]["alarm_hours_top"])
json.dump(report, open(os.path.join(HERE, "alarm_diagnosis.json"), "w"), indent=1, default=str)
