"""A-2: the three remaining data questions (techspec A-2, history.md §21), on the preprocessed files and the serving
ensemble (models/baseline_ensemble.joblib).

    python results/A-2/data_questions.py       # -> results/A-2/data_questions.json (~5 min)

Q1  How many files spend most of their last hour before the failure in non-welding (cap dressing) - windows the gate
    holds, so the model cannot alarm there?
Q2  Do the E02 files of the two collection periods (2019-20 vs 2021) differ - is the gun offset a period effect that
    would need a per-period scaler?
Q3  How do the test guns that weld thick plate (c16 = 8.7) most of the time score - does thickness need its own
    baseline?
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import train as T  # noqa: E402

PRE = os.path.join(ROOT, "preprocessed")
SCALER = json.load(open(os.path.join(PRE, "scaler.json"), encoding="utf-8"))["params"]
SENSORS = ["c1", "c2", "c3", "c4", "c5", "c6", "c13", "c14", "c15", "c16", "c17", "c18"]
THICK = 5.0  # c16 setpoint above this = thick plate (8.7); the thin plates are 1.2-2.0


def raw(df, c):
    p = SCALER[c]
    return df[c] * p["std"] + p["mean"]


def files():
    return sorted(glob.glob(os.path.join(PRE, "E0*.parquet"))) + sorted(glob.glob(os.path.join(PRE, "test", "test_*.parquet")))


def main():
    rows = []
    for f in files():
        df = pd.read_parquet(f, columns=["class", "non_welding", "error_active", "label", "ttf_s", *SENSORS])
        name = os.path.splitext(os.path.basename(f))[0]
        last_h = df["ttf_s"] <= 3600
        normal = (df["label"] == 0) & (df["error_active"] == 0) & (df["non_welding"] == 0)
        c16 = raw(df, "c16")
        welding = df["non_welding"] == 0
        rows.append({"file": name, "split": "test" if name.startswith("test") else "train", "class": str(df["class"].iloc[0]),
                     "start": df.index[0], "nw_last_h": float(df.loc[last_h, "non_welding"].mean()),
                     "nw_all": float(df["non_welding"].mean()),
                     "thick_share": float((c16[welding] > THICK).mean()),
                     **{f"mean_{c}": float(df.loc[normal, c].mean()) for c in SENSORS}})
        print(name, flush=True)
    F = pd.DataFrame(rows)
    out = {}

    # ---- Q1: non-welding in the last hour
    q1 = F.sort_values("nw_last_h", ascending=False)
    out["q1"] = {"files": len(F), "last_h_nw_over_50pct": int((F["nw_last_h"] > 0.5).sum()),
                 "last_h_nw_over_25pct": int((F["nw_last_h"] > 0.25).sum()),
                 "median_last_h_nw": round(float(F["nw_last_h"].median()), 3),
                 "median_all_nw": round(float(F["nw_all"].median()), 3),
                 "top": q1.head(6)[["file", "class", "nw_last_h", "nw_all"]].round(3).to_dict("records")}

    # ---- Q2: collection period
    F["year"] = pd.to_datetime(F["start"]).dt.year
    F["period"] = np.where(F["year"] <= 2020, "2019-20", "2021+")
    out["q2"] = {"files_by_class_period": F.groupby(["class", "period"]).size().unstack(fill_value=0).to_dict()}
    e02 = F[(F["class"] == "E02") & (F["split"] == "train")]
    eff = {}
    for c in SENSORS:
        a, bb = e02.loc[e02["period"] == "2019-20", f"mean_{c}"], e02.loc[e02["period"] == "2021+", f"mean_{c}"]
        if len(a) > 1 and len(bb) > 1:
            sd = np.sqrt((a.var(ddof=1) + bb.var(ddof=1)) / 2) or 1e-9
            eff[c] = round(float((bb.mean() - a.mean()) / sd), 2)
    out["q2"]["e02_period_effect_d"] = eff  # difference of the per-file normal means, in between-file SD units
    # does the period show in the serving model's CV alarm rates? (per-gun normal alarm rate from the ensemble 4-fold)
    m = json.load(open(os.path.join(ROOT, "models", "baseline_ensemble_metrics.json"), encoding="utf-8"))
    rates = {}
    for fo in m["metrics"]["cv"]["folds"]:
        rates.update(fo["alarm_rate_normal_per_file"])
    F["cv_alarm"] = F["file"].map(rates)
    out["q2"]["cv_alarm_by_period"] = F[F["split"] == "train"].groupby("period")["cv_alarm"].agg(
        ["count", "mean", "median", "max"]).round(4).to_dict("index")
    out["q2"]["e02_cv_alarm_by_period"] = e02.assign(cv_alarm=e02["file"].map(rates)).groupby("period")["cv_alarm"].agg(
        ["count", "mean", "max"]).round(4).to_dict("index")

    # ---- Q3: thick plate
    out["q3"] = {"train_thick_share_max": round(float(F.loc[F["split"] == "train", "thick_share"].max()), 4),
                 "train_files_thick_over_10pct": int((F.loc[F["split"] == "train", "thick_share"] > 0.1).sum()),
                 "test_thick_share": F.loc[F["split"] == "test"].set_index("file")["thick_share"].round(3).to_dict()}
    bundle = T.load_bundle(os.path.join(ROOT, "models", "baseline_ensemble.joblib"))
    per = {}
    for f in sorted(glob.glob(os.path.join(PRE, "test", "test_*.parquet"))):
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_parquet(f)
        sf = T.score_frame(bundle, df)
        thick = (raw(df, "c16") > THICK) & (df["non_welding"] == 0)
        w_thick = thick.groupby(df.index.floor("60s")).mean().reindex(sf.index)
        normal = (sf["label"] == 0) & (sf["error_active"] == 0)
        pre = sf["label"] == 1
        d = {"windows": int(len(sf)), "normal_alarm": round(float(sf.loc[normal, "alarm"].mean()), 4)}
        for tag, sel in (("thick", w_thick > 0.5), ("thin", w_thick <= 0.5)):
            n = normal & sel
            d[f"{tag}_normal_windows"] = int(n.sum())
            d[f"{tag}_normal_alarm"] = round(float(sf.loc[n, "alarm"].mean()), 4) if n.any() else None
            d[f"{tag}_normal_score_median"] = round(float(sf.loc[n, "score"].median()), 4) if n.any() else None
            d[f"{tag}_pre_windows"] = int((pre & sel).sum())
        per[name] = d
        print(name, d, flush=True)
    out["q3"]["test_by_thickness"] = per
    json.dump(out, open(os.path.join(HERE, "data_questions.json"), "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "q3"}, indent=1, default=str))
    print(json.dumps(out["q3"], indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
