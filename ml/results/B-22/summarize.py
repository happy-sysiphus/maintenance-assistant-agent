"""B-22 summary: time-of-day (hour_sin/cos) and error-code (error_share_10min) features dropped.
Reads results/B-22/<run>/baseline_<model>_metrics.json -> results/B-22/summary.json (+ prints a table)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["iforest_base46", "iforest_no_time44", "iforest_no_err44", "iforest_no_time_err40",
        "ensemble_base46", "ensemble_no_time_err40"]
out = {}
for r in RUNS:
    model = r.split("_")[0]
    p = os.path.join(HERE, r, f"baseline_{model}_metrics.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8")); m = d["metrics"]; t = m["test"]; cv = m["cv"]
    ew = t.get("early_warning") or {}
    out[r] = {"model": model, "n_model_cols": len(d["model_cols"]), "dropped": d["dropped_features"],
              "test_auroc": t["auroc"], "test_auroc_pre_rule": t["auroc_pre_rule"], "test_auroc_gun_mean": t["auroc_gun_mean"],
              "test_auprc": t["auprc"], "test_alarm_rate_normal": t["alarm_rate_normal"],
              "test_recall_pre_failure": t["recall_pre_failure"],
              "test_per_class_auroc": {k: v.get("auroc") for k, v in t["per_class"].items()},
              "val_auroc": m["val"]["auroc"], "val_auroc_pre_rule": m["val"]["auroc_pre_rule"],
              "cv_auroc_mean": cv["mean"]["auroc"], "cv_auroc_sd": cv["sd"]["auroc"],
              "cv_auroc_pre_rule_mean": cv["mean"]["auroc_pre_rule"], "cv_auroc_pre_rule_sd": cv["sd"]["auroc_pre_rule"],
              "cv_auroc_gun_mean": cv["mean"]["auroc_gun_mean"],
              "test_early_warning": ew, "threshold": d["threshold"]}
json.dump(out, open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for r, v in out.items():
    print(f"{r:24s} cols {v['n_model_cols']}  test {v['test_auroc']:.3f} / pre {v['test_auroc_pre_rule']:.3f} / gun {v['test_auroc_gun_mean']:.3f}"
          f"  cv {v['cv_auroc_mean']:.3f}±{v['cv_auroc_sd']:.3f} / pre {v['cv_auroc_pre_rule_mean']:.3f}±{v['cv_auroc_pre_rule_sd']:.3f}"
          f"  val {v['val_auroc']:.3f}  alarm {v['test_alarm_rate_normal']:.4f}  recall {v['test_recall_pre_failure']:.3f}")
