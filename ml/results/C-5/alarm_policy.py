"""
C-5 (2026-10-06): fewer false CRITICAL alarms. Today a critical (-> RAG handoff) comes from (a) the terminal-code rule
or (b) the model's alarm sustained for 3 windows (180 s). Over 72 guns (64 out-of-fold + 8 test, results/B-11 cache)
(b) gives 0.52 false critical runs per gun-day vs 0.09 for (a), and the model never catches a failure the rule
misses (B-14: no pre-signal before the last ~10 min). Candidate policies for the MODEL part, chosen on the 64 CV guns
by a rule fixed beforehand, then checked on the 8 test guns:

    sustain   windows above threshold before the model alone is critical (3 = today, 5, 10, 15)
    restart   hold the model alarm for H min after a gun comes back from an idle block (>= 30 windows with weld duty
              < 0.02 or mostly non-welding) - 23 % of the false runs start at 21 UTC (06 KST), the morning restart

Selection (fixed): among policies that keep >= 2 of the 3 guns where the model leads the rule by > 10 min (CV:
E01_1, E01_9), the lowest false critical rate; ties -> the simpler one. The rule part is unchanged.

    python results/C-5/alarm_policy.py      -> results/C-5/alarm_policy.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import train as T  # noqa: E402

COLS = ["file", "time", "ttf_s", "terminal_any", "terminal_idx", "alarm", "weld_duty_10min_mean", "non_welding"]
IDLE_DUTY, IDLE_WINDOWS = 0.02, 30


def load():
    parts = [pd.read_parquet(os.path.join(ROOT, "results", "B-11", "cache", f"scores_{s}.parquet"), columns=COLS)
             .assign(split=s) for s in ("cv", "test")]
    return pd.concat(parts).sort_values(["file", "ttf_s"], ascending=[True, False]).reset_index(drop=True)


def since_idle_min(g):
    """Minutes since the end of the last idle block (chronological windows of one gun), inf before the first one."""
    idle = (g["weld_duty_10min_mean"].to_numpy() < IDLE_DUTY) | (g["non_welding"].to_numpy() > 0.5)
    t = pd.to_datetime(g["time"]).to_numpy()
    out, run, last = np.full(len(g), np.inf), 0, None
    for i in range(len(g)):
        if idle[i]:
            run += 1
            continue
        if run >= IDLE_WINDOWS:
            last = t[i]
        run = 0
        if last is not None:
            out[i] = (t[i] - last) / np.timedelta64(1, "m")
    return out


def evaluate(w, sustain, hold):
    per = {}
    for f, g in w.groupby("file", sort=False):
        ttf = g["ttf_s"].to_numpy()
        a = g["alarm"].to_numpy().astype(bool).copy()
        if hold:
            a &= ~(since_idle_min(g) < hold)
        runs = T.alarm_runs(a, sustain)
        trig = T.rule_triggers(g["terminal_any"].to_numpy(), ttf)
        crit_rule = trig & ~T.rule_repeats(trig, g["terminal_idx"].to_numpy(), ttf)
        rl = ttf[crit_rule & (ttf <= 3600)]
        # the model run "start" as served: the window where the run reaches `sustain` windows
        model_hits = [ttf[s + sustain - 1] / 60 for s, e in runs if ttf[s + sustain - 1] <= 3600]
        per[f] = {"days": ttf.max() / 86400, "model_false": sum(ttf[s + sustain - 1] > 3600 for s, e in runs),
                  "rule_false": int((crit_rule & (ttf > 3600)).sum()),
                  "rule_lead": float(rl.max() / 60) if len(rl) else np.nan,
                  "model_lead": float(max(model_hits)) if model_hits else np.nan}
    p = pd.DataFrame(per).T
    days = p["days"].sum()
    early = p.index[(p["model_lead"] - p["rule_lead"]) > 10].tolist()
    return {"model_false_per_gun_day": float(p["model_false"].sum() / days),
            "critical_false_per_gun_day": float((p["model_false"].sum() + p["rule_false"].sum()) / days),
            "guns_model_hit": int(p["model_lead"].notna().sum()), "guns_rule_hit": int(p["rule_lead"].notna().sum()),
            "guns_model_10min_before_rule": early,
            "guns_any_hit": int((p["model_lead"].notna() | p["rule_lead"].notna()).sum()), "guns": len(p)}


def main():
    w = load()
    sid = {}
    out = {"policies": []}
    for sustain in (3, 5, 10, 15):
        for hold in (0, 30, 60):
            r = {"sustain": sustain, "restart_hold_min": hold}
            for split in ("cv", "test"):
                r[split] = evaluate(w[w["split"] == split], sustain, hold)
            out["policies"].append(r)
            c, t = r["cv"], r["test"]
            print(f"sustain {sustain:2d} hold {hold:2d}: CV critical false {c['critical_false_per_gun_day']:.3f}/gun/day "
                  f"(model {c['model_false_per_gun_day']:.3f}), model-early guns {c['guns_model_10min_before_rule']} | "
                  f"test {t['critical_false_per_gun_day']:.3f} (model {t['model_false_per_gun_day']:.3f}), "
                  f"early {t['guns_model_10min_before_rule']}, hit guns {t['guns_any_hit']}/{t['guns']}", flush=True)
    base = out["policies"][0]["cv"]["guns_model_10min_before_rule"]
    ok = [p for p in out["policies"] if len(set(p["cv"]["guns_model_10min_before_rule"]) & set(base)) >= 2]
    best = min(ok, key=lambda p: (round(p["cv"]["critical_false_per_gun_day"], 3), p["sustain"] + p["restart_hold_min"]))
    out["selected"] = {"sustain": best["sustain"], "restart_hold_min": best["restart_hold_min"]}
    print(f"selected (CV, keeps >= 2 of {base}): sustain {best['sustain']}, restart hold {best['restart_hold_min']} min | "
          f"CV {best['cv']['critical_false_per_gun_day']:.3f} vs today {out['policies'][0]['cv']['critical_false_per_gun_day']:.3f}"
          f" | test {best['test']['critical_false_per_gun_day']:.3f} vs {out['policies'][0]['test']['critical_false_per_gun_day']:.3f},"
          f" test early guns {best['test']['guns_model_10min_before_rule']}")
    del sid
    with open(os.path.join(HERE, "alarm_policy.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
