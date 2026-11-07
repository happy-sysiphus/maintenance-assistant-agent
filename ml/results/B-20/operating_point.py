"""
B-20 (2026-11-01): the operating point of the production early warning (C-8). User decisions: the early warning is
a "a failure may come" warning, recall first - many false alarms are acceptable - capped at 24 per gun-day. The rule
below was fixed before `run` was executed.

Rule (pre-registered):
    runs      B-18 nested selection (config + threshold chosen on 3 folds, scored on the 4th) on the 8 OOF score sets
              of B-19 (fold seeds 42, 1, 2, 3 x file / physical-gun folds), budgets 0.5 1 2 3 5 10 20 24 false
              alarms per gun-day
    budget    pool the 8 runs per budget (each gun's nested hit / chance averaged over the runs, gun bootstrap -
              results/B-19/physical_cv.py pooled); choose the LARGEST budget <= 24 whose pooled excess CI lower
              bound is > 0
    config    the most frequent of the 32 nested picks (8 runs x 4 folds) at that budget; tie -> the one picked more
              often at the next lower budget, then B-18 SELECTABLE order
    quantile  the median of that config's picked quantiles at that budget
    none      no budget qualifies -> the early warning stays off
Then the fixed rule (config, quantile) is re-scored on all 8 OOF sets - threshold = that quantile of each set's normal
CV scores, exactly as B-18 builds its grid - and pooled the same way (recall, chance, excess [CI], false alarms per
gun-day), plus the 8 test guns (production-bundle scores, each run's threshold).

    python results/B-20/operating_point.py run      -> early_warning_b20_<scheme>_s<seed>.json/.log (B-18 output)
    python results/B-20/operating_point.py decide   -> operating_point.json
Inputs: results/B-19/cache/scores_cv_*.parquet, results/B-11/cache/scores_test.parquet, results/B-17/cache/meta.csv.
Never writes outside results/B-20/.
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "results", "B-18"))
sys.path.insert(0, os.path.join(ROOT, "results", "B-19"))
import early_warning as EW  # noqa: E402

B19_CACHE = os.path.join(ROOT, "results", "B-19", "cache")
SCHEMES, SEEDS = ("file", "physical"), (42, 1, 2, 3)
BUDGETS = (0.5, 1, 2, 3, 5, 10, 20, 24)
CAP = 24


def log(msg):
    print(msg, flush=True)


def runs():
    return [(sc, s) for s in SEEDS for sc in SCHEMES]


def run():
    for sc, s in runs():
        t0 = time.time()
        tag = f"b20_{sc}_s{s}"
        with open(os.path.join(HERE, f"early_warning_{tag}.log"), "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, os.path.join(ROOT, "results", "B-18", "early_warning.py"),
                            "--cv-scores", os.path.join(B19_CACHE, f"scores_cv_{sc}_s{s}.parquet"), "--tag", tag,
                            "--budgets", ",".join(str(b) for b in BUDGETS), "--out-dir", HERE],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        log(f"{tag}: {time.time() - t0:.0f}s")


def parse(pick):
    name, q = pick.split("@q")
    return name, float(q)


def fixed_rule(name, q):
    """Re-score one (config, quantile) on every OOF set; per-gun hit / chance for pooling + the test check."""
    kind, k = EW.CONFIGS[name]
    te = EW.Frame(EW.load("test"))
    per_gun, test = [], []
    for sc, s in runs():
        cv = EW.Frame(EW.load("cv", os.path.join(B19_CACHE, f"scores_cv_{sc}_s{s}.parquet")))
        w = cv.w
        thr = float(np.quantile(w.loc[(w["lead"] > EW.DAY) & (w["error_active"] == 0) & ~w["warmup"], "score_ens"], q))
        fa, hits = cv.evaluate(cv.condition(kind, thr), k)
        plc = np.nanmean(np.vstack([hits[f"p{j}"] for j in EW.PLACEBO_K]), axis=0)
        per_gun.append({"file": [str(g) for g in cv.guns], "fa": fa.tolist(), "days": cv.days.tolist(),
                        "real": [None if np.isnan(x) else float(x) for x in hits["real"]],
                        "chance": [None if np.isnan(x) else float(x) for x in plc]})
        fa_t, hits_t = te.evaluate(te.condition(kind, thr), k)
        t = EW.summarise(fa_t, hits_t, te.days)
        test.append({"run": f"{sc}_s{s}", "threshold": round(thr, 4), "fa_per_day": round(t["fa_per_day"], 2),
                     "hits": int(np.nansum(hits_t["real"])), "events": t["events"], "chance": round(t["chance"], 3),
                     "hit_guns": [str(g) for g, h in zip(te.guns, hits_t["real"]) if h == 1]})
        log(f"  {sc}_s{s}: thr {thr:.4f}  test {test[-1]}")
    return per_gun, test


def decide():
    import physical_cv as P  # B-19 pooling (imports train / horizon - only needed here)
    res = {}
    for sc, s in runs():
        p = os.path.join(HERE, f"early_warning_b20_{sc}_s{s}.json")
        res[(sc, s)] = json.load(open(p, encoding="utf-8"))["nested"]
    pooled, picks = {}, {}
    for b in BUDGETS:
        key = str(b)
        pooled[key] = P.pooled([r[key]["per_gun"] for r in res.values()])
        picks[key] = [parse(p) for r in res.values() for p in r[key]["picks"]]
        c = Counter(n for n, _ in picks[key])
        log(f"budget {key:>4}: fa {pooled[key]['fa_per_day']}  recall {pooled[key]['recall']}  chance "
            f"{pooled[key]['chance']}  excess {pooled[key]['excess']:+.3f} [{pooled[key]['ci'][0]:+.3f}, "
            f"{pooled[key]['ci'][1]:+.3f}]  picks {dict(c.most_common())}")
    ok = [b for b in BUDGETS if b <= CAP and pooled[str(b)]["ci"][0] > 0]
    out = {"rule": __doc__.split("Rule (pre-registered):")[1].split("Then the fixed rule")[0].strip(),
           "budgets": pooled, "picks": {b: [f"{n}@{q}" for n, q in v] for b, v in picks.items()}}
    if not ok:
        out["decision"] = None
        log("no budget has a pooled CI above 0 -> early warning stays off")
    else:
        b = max(ok)
        key = str(b)
        c = Counter(n for n, _ in picks[key])
        top = max(c.values())
        tied = [n for n in EW.SELECTABLE if c.get(n) == top]
        if len(tied) > 1:
            lower = [x for x in BUDGETS if x < b]
            below = Counter(n for n, _ in picks[str(lower[-1])]) if lower else Counter()
            tied.sort(key=lambda n: (-below.get(n, 0), EW.SELECTABLE.index(n)))
        name = tied[0]
        q = float(np.median([qq for n, qq in picks[key] if n == name]))
        log(f"\ndecision: budget {key} -> {name} at q {q:.4f} ({c[name]}/{len(picks[key])} picks)")
        per_gun, test = fixed_rule(name, q)
        fixed = P.pooled(per_gun)
        log(f"fixed rule pooled over 8 runs: {fixed}")
        kind, k = EW.CONFIGS[name]
        out["decision"] = {"budget": b, "config": name, "kind": kind, "sustain": k, "quantile": q,
                           "picks": f"{c[name]}/{len(picks[key])}", "nested_pooled": pooled[key],
                           "fixed_pooled": fixed, "test": test,
                           "test_hits_mean": round(float(np.mean([t["hits"] for t in test])), 2),
                           "test_fa_per_day_mean": round(float(np.mean([t["fa_per_day"] for t in test])), 2)}
    with open(os.path.join(HERE, "operating_point.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log("saved operating_point.json")


if __name__ == "__main__":
    {"run": run, "decide": decide}[sys.argv[1]]()
