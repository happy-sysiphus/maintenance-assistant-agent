"""
B-18 (2026-11-01): a recall-first early warning tens of minutes ahead. Mentor feedback: a miss costs more than a false
alarm, and AUROC ~0.6 can be meaningful. B-11 found AUROC 0.64-0.65 for windows ~0-50 min before the terminal code,
above the placebo band, but mostly from other error codes (E003 ...) being active. This script asks, at the EVENT level:
if the alarm threshold is lowered, what share of failures gets an alarm 10-60 min before the terminal code, how many
false alarms does that cost, and how much of that recall is chance?

Pre-registered (fixed before any result was computed):
    event       terminal-code onset T (first own code in the last 15 min, results/B-17/cache/meta.csv)
    hit         an alarm is active in a window ending 10-60 min before T (lead >= 10 min); also reported for 0-60 min
    chance      the same hit rule at placebo anchors T - k x 24 h, k = 1..6 (whole days: same clock time, so daily
                schedules such as the KST 03-05 error codes are matched); a zone needs >= 25 of its 51 windows
    false alarm onset of an alarm run in a window ending > 60 min before T, per monitored gun-day (warm-up excluded)
    alarm       score_ens >= threshold (one global threshold, swept over quantiles of normal CV windows), sustained k
                consecutive minutes; warm-up windows never alarm; options below
    configs     model        k = 1 / 3 / 10, non-welding gate on (production: share > 0.5 holds the alarm)
                model_ng     k = 1 / 3, gate off
                model_quiet  k = 1 / 3, gate on, windows in an error state never alarm (sensor part only)
                model_watch  k = 1 / 3, gate on, OR a non-terminal error code is active ("watch" rule)
                watch        a non-terminal error code is active (no model)
    selection   per false-alarm budget (0.5 / 1 / 2 / 3 / 5 / 10 per gun-day): the config + threshold with the largest
                CV excess (recall - chance); gun bootstrap 95% CI; test (8 guns) only checks, with the CV threshold
    "beyond chance" at a budget = CV excess CI excludes 0 and the test excess is not negative

Follow-up checks (added after the first run, so not part of the pre-registration):
    nested      the per-budget selection repeated inside the CV: choose config + threshold on 3 folds' guns, score the
                4th, pool - removes the optimism of picking the best of 9 configs x 31 thresholds on the same guns
    near        a second chance level at T - 2 / 3 / 4 / 6 h (same elapsed time, different clock time): a real excess
                over both day and near anchors is failure-specific, not drift or schedule
    class       recall / chance per class for the chosen configs
    stop_k*     exploratory rule: non-welding share > 0.5 for k consecutive minutes (what the gate-off model adds?)

Inputs: results/B-11/cache/scores_{cv,test}.parquet (64 guns out-of-fold ensemble scores, 8 test guns with the
production bundle), results/B-17/cache/meta.csv. Never writes outside results/B-18/.

    python results/B-18/early_warning.py        -> results/B-18/early_warning.json (+ .log via shell)
    python results/B-18/early_warning.py --cv-scores <parquet> --tag <tag>   -> early_warning_<tag>.json
        (same analysis on other OOF scores, e.g. B-19's physical-gun folds)
        --budgets 0.5,1,...  other false-alarm budgets;  --out-dir <dir>  write the JSON there (B-20)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B11 = os.path.join(ROOT, "results", "B-11", "cache")
META = os.path.join(ROOT, "results", "B-17", "cache", "meta.csv")
COLS = ["file", "time", "error_active", "terminal_any", "non_welding", "warmup", "score_ens"]
ZONE = (600, 3600)
ZONE0 = (0, 3600)
PLACEBO_K = range(1, 7)
NEAR_H = (2, 3, 4, 6)
DAY = 86400
GATE = 0.5
BUDGETS = (0.5, 1, 2, 3, 5, 10)
CONFIGS = {f"model_k{k}": ("model", k) for k in (1, 3, 10)}
CONFIGS |= {f"model_ng_k{k}": ("model_ng", k) for k in (1, 3)}
CONFIGS |= {f"model_quiet_k{k}": ("model_quiet", k) for k in (1, 3)}
CONFIGS |= {f"model_watch_k{k}": ("model_watch", k) for k in (1, 3)}
SELECTABLE = list(CONFIGS)                                 # the pre-registered set
CONFIGS |= {f"stop_k{k}": ("stop", k) for k in (3, 10, 30)}  # exploratory, never selected
SEED = 42


def log(msg):
    print(msg, flush=True)


def load(split, path=None):
    w = pd.read_parquet(path or os.path.join(B11, f"scores_{split}.parquet"))
    w = w[COLS + (["fold"] if "fold" in w else [])]
    meta = pd.read_csv(META, parse_dates=["fault_time"]).set_index("file")
    w = w.sort_values(["file", "time"]).reset_index(drop=True)
    w["lead"] = (w["file"].map(meta["fault_time"]) - (w["time"] + pd.Timedelta(seconds=60))).dt.total_seconds()
    w["cls"] = w["file"].map(meta["class"])
    if "fold" not in w:
        w["fold"] = -1
    w = w[w["lead"] > 0].reset_index(drop=True)          # nothing at or after the terminal code
    w["warmup"] = w["warmup"].astype(bool)
    w["nonterm"] = (w["error_active"] > 0) & (w["terminal_any"] == 0)
    return w


class Frame:
    """Fixed per-split arrays: run breaks, zone masks, gun index."""

    def __init__(self, w):
        self.w = w
        self.gun, self.guns = pd.factorize(w["file"])
        first = w.groupby(self.gun).first()
        self.cls, self.fold = first["cls"].to_numpy(), first["fold"].to_numpy()
        t = w["time"].to_numpy()
        same = np.r_[False, (self.gun[1:] == self.gun[:-1]) & (np.diff(t) == np.timedelta64(60, "s"))]
        self.brk = ~same
        lead = w["lead"].to_numpy()
        ok = ~w["warmup"].to_numpy()
        self.ok = ok
        self.normal = (lead > ZONE[1]) & ok
        self.days = np.bincount(self.gun, self.normal, len(self.guns)) / 1440
        self.zones = {"real": self.zone(lead, ZONE, ok), "real0": self.zone(lead, ZONE0, ok)}
        for k in PLACEBO_K:
            self.zones[f"p{k}"] = self.zone(lead - k * DAY, ZONE, ok)
            self.zones[f"p{k}_0"] = self.zone(lead - k * DAY, ZONE0, ok)
        for h in NEAR_H:
            self.zones[f"n{h}"] = self.zone(lead - h * 3600, ZONE, ok)

    def zone(self, lead, rng, ok):
        z = (lead >= rng[0]) & (lead <= rng[1]) & ok
        n = np.bincount(self.gun, z, len(self.guns))
        valid = n >= 25 * (rng[1] - rng[0]) / 3000      # >= 25 of 51 windows (10-60 min); scaled for 0-60 min
        return z, valid

    def condition(self, kind, thr):
        w = self.w
        s = w["score_ens"].to_numpy() >= thr
        gate = w["non_welding"].to_numpy() <= GATE
        if kind == "model":
            c = s & gate
        elif kind == "model_ng":
            c = s
        elif kind == "model_quiet":
            c = s & gate & (w["error_active"].to_numpy() == 0)
        elif kind == "model_watch":
            c = (s & gate) | w["nonterm"].to_numpy()
        elif kind == "stop":
            c = ~gate
        else:                                               # watch
            c = w["nonterm"].to_numpy().copy()
        return c & self.ok

    def evaluate(self, c, k):
        idx = np.arange(len(c))
        start = c & (self.brk | ~np.r_[False, c[:-1]])
        last = np.maximum.accumulate(np.where(start, idx, -1))
        pos = idx - last
        alarm = c & (pos >= k - 1)
        onset = c & (pos == k - 1)
        fa = np.bincount(self.gun, onset & self.normal, len(self.guns))
        hits = {}
        for name, (z, valid) in self.zones.items():
            h = np.bincount(self.gun, alarm & z, len(self.guns)) > 0
            hits[name] = np.where(valid, h.astype(float), np.nan)
        return fa, hits


def summarise(fa, hits, days, sel=None):
    sel = np.arange(len(days)) if sel is None else sel
    real, real0 = hits["real"][sel], hits["real0"][sel]
    plc = np.nanmean(np.vstack([hits[f"p{k}"][sel] for k in PLACEBO_K]), axis=0)
    plc0 = np.nanmean(np.vstack([hits[f"p{k}_0"][sel] for k in PLACEBO_K]), axis=0)
    near = np.nanmean(np.vstack([hits[f"n{h}"][sel] for h in NEAR_H]), axis=0)
    return {"fa_per_day": float(fa[sel].sum() / days[sel].sum()),
            "recall": float(np.nanmean(real)), "chance": float(np.nanmean(plc)),
            "excess": float(np.nanmean(real) - np.nanmean(plc)), "chance_near": float(np.nanmean(near)),
            "recall_0_60": float(np.nanmean(real0)), "chance_0_60": float(np.nanmean(plc0)),
            "events": int(np.isfinite(real).sum())}


def bootstrap(fa, hits, days, n=2000):
    rng = np.random.default_rng(SEED)
    g = len(days)
    ex = [summarise(fa, hits, days, rng.integers(0, g, g))["excess"] for _ in range(n)]
    return [float(np.nanpercentile(ex, 2.5)), float(np.nanpercentile(ex, 97.5))]


def main():
    t0 = time.time()
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    tag = f"_{args['--tag']}" if "--tag" in args else ""
    budgets = tuple(int(x) if float(x).is_integer() else float(x) for x in args["--budgets"].split(",")) \
        if "--budgets" in args else BUDGETS
    out_dir = args.get("--out-dir", HERE)
    cv, te = Frame(load("cv", args.get("--cv-scores"))), Frame(load("test"))
    w = cv.w
    normal_scores = w.loc[(w["lead"] > DAY) & (w["error_active"] == 0) & ~w["warmup"], "score_ens"]
    qs = np.r_[np.arange(0.50, 0.95, 0.025), np.arange(0.95, 0.995, 0.005), [0.995, 0.997, 0.999]]
    thrs = np.quantile(normal_scores, qs)
    log(f"cv {len(cv.guns)} guns {cv.days.sum():.0f} gun-days, test {len(te.guns)} guns; {len(thrs)} thresholds")

    curves, res = {}, {}                                  # res[name][i] = per-gun (fa, hits) on the CV guns
    for name, (kind, k) in CONFIGS.items():
        rows, res[name] = [], []
        for q, thr in zip(qs, thrs):
            fa, hits = cv.evaluate(cv.condition(kind, thr), k)
            res[name].append((fa, hits))
            rows.append({"q": float(q), "thr": float(thr)} | summarise(fa, hits, cv.days))
            if kind == "stop":
                break                                     # no threshold
        curves[name] = rows
    fa, hits = cv.evaluate(cv.condition("watch", 0), 1)
    watch = summarise(fa, hits, cv.days) | {"ci": bootstrap(fa, hits, cv.days)}
    fa, hits = te.evaluate(te.condition("watch", 0), 1)
    watch_test = summarise(fa, hits, te.days)
    log(f"\nwatch rule alone (non-terminal code active): CV {watch}\n  test {watch_test}")

    log("\nCV curves (recall = alarm 10-60 min before the terminal code; chance = same at -1..-6 days):")
    for name, rows in curves.items():
        log(f"  {name:15s} " + "  ".join(f"q{r['q']:.3f}: fa {r['fa_per_day']:.2f} rec {r['recall']:.2f} "
                                         f"ch {r['chance']:.2f}" for r in rows[::6]))

    chosen = {}
    log("\nper budget: best CV excess -> test check (same threshold)")
    log("budget  config           q      | CV fa  recall chance excess [95% CI]        | test fa recall chance")
    for b in budgets:
        best = None
        for name in SELECTABLE:
            ok = [r for r in curves[name] if r["fa_per_day"] <= b]
            if ok:
                r = max(ok, key=lambda r: r["excess"])
                if best is None or r["excess"] > best[1]["excess"]:
                    best = (name, r)
        if best is None:
            continue
        name, r = best
        kind, k = CONFIGS[name]
        fa, hits = cv.evaluate(cv.condition(kind, r["thr"]), k)
        ci = bootstrap(fa, hits, cv.days)
        fa_t, hits_t = te.evaluate(te.condition(kind, r["thr"]), k)
        t = summarise(fa_t, hits_t, te.days) | {"per_gun": dict(zip(te.guns, hits_t["real"].tolist()))}
        beyond = ci[0] > 0 and t["excess"] >= 0
        by_cls = {c: summarise(fa, hits, cv.days, np.flatnonzero(cv.cls == c)) for c in sorted(set(cv.cls))}
        chosen[str(b)] = {"config": name, "cv": r | {"ci": ci}, "test": t, "beyond_chance": beyond,
                          "cv_by_class": by_cls}
        log(f"{b:5.1f}   {name:15s} {r['q']:.3f}  | {r['fa_per_day']:.2f}  {r['recall']:.3f}  {r['chance']:.3f}  "
            f"{r['excess']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {t['fa_per_day']:.2f}  {t['recall']:.3f}  "
            f"{t['chance']:.3f}  {'BEYOND' if beyond else '-'}")
        log(f"        near chance (T-2..6 h) {r['chance_near']:.3f};  by class recall/chance: " + "  ".join(
            f"{c} {v['recall']:.2f}/{v['chance']:.2f}" for c, v in by_cls.items()))

    log("\nnested: select config + threshold on 3 folds, score the held-out fold (pooled over the 4 folds)")
    nested = {}
    for b in budgets:
        fa_all = np.zeros(len(cv.guns))
        hits_all = {z: np.full(len(cv.guns), np.nan) for z in cv.zones}
        picks = []
        for f in sorted(set(cv.fold)):
            tr, va = np.flatnonzero(cv.fold != f), np.flatnonzero(cv.fold == f)
            best = None
            for name in SELECTABLE:
                for i, (fa, hits) in enumerate(res[name]):
                    s = summarise(fa, hits, cv.days, tr)
                    if s["fa_per_day"] <= b and (best is None or s["excess"] > best[0]):
                        best = (s["excess"], name, i)
            _, name, i = best
            fa, hits = res[name][i]
            fa_all[va] = fa[va]
            for z in hits_all:
                hits_all[z][va] = hits[z][va]
            picks.append(f"{name}@q{qs[i]:.3f}")
        plc = np.nanmean(np.vstack([hits_all[f"p{k}"] for k in PLACEBO_K]), axis=0)
        per_gun = {"file": [str(g) for g in cv.guns], "fa": fa_all.tolist(), "days": cv.days.tolist(),
                   "real": [None if np.isnan(x) else float(x) for x in hits_all["real"]],
                   "chance": [None if np.isnan(x) else float(x) for x in plc]}   # for pooling runs (B-19 summary)
        s = summarise(fa_all, hits_all, cv.days) | {"ci": bootstrap(fa_all, hits_all, cv.days), "picks": picks,
                                                    "per_gun": per_gun}
        nested[str(b)] = s
        log(f"{b:5.1f}  fa {s['fa_per_day']:.2f}  recall {s['recall']:.3f}  chance {s['chance']:.3f}  near "
            f"{s['chance_near']:.3f}  excess {s['excess']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  {picks}")

    log("\nexploratory stop rule (non-welding share > 0.5 for k minutes):")
    for name in [n for n in CONFIGS if n.startswith("stop")]:
        fa, hits = res[name][0]
        s = summarise(fa, hits, cv.days) | {"ci": bootstrap(fa, hits, cv.days)}
        t = summarise(*te.evaluate(te.condition("stop", 0), CONFIGS[name][1]), te.days)
        curves[name][0] |= {"ci": s["ci"], "test": t}
        log(f"  {name:8s} fa {s['fa_per_day']:.2f}  recall {s['recall']:.3f}  chance {s['chance']:.3f}  near "
            f"{s['chance_near']:.3f}  excess {s['excess']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  | test fa "
            f"{t['fa_per_day']:.2f} recall {t['recall']:.3f} chance {t['chance']:.3f}")

    out = {"criteria": __doc__.split("Pre-registered")[1].split("Inputs:")[0].strip(),
           "cv_guns": len(cv.guns), "test_guns": len(te.guns), "watch": {"cv": watch, "test": watch_test},
           "curves": curves, "per_budget": chosen, "nested": nested, "seconds": round(time.time() - t0)}
    dst = os.path.join(out_dir, f"early_warning{tag}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log(f"saved {dst} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
