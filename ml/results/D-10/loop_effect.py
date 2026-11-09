"""
D-10 feedback-effect check (2026-11-09): do the two feedback loops change performance? Scope approved by the user
("이대로 돌리기", 2026-11-09): no paid calls, no new data, no product code changed. R2 is in r2_sim.py (api environment).

PRE-REGISTERED before computing any result below (2026-11-09):
    feedback    SIMULATED from the data's own answer, nobody's judgment. A critical alarm is real if its class is the class
                of one of the physical gun's files and it fires 0-3600 s before that file's end, else false (the rule of
                results/D-9/calibrate.py part 4). Every alarm is closed correctly before the next one
    data        terminal-code triggers at row level (main.Detector.rule_check through results/C-6/rule_rowlevel.py; repeats
                never become cases), 64 preprocessed train files + 8 test files, physical guns per split, walked in time
                order: a decision at an alarm uses only the closures before it
    1 profile   spec 7.6 with the D-9 constants (2 cases on 2 days, applied at once): once a class is excluded its later
                alarms make no case. Before = no profile. Benefit = saved share of the false-alarm cases
    2 R1        api/field_memory.py: same gun and code, past false alarms > past real ones. Every alarm makes a case here.
                Benefit = share of the alarms with the sentence that were false (precision) minus the share of all alarms
                that were false (base rate). Also: share of the false alarms covered, real alarms that got the sentence
    significant gun bootstrap (physical guns, 2000 draws, seed 7) pooled over both splits; each split reported too. "improved"
                needs (a) the benefit's 95% CI above 0 and (b) harm that can be measured: at least 29 exposures = real alarms
                preceded by a false alarm of the same gun and class, with a one-sided 95% upper bound of the wrong rate
                <= 10% (0 wrong of 29 gives 9.8%). Fewer exposures -> "cannot_judge", not "no_improvement"
    3 early warning  the D-9 verdicts unchanged (spec 7.4 / 7.5; B-19 lab, results/D-9/runs/offline-1 with simulated labels):
                "after" = the rules the loop proposes. What-if on the 8 test files at the threshold the apply command would
                produce = the mean of the 8 lab runs' thresholds at that quantile (GATE: the served threshold): false alarms
                per gun-day, recall, excess, D and recall change against the served rule (paired gun bootstrap, seed 7)
    4 sample sizes  two-sided 5%, power 80%, normal approximation.
                Early warning: failures needed for a paired D of 0.05, 0.10 and each challenger's lab |D|; sigma = per-failure
                SD of the paired excess difference in one run, mean over the 8 runs.
                Profile and R1: guns that must benefit so that a gun resample holds none with probability < 2.5%, plus the 29
                exposures above. R1 precision: smallest n from which an exact one-sided binomial test at 2.5% keeps power
                >= 80% for base rate + 0.30

    python results/D-10/loop_effect.py     -> results/D-10/loop_effect.json (+ .log via shell)

Inputs: preprocessed/, results/A-4/cache, results/B-19/cache, results/B-17/cache/meta.csv, results/D-9/runs/.
Writes only results/D-10/loop_effect.json.
"""
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "results", "D-9"))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import calibrate as CAL  # noqa: E402  (brings results/B-18/early_warning and results/C-6/rule_rowlevel, which imports train)
import feedback_eval as fe  # noqa: E402
import feedback_loop as fl  # noqa: E402

LAB = os.path.join(ROOT, "results", "B-19", "cache")
META = os.path.join(ROOT, "results", "B-17", "cache", "meta.csv")
RUNS = os.path.join(ROOT, "results", "D-9", "runs")
STATE = os.path.join(RUNS, "state-test8")
HISTORY = os.path.join(RUNS, "history_sim.json")
REPORT1 = os.path.join(RUNS, "offline-1", "report.json")
MIN_EXPOSED = 29
MAX_WRONG_RATE = 0.10
COLS = ("alarms", "real", "false", "saved", "wrong", "exposed", "r1_fires", "r1_right", "r1_wrong")


def log(msg):
    print(msg, flush=True)


def walk(files):
    """One physical gun in time order; the profile and R1 each in their own world."""
    g = defaultdict(int)
    excluded, nfa, days, nreal = set(), defaultdict(int), defaultdict(set), defaultdict(int)  # the profile's world
    fa_seen, real_seen = defaultdict(int), defaultdict(int)                                   # R1's world
    for tm, c, r in CAL.merged_triggers(files):
        if r:
            continue
        cls = CAL.T.CLASSES[c - 1]
        real = CAL.is_real(tm, cls, files)
        g["alarms"] += 1
        g["real" if real else "false"] += 1
        if real and fa_seen[cls]:
            g["exposed"] += 1
        if fa_seen[cls] > real_seen[cls]:
            g["r1_fires"] += 1
            g["r1_wrong" if real else "r1_right"] += 1
        (real_seen if real else fa_seen)[cls] += 1
        if cls in excluded:                                 # calibrate.replay_profile from here
            g["wrong" if real else "saved"] += 1
            continue
        if real:
            nreal[cls] += 1
        else:
            nfa[cls] += 1
            days[cls].add(tm.date())
        if (nfa[cls] >= fe.PROFILE_CASES and len(days[cls]) >= fe.PROFILE_DAYS and nreal[cls] == 0
                and len(excluded) < len(CAL.T.CLASSES) - 1):
            excluded.add(cls)
    return g


def wrong_upper(wrong, exposed):
    """One-sided 95% upper bound of the wrong rate (Clopper-Pearson); 1.0 with no exposure."""
    if exposed == 0 or wrong >= exposed:
        return 1.0
    return float(stats.beta.ppf(0.95, wrong + 1, exposed - wrong))


def verdict(ci, wrong, exposed):
    if exposed < MIN_EXPOSED:
        return "cannot_judge"
    return "improved" if ci is not None and ci[0] > 0 and wrong_upper(wrong, exposed) <= MAX_WRONG_RATE else "no_improvement"


def block(t):
    """Totals and gun-bootstrap CIs of one set of physical guns."""
    n = len(t)
    idx = np.random.default_rng(fe.SEED_PAIRED).integers(0, n, (fe.N_BOOT, n))
    cols = {c: t[c].to_numpy(float) for c in COLS}

    def draw(c):
        return cols[c][idx].sum(axis=1)

    def ci(x):
        return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]

    tot = {c: int(cols[c].sum()) for c in COLS}
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_saved = draw("saved") / draw("false")
        boot_gain = draw("r1_right") / draw("r1_fires") - draw("false") / draw("alarms")
        boot_cover = draw("r1_right") / draw("false")
    precision = tot["r1_right"] / tot["r1_fires"] if tot["r1_fires"] else None
    base = tot["false"] / tot["alarms"]
    profile_ci, gain_ci = ci(boot_saved), ci(boot_gain) if tot["r1_fires"] else None
    return {"guns": n, **tot,
            "profile": {"false_before": tot["false"], "false_after": tot["false"] - tot["saved"],
                        "saved_share": tot["saved"] / tot["false"] if tot["false"] else None, "ci": profile_ci,
                        "guns_benefiting": int((cols["saved"] > 0).sum()), "real_suppressed": tot["wrong"],
                        "exposed": tot["exposed"], "wrong_upper": wrong_upper(tot["wrong"], tot["exposed"]),
                        "verdict": verdict(profile_ci, tot["wrong"], tot["exposed"])},
            "r1": {"fires": tot["r1_fires"], "guns_fired": int((cols["r1_fires"] > 0).sum()), "precision": precision,
                   "base_rate": base, "gain": None if precision is None else precision - base, "ci": gain_ci,
                   "coverage_of_false": tot["r1_right"] / tot["false"] if tot["false"] else None,
                   "coverage_ci": ci(boot_cover), "wrong": tot["r1_wrong"], "exposed": tot["exposed"],
                   "wrong_upper": wrong_upper(tot["r1_wrong"], tot["exposed"]),
                   "verdict": verdict(gain_ci, tot["r1_wrong"], tot["exposed"])}}


def critical():
    phys = CAL.physical_ids()
    paths = {"train": sorted(glob.glob(os.path.join(ROOT, "preprocessed", "E0*.parquet"))),
             "test": sorted(glob.glob(os.path.join(ROOT, "preprocessed", "test", "test_*.parquet")))}
    rows = []
    for split, ps in paths.items():
        g = defaultdict(list)
        for p in ps:
            stem = os.path.splitext(os.path.basename(p))[0]
            g[phys.get(stem, stem)].append((stem, *CAL.file_events(p)))
        for key, files in sorted(g.items()):
            res = walk(files)
            rows.append({"split": split, "gun": key, "files": ",".join(sorted(f[0] for f in files)),
                         **{c: int(res[c]) for c in COLS}})
        log(f"{split}: {len(ps)} files, {len(g)} physical guns")
    t = pd.DataFrame(rows)
    tr, te = t[t["split"] == "train"], t[t["split"] == "test"]
    got = (int(tr["saved"].sum()), int(tr["false"].sum() - tr["saved"].sum()), int(tr["real"].sum()),
           int(te["saved"].sum()), int(te["false"].sum() - te["saved"].sum()), int(te["real"].sum()))
    assert got == (1, 32, 64, 3, 8, 8), f"does not reproduce calibrate.json rule_v2.profile: {got}"
    log("sanity: profile totals equal calibrate.json (train saved 1, false 32, real 64; test saved 3, false 8, real 8)")
    fired = t.loc[t["r1_fires"] > 0, ["split", "files", "alarms", "false", "r1_fires", "r1_right", "saved"]]
    return t, {"pooled": block(t), "train": block(tr), "test": block(te), "guns_with_r1": fired.to_dict("records")}


def early_warning():
    res, thresholds = {k: [] for k in ("champion", "GATE", "Q+", "Q-")}, {}
    guns = fail_gun = None
    for name, w, f in fl.iter_lab(LAB, META):
        fr = fe.build_frame(w, f)
        guns, fail_gun = fr.guns, fr.fail_gun
        thr = {"champion": fe.quantile_threshold(fr, fe.Q_REFERENCE), "Q+": fe.quantile_threshold(fr, fe.Q_PLUS),
               "Q-": fe.quantile_threshold(fr, fe.Q_MINUS)}
        for key, kind, th in (("champion", "model_ng", thr["champion"]), ("GATE", "model", thr["champion"]),
                              ("Q+", "model_ng", thr["Q+"]), ("Q-", "model_ng", thr["Q-"])):
            r = fe.evaluate(fr, fe.condition(fr, kind, th))
            r.onset = None                                  # 640k booleans per rule and run, not needed below
            res[key].append(r)
        thresholds[name] = thr
        log(f"lab {name}: thr q0.975 {thr['champion']:.4f}")
        del fr
    ev = {k: fe.pool(v, fail_gun, len(guns)) for k, v in res.items()}
    lab = {"champion": fe.summary(ev["champion"])}
    names = ("GATE", "Q+", "Q-")
    for x in names:
        lab[x] = {"summary": fe.summary(ev[x]), "vs_champion": fe.paired(ev[x], ev["champion"])}
    with open(REPORT1, encoding="utf-8") as fh:
        rep = json.load(fh)
    reported = {c["name"]: c for c in rep["early_warning"]["challengers"]}
    for x in names:
        assert abs(reported[x]["lab"]["d_champion"]["D"] - lab[x]["vs_champion"]["D"]) < 1e-9, f"{x}: not the D-9 report"
    log("sanity: lab D of GATE, Q+, Q- equal results/D-9/runs/offline-1/report.json")
    lab_mean = {k: float(np.mean([t[k] for t in thresholds.values()])) for k in ("champion", "Q+", "Q-")}

    windows, _, notes = fl.read_windows(STATE)
    outbox, more = fl.read_outbox(STATE)
    cases, events = fl.read_cases(history=HISTORY)
    failures = fl.failures_from(outbox, fl.case_index(cases, events, outbox), cases)
    fr = fe.build_frame(windows, failures[failures["status"] != "rejected"])
    n_guns = len(fr.guns)
    served = float(windows["ew_threshold"].dropna().iloc[0])
    ev_s = fe.pool([fe.evaluate(fr, fr.w["early_warning"].to_numpy(bool))], fr.fail_gun, n_guns)
    what_if = {"served": {"threshold": served, "summary": fe.summary(ev_s)}}
    for x, kind, thr in (("GATE", "model", served), ("Q+", "model_ng", lab_mean["Q+"]), ("Q-", "model_ng", lab_mean["Q-"])):
        ev_x = fe.pool([fe.evaluate(fr, fe.condition(fr, kind, thr))], fr.fail_gun, n_guns)
        what_if[x] = {"threshold": thr, "summary": fe.summary(ev_x), "vs_served": fe.paired(ev_x, ev_s)}
    assert abs(what_if["served"]["summary"]["fa_per_day"] - rep["early_warning"]["champion"]["field"]["fa_per_day"]) < 1e-9

    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    need = {}
    for x in names:
        sd = float(np.mean([np.nanstd((a.real - a.chance) - (b.real - b.chance), ddof=1)
                            for a, b in zip(res[x], res["champion"])]))
        d_lab = abs(lab[x]["vs_champion"]["D"])
        need[x] = {"sigma": sd, "lab_abs_D": d_lab,
                   "failures_for": {f"{d:.3f}": math.ceil((z * sd / d) ** 2) for d in (0.05, 0.10, d_lab)},
                   "power_at_64_for_lab_D": float(stats.norm.cdf(d_lab * math.sqrt(64) / sd - stats.norm.ppf(0.975)))}
    return {"loop_decision": {c["name"]: {"verdict": c["verdict"], "verdict_ko": c["verdict_ko"]} for c in reported.values()},
            "lab": lab, "lab_mean_thresholds": lab_mean, "test_what_if": what_if, "sample_sizes": need,
            "notes": notes + more}


def guns_needed(n_guns):
    """Smallest number of benefiting guns so that a gun resample holds none of them with probability < 2.5%."""
    return next(g for g in range(1, n_guns + 1) if (1 - g / n_guns) ** n_guns < 0.025)


def binomial_n(p0, gain=0.30, alpha=0.025, power=0.80, nmax=500):
    """Smallest n from which an exact one-sided binomial test keeps the power at p0 + gain, up to nmax."""
    p1 = min(p0 + gain, 0.999)
    last_bad = 0
    for n in range(1, nmax + 1):
        k = next((k for k in range(n + 2) if stats.binom.sf(k - 1, n, p0) <= alpha), n + 1)
        if stats.binom.sf(k - 1, n, p1) < power:
            last_bad = n
    return last_bad + 1


def main():
    t0 = time.time()
    t, crit = critical()
    p = crit["pooled"]
    log(f"profile pooled: false {p['profile']['false_before']} -> {p['profile']['false_after']}, saved share "
        f"{p['profile']['saved_share']:.3f} CI {p['profile']['ci']}, guns benefiting {p['profile']['guns_benefiting']}, "
        f"real suppressed {p['profile']['real_suppressed']}, exposed {p['profile']['exposed']} -> {p['profile']['verdict']}")
    log(f"R1 pooled: fires {p['r1']['fires']} on {p['r1']['guns_fired']} guns, precision {p['r1']['precision']}, base "
        f"{p['r1']['base_rate']:.3f}, gain CI {p['r1']['ci']}, coverage {p['r1']['coverage_of_false']:.3f}, "
        f"wrong {p['r1']['wrong']}, exposed {p['r1']['exposed']} -> {p['r1']['verdict']}")
    log(f"test only: profile {crit['test']['profile']['saved_share']:.3f} CI {crit['test']['profile']['ci']}; "
        f"R1 fires {crit['test']['r1']['fires']} on {crit['test']['r1']['guns_fired']} guns")
    ew = early_warning()
    log("loop decision: " + ", ".join(f"{k} {v['verdict']}" for k, v in ew["loop_decision"].items()))
    for x in ("GATE", "Q+", "Q-"):
        s, d = ew["lab"][x]["summary"], ew["lab"][x]["vs_champion"]
        log(f"lab {x}: FA {s['fa_per_day']:.2f} recall {s['recall']:.3f} D {d['D']:+.3f} {d['ci']} recall change "
            f"{d['recall_change']:+.3f} {d['recall_change_ci']}")
    s = ew["test_what_if"]["served"]["summary"]
    log(f"test served {ew['test_what_if']['served']['threshold']:.4f}: FA {s['fa_per_day']:.2f} recall {s['recall']:.3f} "
        f"chance {s['chance']:.3f} failures {s['failures']}")
    for x in ("GATE", "Q+", "Q-"):
        v = ew["test_what_if"][x]
        log(f"test what-if {x} @ {v['threshold']:.4f}: FA {v['summary']['fa_per_day']:.2f} recall {v['summary']['recall']:.3f} "
            f"D {v['vs_served']['D']:+.3f} {v['vs_served']['ci']} recall change {v['vs_served']['recall_change']:+.3f} "
            f"{v['vs_served']['recall_change_ci']}")
    log(f"sample sizes (failures for 80% power): {json.dumps(ew['sample_sizes'])}")
    needed = {"exposures_for_wrong_rate_under_10pct": MIN_EXPOSED, "guns_benefiting_pooled": guns_needed(p["guns"]),
              "r1_alarms_for_base_plus_30pts": binomial_n(p["r1"]["base_rate"])}
    log(f"sample sizes (critical, R1): {needed}")
    out = {"criteria": __doc__.split("PRE-REGISTERED")[1].split("    python")[0].strip(), "critical": crit,
           "early_warning": ew, "needed": needed, "seconds": round(time.time() - t0)}
    with open(os.path.join(HERE, "loop_effect.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    log(f"saved loop_effect.json ({out['seconds']}s)")


if __name__ == "__main__":
    main()
