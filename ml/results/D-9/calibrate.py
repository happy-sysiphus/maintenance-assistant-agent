"""
D-9 calibration (2026-11-08): set the designer defaults of the ML feedback loop spec
(docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md, 7.2 / 7.4 / 7.6) from the data instead of picking them.
The user approved the targets below before anything was computed ("이 목표로 돌리기", 2026-11-08).

PRE-REGISTERED before computing any result below (2026-11-08):
    data        choose on the 64 CV guns (B-19's 8 out-of-fold score runs; the 64 preprocessed train files for the
                rule); the 8 test files are a check only
    1 noise     champion = B-20 fixed rule (model_ng, k1, threshold = each run's 0.975 quantile of its normal CV scores).
                Per run: excess and false alarms per gun-day (B-18 Frame). Over the 28 pairs of runs:
                MARGIN = 95th percentile of |excess_r - excess_s|;
                FA noise = 95th percentile of |FA_r - FA_s| / mean(FA_r, FA_s); CHEAPER = 1 - FA noise
    2 quantiles Q- = the lowest q of B-18's grid whose pooled false alarms per gun-day are <= 24 at q and at every higher
                grid q; Q+ = the lowest grid q above 0.975 whose pooled false alarms per gun-day are <= CHEAPER x the
                champion's
    3 evidence  for n in (5, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64): 200 draws of (one of the 8 runs at random, n of the
                64 CV guns without replacement). Field verdict = spec 7.4 with MARGIN, CHEAPER, cap 24 and a paired gun
                bootstrap of 2000 on that run and those guns, for six pairs: X vs champion and champion vs X, X in
                (GATE, Q-, Q+). Reference verdict = the same rule on all 64 guns with the 8 runs pooled (spec 7.3,
                seed 7). False adoption(n) = share of draws whose verdict is pass, for a pair whose reference verdict is
                not pass. MIN_FAILURES = the smallest grid n whose false adoption is <= 0.05 for every such pair, at n
                and at every larger grid n. Pass rates of pairs whose reference verdict is pass: reported only (power)
    4 profile   terminal-code triggers at row level as main.Detector.rule_check (results/C-6/rule_rowlevel.py
                triggers: per-code cooldown, 24 h repeat; repeats never become cases). Unit = physical gun (B-19 ids:
                A-4 c7|c8|c9); the triggers of a gun's files merged in time, de-duplicated on (code, minute), a repeat
                if any copy is a repeat. A trigger is real if its class is the class of one of the gun's files and it
                fires 0-3600 s before that file's end (C-6 hit window), else false; operators label correctly.
                Spec 7.6 replayed in time order for k (cases) in 1..5 and d (days) in 1..3, starting with no profile:
                once a class is excluded, its later triggers are warnings (no case) - a real one is a wrong exclusion,
                a false one a saved false alarm; a class that would empty the profile is never excluded.
                Choose on the 64 train files: wrong exclusions = 0 (else the fewest), then the most saved, ties ->
                larger k, then larger d. Check the chosen (k, d) on the 8 test files
    not set by data  POST_FAILURE_S (every file ends 10 min after its failure), MIN_GUNS (one failure per gun), the
                phase-2 values (no intervention records)

Follow-up checks (added after the first run, so not part of the pre-registration):
    chances     how often the data could have shown a wrong exclusion at all: real failures preceded by a false trigger
                of the same class on the same physical gun (any k, d)
    creep       the "cheaper" path against the FIXED champion for every grid q above 0.975 (what repeated adoption could
                reach), with and without B-20's beyond-chance guard (the challenger's own excess CI lower bound > 0)

Rule v2 (user decision 2026-11-08 "보정 더하기", taken after the follow-ups; same draws and seeds as part 3):
    the cheaper path also needs no evidence of lost catches (D CI upper bound >= 0) and compares against the fixed
    B-20 operating point (in the lab that is the champion itself). MIN_FAILURES re-derived with part 3's procedure.
    Profile: k = 1 + one tolerated human mislabel = 2 (the data had no chance to show a wrong exclusion), d by part 4's
    tie-break among k = 2

    python results/D-9/calibrate.py     -> results/D-9/calibrate.json (+ .log via shell)

Inputs: results/B-19/cache/scores_cv_*.parquet and results/B-17/cache/meta.csv (through results/B-18/early_warning.py),
preprocessed/E0*.parquet and preprocessed/test/test_*.parquet (through results/C-6/rule_rowlevel.py),
results/A-4/cache/static_*.parquet. Writes only results/D-9/calibrate.json.
"""
import glob
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "results", "B-18"))
sys.path.insert(0, os.path.join(ROOT, "results", "C-6"))
import early_warning as EW  # noqa: E402
import rule_rowlevel as R6  # noqa: E402  (imports train as T)

T = R6.T
B19 = os.path.join(ROOT, "results", "B-19", "cache")
A4 = os.path.join(ROOT, "results", "A-4", "cache")
RUNS = [(sc, s) for s in (42, 1, 2, 3) for sc in ("file", "physical")]
QS = np.r_[np.arange(0.50, 0.95, 0.025), np.arange(0.95, 0.995, 0.005), [0.995, 0.997, 0.999]]   # B-18's grid
Q_CHAMP = 0.975
FA_CAP = 24
N_BOOT = 2000
SEED_SINGLE, SEED_PAIRED = 42, 7
N_GRID = (5, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
DRAWS = 200
SEED_SIM = 20261108
ALPHA = 0.05
KS, DS = (1, 2, 3, 4, 5), (1, 2, 3)
HIT_WINDOW_S = 3600
TOLERATED_MISLABELS = 1   # rule v2

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of all-NaN placebo slices, as in B-18 / A-5


def log(msg):
    print(msg, flush=True)


def qname(q):
    return f"q{q:.3f}"


def normal_thr(fr, q):
    w = fr.w
    return np.quantile(w.loc[(w["lead"] > EW.DAY) & (w["error_active"] == 0) & ~w["warmup"], "score_ens"], q)


def per_gun(fr, c):
    fa, hits = fr.evaluate(c, 1)
    ch = np.nanmean(np.vstack([hits[f"p{k}"] for k in EW.PLACEBO_K]), axis=0)
    return {"real": hits["real"], "chance": ch, "fa": fa.astype(float), "days": fr.days.astype(float)}


def stack(runs):
    real = np.nanmean(np.vstack([r["real"] for r in runs]), axis=0)
    ch = np.nanmean(np.vstack([r["chance"] for r in runs]), axis=0)
    fa = sum(r["fa"].sum() for r in runs) / sum(r["days"].sum() for r in runs)
    return real, ch, fa


def summary(runs):
    """results/A-5/overlap.py summary (seed 42)."""
    real, ch, fa = stack(runs)
    rng = np.random.default_rng(SEED_SINGLE)
    n = len(real)
    ex = [np.nanmean(real[i]) - np.nanmean(ch[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return {"fa_per_day": float(fa), "recall": float(np.nanmean(real)), "chance": float(np.nanmean(ch)),
            "excess": float(np.nanmean(real) - np.nanmean(ch)),
            "ci": [float(np.nanpercentile(ex, q)) for q in (2.5, 97.5)]}


def paired(a, b):
    """results/A-5/overlap.py paired (seed 7), unrounded: excess(a) - excess(b) over the same guns."""
    ra, ca, _ = stack(a)
    rb, cb, _ = stack(b)
    rng = np.random.default_rng(SEED_PAIRED)
    n = len(ra)
    d = lambda i: (np.nanmean(ra[i]) - np.nanmean(ca[i])) - (np.nanmean(rb[i]) - np.nanmean(cb[i]))  # noqa: E731
    boot = [d(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return float(d(np.arange(n))), [float(np.nanpercentile(boot, q)) for q in (2.5, 97.5)]


def verdict(ci, fa_ch, fa_cur, margin, cheaper, no_loss=False):
    """spec 7.4 pass / fail (no evidence floor). no_loss (rule v2): the cheaper path also needs ci[1] >= 0."""
    better = ci[0] > 0 and fa_ch <= FA_CAP
    cheap = ci[0] > -margin and fa_ch <= cheaper * fa_cur and (not no_loss or ci[1] >= 0)
    return "pass" if (better or cheap) else "fail"


def field_verdict(ch, cur, idx, rng, margin, cheaper, no_loss=False):
    """spec 7.4 on one run and a subset of guns (the field: one model), vectorised paired gun bootstrap."""
    rc, cc, ru, cu = ch["real"][idx], ch["chance"][idx], cur["real"][idx], cur["chance"][idx]
    b = rng.integers(0, len(idx), (N_BOOT, len(idx)))
    d = (np.nanmean(rc[b], axis=1) - np.nanmean(cc[b], axis=1)) - (np.nanmean(ru[b], axis=1) - np.nanmean(cu[b], axis=1))
    ci = (float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)))
    fa_ch = ch["fa"][idx].sum() / ch["days"][idx].sum()
    fa_cur = cur["fa"][idx].sum() / cur["days"][idx].sum()
    return verdict(ci, fa_ch, fa_cur, margin, cheaper, no_loss)


def reference(rules, rule_of, pair_keys, margin, cheaper, no_loss):
    """Reference verdicts: all 64 guns, 8 runs pooled (spec 7.3)."""
    ref = {}
    for ch, cur in pair_keys:
        d, ci = paired(rules[rule_of[ch]], rules[rule_of[cur]])
        fa_ch, fa_cur = stack(rules[rule_of[ch]])[2], stack(rules[rule_of[cur]])[2]
        ref[f"{ch} vs {cur}"] = {"challenger": ch, "current": cur, "D": d, "ci": ci, "fa_challenger": fa_ch,
                                 "fa_current": fa_cur, "verdict": verdict(ci, fa_ch, fa_cur, margin, cheaper, no_loss)}
    return ref


def simulate(ref, rules, rule_of, n_guns, margin, cheaper, no_loss, label, t0):
    """Part 3: pass rates by n over the same seeded draws; MIN_FAILURES by the pre-registered procedure."""
    counts = {key: {n: 0 for n in N_GRID} for key in ref}
    for n in N_GRID:
        for j in range(DRAWS):
            rng = np.random.default_rng([SEED_SIM, n, j])
            r = int(rng.integers(0, len(RUNS)))
            idx = np.sort(rng.choice(n_guns, size=n, replace=False))
            for key, v in ref.items():
                if field_verdict(rules[rule_of[v["challenger"]]][r], rules[rule_of[v["current"]]][r], idx, rng,
                                 margin, cheaper, no_loss) == "pass":
                    counts[key][n] += 1
        log(f"  {label} n {n:2d}: " + "  ".join(f"{key} {counts[key][n] / DRAWS:.3f}" for key in ref) +
            f"  ({time.time() - t0:.0f}s)")
    rates = {key: {n: c / DRAWS for n, c in v.items()} for key, v in counts.items()}
    not_pass = [key for key, v in ref.items() if v["verdict"] != "pass"]
    ok_n = [n for i, n in enumerate(N_GRID) if all(rates[k][m] <= ALPHA for k in not_pass for m in N_GRID[i:])]
    return rates, not_pass, (min(ok_n) if ok_n else None)


def physical_ids():
    """B-19 physical_cv.physical_ids (A-4 static values c7|c8|c9)."""
    st = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(A4, "static_*.parquet"))])
    key = st[["c7", "c8", "c9"]].round(4).astype(str).agg("|".join, axis=1)
    return dict(zip(st["file"], key))


def file_events(path):
    """(class, file end, [(time, code id, repeat)]) with the serving rule at row level (C-6)."""
    d = pd.read_parquet(path, columns=["class", "error_code", "ttf_s"])
    code = d["error_code"].map(R6.CODES).fillna(0).astype(int).to_numpy()
    t = (d.index - d.index[0]).total_seconds().to_numpy()
    end = d.index[0] + pd.Timedelta(seconds=float(d["ttf_s"].iloc[0]))
    return str(d["class"].iloc[0]), end, [(d.index[i], c, r) for i, c, r in R6.triggers(code, t, True)]


def merged_triggers(files):
    """A physical gun's triggers over its files, in time order, de-duplicated on (code, minute)."""
    merged = {}
    for _, _, _, ev in files:
        for tm, c, r in ev:
            kk = (c, tm.floor("min"))
            merged[kk] = [tm, c, (merged[kk][2] if kk in merged else False) or r]
    return sorted(merged.values(), key=lambda e: e[0])


def is_real(tm, cls, files):
    return any(cl == cls and 0 <= (end - tm).total_seconds() <= HIT_WINDOW_S for _, cl, end, _ in files)


def replay_profile(groups, k, dmin):
    """spec 7.6 replayed per physical gun in time order; groups: key -> [(file stem, class, end, events)]."""
    out = {"wrong": 0, "saved": 0, "false_cases": 0, "real_cases": 0, "exclusions": []}
    for _, files in sorted(groups.items()):
        excluded, nfa, days, nreal = set(), defaultdict(int), defaultdict(set), defaultdict(int)
        for tm, c, r in merged_triggers(files):
            if r:
                continue
            cls = T.CLASSES[c - 1]
            real = is_real(tm, cls, files)
            if cls in excluded:
                out["wrong" if real else "saved"] += 1
                continue
            if real:
                nreal[cls] += 1
                out["real_cases"] += 1
            else:
                nfa[cls] += 1
                days[cls].add(tm.date())
                out["false_cases"] += 1
            if nfa[cls] >= k and len(days[cls]) >= dmin and nreal[cls] == 0 and len(excluded) < len(T.CLASSES) - 1:
                excluded.add(cls)
                out["exclusions"].append({"files": sorted(stem for stem, _, _, _ in files), "class": cls,
                                          "at": str(tm), "false_cases": nfa[cls], "days": len(days[cls])})
    base = out["false_cases"] + out["saved"]
    out["saved_share"] = out["saved"] / base if base else 0.0
    return out


def wrong_chances(groups):
    """Real failures preceded by a false trigger of the same class on the same physical gun (follow-up)."""
    found = []
    for _, files in groups.items():
        seen_false = defaultdict(int)
        for tm, c, r in merged_triggers(files):
            if r:
                continue
            cls = T.CLASSES[c - 1]
            real = is_real(tm, cls, files)
            if real and seen_false[cls]:
                found.append({"files": sorted(stem for stem, _, _, _ in files), "class": cls, "at": str(tm)})
            if not real:
                seen_false[cls] += 1
    return {"count": len(found), "cases": found,
            "multi_file_guns": sorted(sorted(stem for stem, _, _, _ in fs) for fs in groups.values() if len(fs) > 1)}


def main():
    t0 = time.time()
    # ---- 1 + 2: per-run per-gun results for every grid quantile and GATE ----
    rules, guns = {}, None
    for sc, s in RUNS:
        fr = EW.Frame(EW.load("cv", os.path.join(B19, f"scores_cv_{sc}_s{s}.parquet")))
        guns = guns or list(fr.guns)
        assert list(fr.guns) == guns
        thrs = normal_thr(fr, QS)
        for q, thr in zip(QS, thrs):
            rules.setdefault(qname(q), []).append(per_gun(fr, fr.condition("model_ng", thr)))
        i_champ = int(np.argmin(np.abs(QS - Q_CHAMP)))
        rules.setdefault("GATE", []).append(per_gun(fr, fr.condition("model", thrs[i_champ])))
        log(f"{sc}_s{s}: {len(guns)} guns, thr q0.975 {thrs[i_champ]:.4f}  ({time.time() - t0:.0f}s)")
        del fr
    champ = qname(Q_CHAMP)
    s_champ = summary(rules[champ])
    log(f"champion pooled: fa {s_champ['fa_per_day']:.2f} recall {s_champ['recall']:.3f} chance {s_champ['chance']:.3f} "
        f"excess {s_champ['excess']:+.3f} [{s_champ['ci'][0]:+.3f}, {s_champ['ci'][1]:+.3f}]  (B-20: 16.40 0.488 0.343 "
        f"+0.146 [+0.059, +0.221])")
    d_gate, ci_gate = paired(rules["GATE"], rules[champ])
    log(f"GATE - champion: D {d_gate:+.3f} [{ci_gate[0]:+.3f}, {ci_gate[1]:+.3f}]  (11-08 probe: -0.066 [-0.128, -0.013])")

    per_run = [(float(np.nanmean(r["real"]) - np.nanmean(r["chance"])), float(r["fa"].sum() / r["days"].sum()))
               for r in rules[champ]]
    pairs = [(a, b) for a in range(len(per_run)) for b in range(a + 1, len(per_run))]
    dx = [abs(per_run[a][0] - per_run[b][0]) for a, b in pairs]
    dfa = [abs(per_run[a][1] - per_run[b][1]) / ((per_run[a][1] + per_run[b][1]) / 2) for a, b in pairs]
    margin = float(np.percentile(dx, 95))
    fa_noise = float(np.percentile(dfa, 95))
    cheaper = 1 - fa_noise
    log(f"\n1 noise over {len(pairs)} run pairs: per-run excess {[round(x, 3) for x, _ in per_run]}, "
        f"FA {[round(f, 2) for _, f in per_run]}")
    log(f"  MARGIN = p95 |d excess| = {margin:.4f}  (max {max(dx):.4f});  FA noise = p95 rel |d FA| = {fa_noise:.4f} "
        f"-> CHEAPER = {cheaper:.4f}")

    grid_fa = {float(q): stack(rules[qname(q)])[2] for q in QS}
    ok = [q for q in grid_fa if all(f <= FA_CAP for q2, f in grid_fa.items() if q2 >= q - 1e-12)]
    q_minus = min(ok)
    fa_champ = grid_fa[float(QS[int(np.argmin(np.abs(QS - Q_CHAMP)))])]
    cand = [q for q in grid_fa if q > Q_CHAMP + 1e-9 and grid_fa[q] <= cheaper * fa_champ]
    q_plus = min(cand) if cand else float(max(QS))
    log("\n2 grid pooled FA/gun-day: " + "  ".join(f"{q:.3f}:{f:.2f}" for q, f in grid_fa.items() if q >= 0.95))
    log(f"  Q- = {q_minus:.3f} (FA {grid_fa[q_minus]:.2f});  Q+ = {q_plus:.3f} (FA {grid_fa[q_plus]:.2f}, target <= "
        f"{cheaper * fa_champ:.2f})")

    # ---- reference verdicts + 3: evidence floor (pre-registered rule) ----
    xs = {"GATE": "GATE", "Q-": qname(q_minus), "Q+": qname(q_plus)}
    rule_of = {"champion": champ, **xs}
    pair_keys = [(x, "champion") for x in xs] + [("champion", x) for x in xs]
    ref = reference(rules, rule_of, pair_keys, margin, cheaper, no_loss=False)
    log("\nreference verdicts (64 guns, 8 runs pooled):")
    for key, v in ref.items():
        log(f"  {key:22s} D {v['D']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  FA {v['fa_challenger']:.2f} vs "
            f"{v['fa_current']:.2f}  -> {v['verdict']}")
    lab = {name: summary(rules[r]) for name, r in rule_of.items()}
    rates, not_pass, min_failures = simulate(ref, rules, rule_of, len(guns), margin, cheaper, False, "v1", t0)
    log(f"\n3 false adoption <= {ALPHA} for {not_pass} from n = {min_failures}")

    # ---- 4: profile thresholds ----
    phys = physical_ids()
    paths = {"train": sorted(glob.glob(os.path.join(ROOT, "preprocessed", "E0*.parquet"))),
             "test": sorted(glob.glob(os.path.join(ROOT, "preprocessed", "test", "test_*.parquet")))}
    groups = {}
    for split, ps in paths.items():
        g = defaultdict(list)
        for p in ps:
            stem = os.path.splitext(os.path.basename(p))[0]
            g[phys.get(stem, stem)].append((stem, *file_events(p)))
        groups[split] = g
        log(f"4 {split}: {len(ps)} files, {len(g)} physical guns, "
            f"{sum(len(e) for fs in g.values() for _, _, _, e in fs)} triggers  ({time.time() - t0:.0f}s)")
    prof = {split: {(k, dmin): replay_profile(groups[split], k, dmin) for k in KS for dmin in DS} for split in groups}
    order = sorted(prof["train"], key=lambda kd: (prof["train"][kd]["wrong"], -prof["train"][kd]["saved"], -kd[0], -kd[1]))
    chosen = order[0]
    log("  train (k, d): wrong / saved of false-alarm cases")
    for kd in sorted(prof["train"]):
        v, t = prof["train"][kd], prof["test"][kd]
        log(f"    k {kd[0]} d {kd[1]}: train wrong {v['wrong']} saved {v['saved']}/{v['saved'] + v['false_cases']} "
            f"({v['saved_share']:.2f}) | test wrong {t['wrong']} saved {t['saved']}/{t['saved'] + t['false_cases']}")
    log(f"  chosen on train: k {chosen[0]} d {chosen[1]} -> test wrong {prof['test'][chosen]['wrong']} saved "
        f"{prof['test'][chosen]['saved']}; test exclusions {prof['test'][chosen]['exclusions']}")

    # ---- follow-up (not pre-registered) ----
    chances = {split: wrong_chances(groups[split]) for split in groups}
    log("\nfollow-up chances (a wrong exclusion was possible at all): " +
        "  ".join(f"{split} {v['count']} (multi-file guns {v['multi_file_guns']})" for split, v in chances.items()))
    creep = {}
    for q in [q for q in grid_fa if q > Q_CHAMP + 1e-9]:
        s = summary(rules[qname(q)])
        d, ci = paired(rules[qname(q)], rules[champ])
        cheap_ok = ci[0] > -margin and s["fa_per_day"] <= cheaper * fa_champ
        creep[f"{q:.3f}"] = {"fa_per_day": s["fa_per_day"], "excess": s["excess"], "excess_ci": s["ci"], "D": d, "ci": ci,
                             "cheaper_pass": cheap_ok, "beyond_chance": s["ci"][0] > 0,
                             "pass_with_guard": cheap_ok and s["ci"][0] > 0, "pass_rule_v2": cheap_ok and ci[1] >= 0}
        log(f"  creep q{q:.3f}: fa {s['fa_per_day']:.2f} excess {s['excess']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] "
            f"D {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] -> cheaper {cheap_ok}, with guard {cheap_ok and s['ci'][0] > 0}, "
            f"rule v2 {cheap_ok and ci[1] >= 0}")

    # ---- rule v2 (user decision 2026-11-08 "보정 더하기") ----
    ref2 = reference(rules, rule_of, pair_keys, margin, cheaper, no_loss=True)
    log("\nrule v2 reference verdicts: " + "  ".join(f"{key} {v['verdict']}" for key, v in ref2.items()))
    rates2, not_pass2, min_failures2 = simulate(ref2, rules, rule_of, len(guns), margin, cheaper, True, "v2", t0)
    k2 = 1 + TOLERATED_MISLABELS
    chosen2 = sorted([kd for kd in prof["train"] if kd[0] == k2],
                     key=lambda kd: (prof["train"][kd]["wrong"], -prof["train"][kd]["saved"], -kd[1]))[0]
    log(f"rule v2: false adoption <= {ALPHA} for {not_pass2} from n = {min_failures2};  profile k {chosen2[0]} "
        f"d {chosen2[1]} -> train saved {prof['train'][chosen2]['saved']}, test wrong {prof['test'][chosen2]['wrong']} "
        f"saved {prof['test'][chosen2]['saved']}; test exclusions {prof['test'][chosen2]['exclusions']}")

    out = {"criteria": __doc__.split("PRE-REGISTERED")[1].split("    python")[0].strip(),
           "sanity": {"champion_pooled": s_champ, "gate_minus_champion": {"D": d_gate, "ci": ci_gate}},
           "noise": {"per_run": [{"run": f"{sc}_s{s}", "excess": x, "fa_per_day": f}
                                 for (sc, s), (x, f) in zip(RUNS, per_run)],
                     "margin": margin, "fa_noise": fa_noise, "cheaper": cheaper, "max_d_excess": max(dx)},
           "grid_fa_per_day": {f"{q:.3f}": f for q, f in grid_fa.items()},
           "quantiles": {"q_minus": q_minus, "q_plus": q_plus},
           "lab": lab, "reference": ref,
           "evidence": {"n_grid": list(N_GRID), "draws": DRAWS, "false_adoption_pairs": not_pass,
                        "pass_rate": rates, "min_failures": min_failures},
           "profile": {"grid": {split: {f"k{k}_d{d}": v for (k, d), v in res.items()} for split, res in prof.items()},
                       "chosen": {"k": chosen[0], "d": chosen[1]}, "test_check": prof["test"][chosen]},
           "follow_up": {"wrong_exclusion_chances": chances, "creep_vs_fixed_champion": creep},
           "rule_v2": {"reference": ref2, "false_adoption_pairs": not_pass2, "pass_rate": rates2,
                       "min_failures": min_failures2,
                       "profile": {"k": chosen2[0], "d": chosen2[1], "train": prof["train"][chosen2],
                                   "test_check": prof["test"][chosen2]}},
           "seconds": round(time.time() - t0)}
    with open(os.path.join(HERE, "calibrate.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False,
                  default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log(f"saved calibrate.json ({out['seconds']}s)")


if __name__ == "__main__":
    main()
