"""
D-9 follow-up (2026-11-09): does MIN_FAILURES = 10 still hold now that the field judges Q+ / Q- at the threshold their
apply command would make (the lab mean) instead of the field's own quantile? User decision "바꾸기" (2026-11-09).

PRE-REGISTERED before computing any result below (2026-11-09):
    procedure   calibrate.py part 3 under rule v2: same draws, seeds, n grid, MARGIN, CHEAPER and reference verdicts.
                One change: in a draw on run r, Q+ and Q- use the mean of the OTHER 7 runs' thresholds at their quantile
                (a field is not one of the runs its lab mean comes from). The champion and GATE keep run r's 0.975.
    sanity      simulate() on each run's own quantiles must reproduce calibrate.json rule_v2.pass_rate exactly
    decision    MIN_FAILURES = 10 stands if false adoption <= 0.05 for every not-pass pair at n = 10 and at every larger
                grid n. Otherwise this script only reports it: the constant is the user's to change

    python results/D-9/apply_threshold_check.py     -> results/D-9/apply_threshold_check.json (+ .log via shell)

Inputs: results/B-19/cache/scores_cv_*.parquet, results/B-17/cache/meta.csv, results/D-9/calibrate.json.
Writes only results/D-9/apply_threshold_check.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate as CAL  # noqa: E402

MIN_FAILURES = 10


def grid(q):
    return float(CAL.QS[int(np.argmin(np.abs(CAL.QS - q)))])


def frames():
    for sc, s in CAL.RUNS:
        yield f"{sc}_s{s}", CAL.EW.Frame(CAL.EW.load("cv", os.path.join(CAL.B19, f"scores_cv_{sc}_s{s}.parquet")))


def main():
    t0 = time.time()
    with open(os.path.join(HERE, "calibrate.json"), encoding="utf-8") as fh:
        cal = json.load(fh)
    margin, cheaper = cal["noise"]["margin"], cal["noise"]["cheaper"]
    q_champ, q_minus, q_plus = grid(CAL.Q_CHAMP), grid(cal["quantiles"]["q_minus"]), grid(cal["quantiles"]["q_plus"])
    qs = np.array([q_champ, q_minus, q_plus])
    champ, minus, plus = CAL.qname(q_champ), CAL.qname(q_minus), CAL.qname(q_plus)
    rule_of = {"champion": champ, "GATE": "GATE", "Q-": minus, "Q+": plus}
    pair_keys = [(x, "champion") for x in ("GATE", "Q-", "Q+")] + [("champion", x) for x in ("GATE", "Q-", "Q+")]

    names, thr, guns = [], [], None
    for name, fr in frames():
        names.append(name)
        thr.append(CAL.normal_thr(fr, qs))
        guns = guns or list(fr.guns)
        del fr
    thr = np.array(thr)
    log = CAL.log
    log("per-run thresholds (champion, Q-, Q+): " + "; ".join(f"{n} {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}" for n, t in zip(names, thr)))

    own, loo = {}, {}
    for r, (name, fr) in enumerate(frames()):
        others = np.delete(thr, r, axis=0).mean(axis=0)
        for rules, t_minus, t_plus in ((own, thr[r, 1], thr[r, 2]), (loo, others[1], others[2])):
            rules.setdefault(champ, []).append(CAL.per_gun(fr, fr.condition("model_ng", thr[r, 0])))
            rules.setdefault("GATE", []).append(CAL.per_gun(fr, fr.condition("model", thr[r, 0])))
            rules.setdefault(minus, []).append(CAL.per_gun(fr, fr.condition("model_ng", t_minus)))
            rules.setdefault(plus, []).append(CAL.per_gun(fr, fr.condition("model_ng", t_plus)))
        log(f"{name}: Q- own {thr[r, 1]:.4f} / other runs {others[1]:.4f}, Q+ own {thr[r, 2]:.4f} / other runs "
            f"{others[2]:.4f}  ({time.time() - t0:.0f}s)")
        del fr

    ref = CAL.reference(own, rule_of, pair_keys, margin, cheaper, no_loss=True)
    assert {k: v["verdict"] for k, v in ref.items()} == {k: v["verdict"] for k, v in cal["rule_v2"]["reference"].items()}
    rates_own, not_pass, n_own = CAL.simulate(ref, own, rule_of, len(guns), margin, cheaper, True, "own", t0)
    expected = {k: {int(n): r for n, r in v.items()} for k, v in cal["rule_v2"]["pass_rate"].items()}
    assert rates_own == expected, "own-quantile rerun differs from calibrate.json rule_v2"
    assert n_own == cal["rule_v2"]["min_failures"]
    log(f"sanity: own-quantile rerun = calibrate.json rule_v2, MIN_FAILURES {n_own}")

    rates_loo, _, n_loo = CAL.simulate(ref, loo, rule_of, len(guns), margin, cheaper, True, "lab-mean", t0)
    worst = {n: max(rates_loo[k][n] for k in not_pass) for n in CAL.N_GRID}
    stands = all(worst[n] <= CAL.ALPHA for n in CAL.N_GRID if n >= MIN_FAILURES)
    log(f"not-pass pairs: {not_pass}")
    log("worst false adoption at the lab-mean thresholds: " + "  ".join(f"n{n} {v:.3f}" for n, v in worst.items()))
    log(f"MIN_FAILURES own {n_own}, lab-mean {n_loo}; {MIN_FAILURES} stands: {stands}")

    out = {"criteria": __doc__.split("PRE-REGISTERED")[1].split("    python")[0].strip(), "not_pass": not_pass,
           "thresholds": {n: {"champion": float(t[0]), "Q-": float(t[1]), "Q+": float(t[2])} for n, t in zip(names, thr)},
           "pass_rate_own": rates_own, "pass_rate_lab_mean": rates_loo, "worst_false_adoption_lab_mean": worst,
           "min_failures_own": n_own, "min_failures_lab_mean": n_loo, "ten_stands": stands,
           "seconds": round(time.time() - t0)}
    with open(os.path.join(HERE, "apply_threshold_check.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    log(f"wrote apply_threshold_check.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
