"""
D-10 R2 simulation (2026-11-09): what the card-order rule can do, under an ASSUMED mix of true causes. The data has no record
of which check was abnormal, so this is the mechanism, not performance. Uses the real api/field_memory.py (order_for,
apply_order, first_check) and the real mapping card; run with the api environment:

    .venv/Scripts/python.exe ml/results/D-10/r2_sim.py     -> ml/results/D-10/r2_sim.json (+ .log via shell)

PRE-REGISTERED before computing any result below (2026-11-09):
    setting   code E012 (two checks in two causes, no action guard on the mapping card). Every new case is resolved, at one
              of 8 guns drawn uniformly, and has exactly one truly abnormal check: the lower one with probability p
    worker    walks the card from the top, records "normal" until the true check, records it "abnormal" and stops. With
              probability b (top-item bias: the "next" button needs only one record) the worker instead records the top item
              abnormal and stops
    order     at each new case, order_for over the closed cases so far decides the card (once, at ingest, as the api does)
    measure   position of the true check on the card: mapping order (before) against R2's order (after), same draws. Saving
              per case = before - after, mean over the first N cases of a run
    grid      p in (0.1, 0.3, 0.5, 0.7, 0.9), b in (0, 0.3), N in (12, 50), 1000 runs per (p, b), seeded
    report    mean saving and the 2.5-97.5 percentile range over runs, share of runs where R2 fired within N cases, median case
              number of the first firing, share of cases seen with the moved card
    sanity    p = 0.5, b = 0: the share of runs with R2 on at some point after 12 closed cases is the spec's 38.8% (397/1024,
              spec 10) within 0.03
    no significance claim: every number follows from the assumed p and b
    study size  a real comparison of R2 on and off, cases randomly assigned: n per arm for "the first check is the true one"
              (1 - p against p), two-sided 5%, power 80%, for p in (0.6, 0.7, 0.9)
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "api"), str(REPO)]
import field_memory as FM  # noqa: E402
import guidance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "E012"
PS = (0.1, 0.3, 0.5, 0.7, 0.9)
BS = (0.0, 0.3)
NS = (12, 50)
RUNS = 1000
GUNS = 8
SEED = 20261109
Z = 1.959964 + 0.841621


def run_once(rng, p, b, n, checks, first_of):
    cases, before, after, moved, first_fire = [], np.zeros(n), np.zeros(n), np.zeros(n, bool), None
    for i in range(n):
        decision = FM.order_for(CODE, cases)
        top = first_of[decision["lead"]] if decision else checks[0]
        order = [top] + [c for c in checks if c != top]
        if top != checks[0]:
            moved[i] = True
            if first_fire is None:
                first_fire = i + 1
        true = checks[1] if rng.random() < p else checks[0]
        before[i], after[i] = checks.index(true) + 1, order.index(true) + 1
        if rng.random() < b:
            recs = {order[0]: {"result": "abnormal"}}
        else:
            recs = {}
            for c in order:
                recs[c] = {"result": "abnormal" if c == true else "normal"}
                if c == true:
                    break
        cases.append({"case_id": f"C{i}", "event": {"gun_id": f"g{int(rng.integers(GUNS))}", "trigger": {"rule_code": CODE}},
                      "records": {"closure": {"outcome": "resolved"}, "checks": recs}})
    return before, after, moved, first_fire


def main():
    t0 = time.time()
    checks = FM.mapping_checks(CODE)
    card = guidance.guidance_for({"trigger": {"rule_code": CODE}})
    assert FM.first_check(card) == checks[0], "the mapping card does not start with the first mapping check"
    first_of = {c: FM.first_check(FM.apply_order(card, c)) for c in checks}
    assert first_of[checks[1]] == checks[1], "the action guard blocks the move on this card"
    print(f"{CODE}: checks {checks}, the lower check can be moved first on the mapping card", flush=True)
    out = []
    for ip, p in enumerate(PS):
        for ib, b in enumerate(BS):
            nmax = max(NS) + 1
            mats = [run_once(np.random.default_rng([SEED, ip, ib, r]), p, b, nmax, checks, first_of) for r in range(RUNS)]
            before = np.vstack([m[0] for m in mats])
            after = np.vstack([m[1] for m in mats])
            moved = np.vstack([m[2] for m in mats])
            first = np.array([m[3] if m[3] is not None else 10 ** 6 for m in mats])
            row = {"p_lower": p, "top_bias": b, "by_n": {}}
            for n in NS:
                saving = (before[:, :n] - after[:, :n]).mean(axis=1)
                fired = first <= n
                row["by_n"][str(n)] = {
                    "mean_position_before": float(before[:, :n].mean()), "mean_position_after": float(after[:, :n].mean()),
                    "mean_saving": float(saving.mean()),
                    "range": [float(np.percentile(saving, 2.5)), float(np.percentile(saving, 97.5))],
                    "fired_share": float(fired.mean()),
                    "median_first_fire": float(np.median(first[fired])) if fired.any() else None,
                    "moved_case_share": float(moved[:, :n].mean())}
            row["on_after_12_closed"] = float((first <= 13).mean())
            out.append(row)
            v12, v50 = row["by_n"]["12"], row["by_n"]["50"]
            print(f"p {p} b {b}: N12 saving {v12['mean_saving']:+.3f} {np.round(v12['range'], 3)} fired {v12['fired_share']:.3f} | "
                  f"N50 saving {v50['mean_saving']:+.3f} {np.round(v50['range'], 3)} fired {v50['fired_share']:.3f} "
                  f"median first {v50['median_first_fire']} moved {v50['moved_case_share']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    sanity = next(r["on_after_12_closed"] for r in out if r["p_lower"] == 0.5 and r["top_bias"] == 0.0)
    assert abs(sanity - 397 / 1024) <= 0.03, f"on after 12 closed {sanity}, the spec computes {397 / 1024:.4f}"
    print(f"sanity: p 0.5, b 0: on after 12 closed {sanity:.3f} (spec 10: {397 / 1024:.3f})", flush=True)
    study = {}
    for p in (0.6, 0.7, 0.9):
        a, c = 1 - p, p
        study[str(p)] = math.ceil(Z ** 2 * (a * (1 - a) + c * (1 - c)) / (a - c) ** 2)
    print(f"study size per arm: {study}", flush=True)
    res = {"criteria": __doc__.split("PRE-REGISTERED")[1].split("    no significance")[0].strip(), "code": CODE, "checks": checks,
           "scenarios": out, "sanity_on_after_12_closed": sanity, "study_n_per_arm": study, "seconds": round(time.time() - t0)}
    with open(os.path.join(HERE, "r2_sim.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"saved r2_sim.json ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
