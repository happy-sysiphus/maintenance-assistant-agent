"""D-11 what-if: which model swaps would the replay of 2026-11-12 have made under looser decision rules?

Nothing is trained and the registry's decisions are not touched. The five versions of models/registry/ are scored on
the three shadow sets of the replay (scores the registry does not have yet are computed once and cached next to the
bundles), then every rule variant is replayed over the same three judgement dates. A swap changes who the champion
is at the next judgement, so the chain is followed, not just the three recorded pairs.

    python results/D-11/whatif_rules.py        ->  results/D-11/whatif_rules.json + a table on the console
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJECT_ROOT, ".py"))

import model_loop as ml  # noqa: E402
import model_registry as mr  # noqa: E402

EPS = 1e-9
# check 1: the gain B - A must exceed this many noise widths (None: the gain is not looked at)
MARGINS = {"> 우연 폭": 1.0, "> 우연 폭의 절반": 0.5, "> 0": 0.0, "> -우연 폭": -1.0}
MAJORITY = {"과반": True, "끔": False}  # check 2
RATIOS = {"1.2배": 1.2, "1.5배": 1.5, "2.0배": 2.0, "끔": None}  # check 4
NOW = ("> 우연 폭", "과반", "1.2배")


def checks(a, b, noise, margin, majority, ratio, cfg):
    """The five checks of model_loop.judge with three of them adjustable. Returns the names of the failed ones."""
    both = [(a["per_file"][f], b["per_file"][f]) for f in a["per_file"]
            if a["per_file"][f] is not None and b["per_file"].get(f) is not None]
    wins = sum(y >= x for x, y in both)
    gain = b["auroc"] - a["auroc"]
    ok = {"1": gain > margin * noise,
          "2": (not majority) or wins * 2 > len(both),
          "3": b["ew"] and b["hits"] >= a["hits"],
          "4": ratio is None or (not a["ew"]) or b["fa"] <= a["fa"] * ratio + EPS,
          "5": b["fa"] <= cfg.fa_cap + EPS}
    detail = {"gain": gain, "noise": noise, "wins": wins, "files": len(both), "hits_a": a["hits"], "hits_b": b["hits"],
              "fa_a": a["fa"], "fa_b": b["fa"], "fa_ratio": b["fa"] / a["fa"] if a["fa"] else None}
    return [k for k, v in ok.items() if not v], detail


def main():
    reg = mr.Registry(os.path.join(PROJECT_ROOT, "models", "registry"))
    data = reg.load()
    cfg = ml.LoopConfig()
    ev = ml.make_eval_fn(reg, os.path.join(PROJECT_ROOT, "preprocessed"), cfg)
    steps = [(d["at"][:10], d["challenger"], d["shadow_files"]) for d in data["decisions"]]
    first = data["decisions"][0]["champion"]

    def noise_of(b, shadow):
        s = [ev(b, None, shadow)["auroc"]] + [ev(b, x, shadow)["auroc"] for x in cfg.seeds[1:]]
        return max(s) - min(s)

    # every pair that any chain can reach: the champion at a step is the first model or an earlier challenger
    pairs = {}
    for i, (at, b, shadow) in enumerate(steps):
        for a in [first] + [s[1] for s in steps[:i]]:
            failed, detail = checks(ev(a, None, shadow), ev(b, None, shadow), noise_of(b, shadow), 1.0, True, 1.2, cfg)
            pairs[f"{at} {a} vs {b}"] = {"shadow_files": len(shadow), "failed_now": failed, **detail}

    rows = []
    for m, j, r in itertools.product(MARGINS, MAJORITY, RATIOS):
        champ, seq = first, []
        for at, b, shadow in steps:
            failed, _ = checks(ev(champ, None, shadow), ev(b, None, shadow), noise_of(b, shadow), MARGINS[m],
                               MAJORITY[j], RATIOS[r], cfg)
            seq.append({"at": at, "champion": champ, "challenger": b, "swap": not failed, "failed": failed})
            if not failed:
                champ = b
        rows.append({"check1": m, "check2": j, "check4": r, "changed": sum(x != y for x, y in zip((m, j, r), NOW)),
                     "swaps": sum(s["swap"] for s in seq), "final": champ, "steps": seq})
    rows.sort(key=lambda x: (x["changed"], x["swaps"]))

    out = os.path.join(HERE, "whatif_rules.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"note": "what-if on the replay of 2026-11-12; no training, the registry's decisions are unchanged",
                   "pairs": pairs, "variants": rows}, f, ensure_ascii=False, indent=1)

    print("pairs (current rule):")
    for k, p in pairs.items():
        ratio = "-" if p["fa_ratio"] is None else f"{p['fa_ratio']:.2f}x"
        print(f"  {k} ({p['shadow_files']} files): gain {p['gain']:+.4f} noise {p['noise']:.4f}  B>=A {p['wins']}/"
              f"{p['files']}  hits {p['hits_a']}->{p['hits_b']}  fa {p['fa_a']:.2f}->{p['fa_b']:.2f} ({ratio})  "
              f"failed {','.join(p['failed_now']) or '-'}")
    print("\nvariants (changed = how many of checks 1 / 2 / 4 differ from the current rule):")
    for x in rows:
        seq = "  ".join(f"{s['at'][5:]} {s['champion']}{'>' if s['swap'] else '|'}{s['challenger']}"
                        f"{'' if s['swap'] else '(' + ','.join(s['failed']) + ')'}" for s in x["steps"])
        print(f"  [{x['changed']}] 1: {x['check1']:<10} 2: {x['check2']:<3} 4: {x['check4']:<5} swaps {x['swaps']} "
              f"final {x['final']}   {seq}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
