"""
D-11 model loop: retrain on the files that have ended, compare the new model with the serving one on files neither
has seen (shadow comparison), and switch the serving model when the new one is better.
Design: docs/superpowers/specs/2026-11-11-model-loop-design.md.

    champion    the model the server alarms with
    challenger  a newer model that only scores, until it is judged
    tick        one check at a time `now`: judge the challenger if enough shadow files have ended, train a new
                challenger if enough new files have ended

The registry (model_registry.py) holds the versions and every decision. Training and scoring run train.py in a
separate process, with the production recipe.

    python .py/model_loop.py replay --start 2021-08-16      # the files of preprocessed/ in time order, a tick a week;
                                                            # the same command again continues an interrupted run
    python .py/model_loop.py report --out results/D-11/runs/replay-1   # ticks, decisions, every version on test/
    python .py/model_loop.py tick [--now 2021-09-20] [--server-url http://localhost:8000]   # one tick (a scheduler)
    python .py/model_loop.py status
    python .py/model_loop.py rollback [--to v001] [--server-url ...]   # the champion before this one, by hand
    python .py/model_loop.py feedback pull --api-url http://localhost:9000   # D-12: the operator verdicts of the
                                                            # closed early-warning cases, for the next training

--registry (default models/registry) and --data-dir (default preprocessed) are taken from the project root. The
settings (--initial-min-files 16, --train-min-new 5, --shadow-min-files 5, --seeds 42,43,44, --fa-ratio-max 1.2,
--fa-cap 24, --recipe-file) are stored in the registry; a registry is used with the settings it was made with.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Callable

import pandas as pd

import model_feedback as mf
import model_registry as mr

# the recipe of the serving bundle (models/baseline_ensemble.joblib, 2026-11-01); "--seed" is added per model
DEFAULT_RECIPE: tuple[str, ...] = (
    "--model", "ensemble", "--ensemble-components", "iforest,lgbm", "--lgbm-weight", "0.5", "--window", "60",
    "--val-frac", "0.25", "--threshold-q", "0.99", "--sustain", "3", "--critical-sustain", "10",
    "--restart-hold-min", "30", "--restart-idle-windows", "30", "--alarm-max-non-welding", "0.5",
    "--exclude-non-welding", "--drop-features", "c19", "--gun-norm", "center", "--warmup-hours", "6",
    "--cv", "4", "--early-warning-rule", "model_ng_k1", "--early-warning-q", "0.975",
    "--early-warning-cooldown-min", "60",
)


# ------------------------------------------------------------- decision rule (spec 5)
CHECKS = ("auroc_margin", "file_majority", "ew_hits", "fa_ratio", "fa_cap")
EPS = 1e-9  # "at most 1.2 x" must not fail on floating-point noise


@dataclass(frozen=True)
class LoopConfig:
    """Spec 3 and 5. The registry stores these values and refuses a run with other ones."""
    initial_min_files: int = 16  # ended files before the first model is trained
    train_min_new: int = 5  # new ended files before a challenger is trained
    shadow_min_files: int = 5  # shadow files before a challenger is judged (and before a provisional swap is checked)
    seeds: tuple[int, ...] = (42, 43, 44)  # seeds[0]: the version itself, the others: the noise models
    fa_ratio_max: float = 1.2  # B's early-warning false alarms <= this x A's
    fa_cap: float = 24.0  # ... and <= this per gun-day (B-20)
    # check 1: the gain B - A must exceed this many noise widths. 0 = any gain (user decision 2026-11-12: a small
    # improvement is enough, the swap is provisional and checked afterwards); 1 = above the noise (the first rule)
    auroc_margin: float = 0.0
    # the checks that decide. The others are computed and recorded, but they do not stop a swap
    gates: tuple[str, ...] = ("auroc_margin", "ew_hits", "fa_cap")
    # D-12: seconds from the warning an operator confirmed that are not normal data (= the early-warning cooldown)
    feedback_span_s: int = 3600
    recipe: tuple[str, ...] = DEFAULT_RECIPE

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {**d, "seeds": list(self.seeds), "gates": list(self.gates), "recipe": list(self.recipe)}


def _num(v: Any) -> float | None:
    """A JSON number, or None for a missing value / NaN."""
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def metrics_summary(m: dict[str, Any]) -> dict[str, Any]:
    """The "test" block that train.py --evaluate writes -> what the decision rule reads.
    A bundle without an early warning counts as 0 hits and 0 false alarms."""
    ew = m.get("early_warning")
    hits = [_num(v.get("hit")) for v in (ew or {}).get("per_file", {}).values()]
    return {"auroc": _num(m.get("auroc_gun_mean")),
            "per_file": {f: _num(v.get("auroc")) for f, v in m.get("per_file", {}).items()},
            "ew": ew is not None,
            "hits": int(sum(h for h in hits if h is not None)),
            "fa": (_num(ew.get("false_alarms_per_gun_day")) or 0.0) if ew else 0.0}


def noise_side(gain: float | None, noise: float | None) -> str | None:
    """Where the gain B - A stands against the noise width, in the words of the records."""
    if gain is None or noise is None:
        return None
    return "우연 폭 밖" if gain > noise else "우연 폭보다 크게 낮음" if gain < -noise else "우연 폭 안"


def judge(a: dict[str, Any], b: dict[str, Any], seed_aurocs: list[float | None], cfg: LoopConfig) -> dict[str, Any]:
    """Champion a against challenger b on the same shadow files (metrics_summary dicts). seed_aurocs: the per-gun
    AUROC mean of b and of the models trained on b's data with other seeds; their spread is the noise the gain must
    exceed (cfg.auroc_margin times). All five checks are computed and recorded; the ones in cfg.gates decide. A
    challenger without an early warning never passes check 3: the serving model always has one (user decision
    2026-11-12)."""
    gates = [k for k in CHECKS if k in cfg.gates]
    seeds = [s for s in seed_aurocs if s is not None]
    noise = max(seeds) - min(seeds) if len(seeds) >= 2 else None
    both = [(a["per_file"][f], b["per_file"][f]) for f in a["per_file"]
            if a["per_file"][f] is not None and b["per_file"].get(f) is not None]
    wins = sum(y >= x for x, y in both)
    values = {"auroc_a": a["auroc"], "auroc_b": b["auroc"], "noise": noise, "files_b_ge_a": wins,
              "files_compared": len(both), "hits_a": a["hits"], "hits_b": b["hits"], "fa_a": a["fa"], "fa_b": b["fa"]}
    missing = [k for k in ("auroc_a", "auroc_b", "noise") if values[k] is None]
    if missing:
        return {"values": values, "checks": dict.fromkeys(CHECKS, False), "gates": gates, "above_noise": None,
                "swap": False, "reason": f"판정 불가: {', '.join(missing)} 없음"}
    gain = b["auroc"] - a["auroc"]
    need = cfg.auroc_margin * noise
    checks = {"auroc_margin": gain > need,
              "file_majority": wins * 2 > len(both),
              "ew_hits": b["ew"] and b["hits"] >= a["hits"],
              "fa_ratio": (not a["ew"]) or b["fa"] <= a["fa"] * cfg.fa_ratio_max + EPS,
              "fa_cap": b["fa"] <= cfg.fa_cap + EPS}
    detail = {"auroc_margin": (f"B-A {gain:.4f} <= 폭 {noise:.4f}" if cfg.auroc_margin == 1.0
                               else f"B-A {gain:.4f} <= {need:.4f}"),
              "file_majority": f"B>=A {wins}/{len(both)}",
              "ew_hits": f"B {b['hits']} < A {a['hits']}" if b["ew"] else "B에 조기 경보 없음",
              "fa_ratio": f"B {b['fa']:.2f} > A {a['fa']:.2f} x {cfg.fa_ratio_max}",
              "fa_cap": f"B {b['fa']:.2f} > {cfg.fa_cap}"}
    swap = all(checks[k] for k in gates)
    above = gain > noise
    if not swap:
        reason = ", ".join(f"{k}({detail[k]})" for k in gates if not checks[k])
    elif len(gates) == len(CHECKS):
        reason = "다섯 조건을 모두 만족"
    else:
        reason = f"조건 {' · '.join(str(CHECKS.index(k) + 1) for k in gates)} 만족"
    if cfg.auroc_margin != 1.0:
        reason += f"; B-A {gain:+.4f}는 {noise_side(gain, noise)}(폭 {noise:.4f})"
    aside = [f"{k}({detail[k]})" for k in CHECKS if k not in gates and not checks[k]]
    if aside:
        reason += "; 참고(판정에 쓰지 않음): " + ", ".join(aside)
    if not a["ew"]:
        reason += "; A에 조기 경보가 없어 오알람 비율 조건을 건너뜀"
    return {"values": values, "checks": checks, "gates": gates, "above_noise": above, "swap": swap, "reason": reason}


def probe(prev: dict[str, Any], new: dict[str, Any], seed_aurocs: list[float | None]) -> dict[str, Any]:
    """The check of a provisional swap on the files that ended after it (metrics_summary dicts; seed_aurocs as in
    judge, of the new champion). The swap is taken back when the new champion is worse than the one it replaced by
    more than the noise, and confirmed otherwise. revert None: no AUROC or no noise, nothing can be said yet."""
    seeds = [s for s in seed_aurocs if s is not None]
    noise = max(seeds) - min(seeds) if len(seeds) >= 2 else None
    values = {"auroc_prev": prev["auroc"], "auroc_new": new["auroc"], "noise": noise, "hits_prev": prev["hits"],
              "hits_new": new["hits"], "fa_prev": prev["fa"], "fa_new": new["fa"]}
    missing = [k for k in ("auroc_prev", "auroc_new", "noise") if values[k] is None]
    if missing:
        return {"values": values, "revert": None, "reason": f"확인 불가: {', '.join(missing)} 없음"}
    diff = new["auroc"] - prev["auroc"]
    revert = diff < -noise
    reason = (f"되돌림: 새 모델 - 이전 모델 {diff:+.4f}, 폭 {noise:.4f}보다 크게 낮음" if revert
              else f"확정: 새 모델 - 이전 모델 {diff:+.4f}, 폭 {noise:.4f}보다 크게 낮지 않음")
    return {"values": values, "revert": revert, "reason": reason}


# ------------------------------------------------------------- tick / replay (spec 3)
# (version, train file names, seeds) -> {"created", "bundle"}: seeds[0] trains the version, the others the noise models.
# D-12: with a verdict store a fourth argument follows, the spans of the operator verdicts on these files
# ({"confirmed": {file: [[start, end]]}, "false_alarm": {...}}), and the result may carry "feedback" (train.py's counts)
TrainFn = Callable[..., dict[str, Any]]
# (version, seed of a noise model or None for the version itself, file names) -> metrics_summary dict. D-12: the same
# fourth argument; the result may then carry "before" (the summary without the verdicts, None when no window was
# confirmed) and "feedback"
EvalFn = Callable[..., dict[str, Any]]
NotifyFn = Callable[[], None]  # called once after the champion changed
# D-12: file names -> {"spans", "counts", "event_ids"} (model_feedback.collect), None when there is no verdict store
FeedbackFn = Callable[[list[str]], "dict[str, Any] | None"]

QUIET = ("wait_first", "defer", "idle")  # tick results that change nothing


def parse_time(s: Any) -> pd.Timestamp:
    """A tick time as naive UTC (what manifest.csv holds): "2021-09-20", "...T00:00:00", "...T00:00:00Z"."""
    t = pd.Timestamp(s)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _iso(t: pd.Timestamp) -> str:
    return t.isoformat()


def load_manifest(data_dir: str) -> list[tuple[str, pd.Timestamp]]:
    """(file, end) of every preprocessed file that was kept, oldest end first."""
    path = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(path):
        raise ValueError(f"{path} not found - run preprocess.py first")
    m = pd.read_csv(path, dtype={"file": str, "dropped": str})
    kept = m[(m["dropped"].isna() | (m["dropped"].str.strip() == "")) & m["end"].notna()]
    twice = sorted(set(kept["file"][kept["file"].duplicated()]))
    if twice:
        raise ValueError(f"{path} lists {', '.join(twice)} more than once")
    return sorted(((f, parse_time(e)) for f, e in zip(kept["file"], kept["end"])), key=lambda x: (x[1], x[0]))


def _log(data: dict[str, Any], at: str, actions: list[str]) -> None:
    """Record what this tick did. A tick that is continued after a failure adds to its own entry."""
    if data["ticks"] and data["ticks"][-1]["at"] == at:
        data["ticks"][-1]["actions"].extend(actions)
        data["last_tick"] = at
    else:
        mr.record_tick(data, at, actions)


def _notify(notify_fn: NotifyFn | None) -> None:
    if notify_fn is None:
        return
    try:
        notify_fn()
    except Exception as e:  # noqa: BLE001 - the registry is the truth; the server follows on its next reload / start
        print(f"warning: could not tell the server about the new champion ({e}); "
              "it will pick it up on its next reload or restart", flush=True)


def _fb_record(fb: dict[str, Any] | None, block: dict[str, Any] | None) -> dict[str, Any] | None:
    """What a version, a decision or a probation keeps of the operator verdicts on its files (D-12 spec 6.3): the
    counts per verdict, the warnings whose spans were used and what train.py reports about the spans."""
    if fb is None:
        return None
    return {**fb["counts"], "event_ids": list(fb["event_ids"]), **(block or {})}


def tick(reg: mr.Registry, files: list[tuple[str, pd.Timestamp]], now: pd.Timestamp, cfg: LoopConfig,
         train_fn: TrainFn, eval_fn: EvalFn, notify_fn: NotifyFn | None = None,
         feedback_fn: FeedbackFn | None = None) -> list[str]:
    """One check at `now` (spec 3). Returns what it did: "wait_first:N", "first:v001", "defer:N", "swap:A>B",
    "keep:A|B", "train:v00N", "confirm:B" / "revert:B>A" (a provisional swap checked, spec 6) or "idle". The
    registry is saved only after a step is complete, so a failed training or scoring leaves it as it was and the
    next call does that step again.
    feedback_fn (D-12): the operator verdicts on a list of files. With it, a training and a judgement get the spans
    of the confirmed warnings (their windows are not normal data) and the records say what was used. The figures a
    decision is made on are the ones with the verdicts; the ones without them are kept beside ("values_before").
    Verdicts alone start no training."""
    with reg.lock():
        data = reg.init(cfg.as_dict())
        at = _iso(now)
        if data["last_tick"] is not None and now < parse_time(data["last_tick"]):
            raise ValueError(f"tick time {at} is before the last tick {data['last_tick']}")
        ended = [(f, e) for f, e in files if e <= now]
        names = [f for f, _ in ended]
        done: list[str] = []

        def commit(action: str) -> None:
            _log(data, at, [action])
            reg.save(data)
            done.append(action)

        def quiet(action: str) -> list[str]:
            if not (data["ticks"] and data["ticks"][-1]["at"] == at):  # the same tick again writes nothing
                commit(action)
            return [action]

        def train(version: str, fs: list[str], seeds: list[int]) -> tuple[dict[str, Any], dict[str, Any] | None]:
            fb = feedback_fn(fs) if feedback_fn is not None else None
            out = train_fn(version, fs, seeds) if fb is None else train_fn(version, fs, seeds, fb["spans"])
            return out, _fb_record(fb, out.get("feedback"))

        def scorer(fs: list[str]) -> tuple[dict[str, Any] | None, Callable[[str, int | None], dict[str, Any]]]:
            """The verdicts on these files and the scoring of one model on them (all models on the same spans)."""
            fb = feedback_fn(fs) if feedback_fn is not None else None

            def score(version: str, seed: int | None) -> dict[str, Any]:
                return eval_fn(version, seed, fs) if fb is None else eval_fn(version, seed, fs, fb["spans"])

            return fb, score

        def before(ms: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
            """The same models without the verdicts, when the scoring gave those figures for the first two."""
            return [m.get("before") or {} for m in ms] if all(m.get("before") for m in ms[:2]) else None

        if data["champion"] is None:
            if len(ended) < cfg.initial_min_files:
                return quiet(f"wait_first:{len(ended)}")
            v = mr.next_version(data)
            out, fb_rec = train(v, names, list(cfg.seeds[:1]))
            mr.add_version(data, v, cutoff=at, train_files=names, seed=cfg.seeds[0], created=out["created"],
                           bundle=out["bundle"], status="champion", feedback=fb_rec)
            mr.set_champion(data, v, at=at, by="first")
            commit(f"first:{v}")
            return done

        prov = data["provisional"]
        if prov is not None:
            # a swap is provisional until the files that ended after it have been seen (spec 6). Before the
            # challenger is judged: it must meet the champion that this check leaves in place
            new, prev = prov["version"], prov["previous"]
            used = set(data["versions"][new]["train_files"]) | set(prov["shadow_files"])
            fresh = [f for f in names if f not in used]
            if len(fresh) >= cfg.shadow_min_files:
                fb, score = scorer(fresh)
                ms = [score(prev, None), score(new, None)] + [score(new, s) for s in cfg.seeds[1:]]
                res = probe(ms[0], ms[1], [m["auroc"] for m in ms[1:]])
                if res["revert"] is not None:  # None: nothing can be said on these files, the next tick tries again
                    bs = before(ms)
                    was = probe(bs[0], bs[1], [m.get("auroc") for m in bs[1:]]) if bs else None
                    mr.record_probation(data, {"at": at, "version": new, "previous": prev, "files": fresh,
                                               "by": "auto", **res, "feedback": _fb_record(fb, ms[1].get("feedback")),
                                               "values_before": was and was["values"],
                                               "revert_before": was and was["revert"]})
                    mr.clear_provisional(data)
                    if res["revert"]:
                        mr.set_champion(data, prev, at=at, by="probation")
                        mr.reject(data, new)
                        commit(f"revert:{new}>{prev}")
                        _notify(notify_fn)
                    else:
                        commit(f"confirm:{new}")

        if data["challenger"] is not None:
            a, b = data["champion"], data["challenger"]
            # files neither model was trained on. By membership, not by `end > cutoff`: a file that ended before b was
            # trained but reached the folder after it is unseen too (in a replay the two are the same)
            seen = set(data["versions"][b]["train_files"])
            shadow = [f for f in names if f not in seen]
            if len(shadow) < cfg.shadow_min_files:
                if not done:
                    return quiet(f"defer:{len(shadow)}")
                commit(f"defer:{len(shadow)}")
                return done
            fb, score = scorer(shadow)
            ms = [score(a, None), score(b, None)] + [score(b, s) for s in cfg.seeds[1:]]
            bs = before(ms)
            was = judge(bs[0], bs[1], [m.get("auroc") for m in bs[1:]], cfg) if bs else None
            decision = {"at": at, "champion": a, "challenger": b, "shadow_files": shadow,
                        **judge(ms[0], ms[1], [m["auroc"] for m in ms[1:]], cfg),
                        "feedback": _fb_record(fb, ms[1].get("feedback")),
                        "values_before": was and was["values"],
                        # no width without the noise models' figures: what the conclusion was cannot be said
                        "swap_before": was and (None if was["above_noise"] is None else was["swap"])}
            mr.record_decision(data, decision)
            if decision["swap"]:
                mr.set_champion(data, b, at=at, by="decision")
                mr.set_provisional(data, b, previous=a, since=at, shadow_files=shadow)
                commit(f"swap:{a}>{b}")
                _notify(notify_fn)
            else:
                mr.reject(data, b)
                commit(f"keep:{a}|{b}")

        seen = set(data["versions"][mr.latest_version(data)]["train_files"])  # of the version trained last
        if sum(f not in seen for f in names) >= cfg.train_min_new:
            v = mr.next_version(data)
            out, fb_rec = train(v, names, list(cfg.seeds))
            mr.add_version(data, v, cutoff=at, train_files=names, seed=cfg.seeds[0], created=out["created"],
                           bundle=out["bundle"], status="challenger", feedback=fb_rec)
            mr.set_challenger(data, v)
            commit(f"train:{v}")
        return done or quiet("idle")


def replay(reg: mr.Registry, files: list[tuple[str, pd.Timestamp]], start: pd.Timestamp, end: pd.Timestamp | None,
           interval_days: int, cfg: LoopConfig, train_fn: TrainFn, eval_fn: EvalFn,
           notify_fn: NotifyFn | None = None, feedback_fn: FeedbackFn | None = None) -> list[tuple[str, list[str]]]:
    """Ticks at start + k * interval_days (k = 1, 2, ...) up to `end`, or up to the first tick at or after the last
    file's end. Ticks the registry already has are skipped, so the same call continues an interrupted run; the last
    recorded tick is run again because it may have stopped half way (it does nothing if it was complete)."""
    if interval_days < 1:
        raise ValueError("the tick interval must be at least 1 day")
    if end is None and not files:
        raise ValueError("no preprocessed files to replay")
    stop = end if end is not None else max(e for _, e in files)
    data = reg.load()
    last = parse_time(data["last_tick"]) if data and data["last_tick"] else None
    out: list[tuple[str, list[str]]] = []
    k = 1
    while True:
        now = start + pd.Timedelta(days=interval_days * k)
        if end is not None and now > end:
            break
        if last is None or now >= last:
            actions = tick(reg, files, now, cfg, train_fn, eval_fn, notify_fn, feedback_fn)
            if last is None or now > last or not all(a.split(":")[0] in QUIET for a in actions):
                out.append((_iso(now), actions))
                print(f"{_iso(now)}  {'  '.join(actions)}", flush=True)
        if end is None and now >= stop:
            break
        k += 1
    return out


# ------------------------------------------------------------- training and scoring through train.py
PY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PY_DIR)
TRAIN_PY = os.path.join(PY_DIR, "train.py")
CLASSES = ("E01", "E02", "E03", "E04")
# what a noise model does not need - it is only scored for its per-gun AUROC (option -> number of values it takes)
SEED_DROP = {"--cv": 1, "--early-warning-rule": 1, "--early-warning-q": 1, "--early-warning-cooldown-min": 1,
             "--no-early-warning": 0}


def model_name(recipe: Any) -> str:
    r = list(recipe)
    return r[r.index("--model") + 1] if "--model" in r else "ensemble"


def seed_recipe(recipe: Any) -> list[str]:
    """The recipe of a noise model: the same model without the cross-validation and the early-warning calibration."""
    out, skip = [], 0
    for a in recipe:
        if skip:
            skip -= 1
        elif a in SEED_DROP:
            skip = SEED_DROP[a]
        else:
            out.append(a)
    return out + ["--no-early-warning"]


def _read_names(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _write_names(path: str, names: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")


def _has_spans(spans: dict[str, Any] | None) -> bool:
    """Is there any operator verdict to hand to train.py (D-12)? Without one, --feedback is not passed at all."""
    return bool(spans) and any(spans.get(k) for k in ("confirmed", "false_alarm"))


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def can_reuse(folder: str, names: list[str], model: str, spans: dict[str, Any] | None = None) -> bool:
    """A finished training of exactly these files, with exactly these operator verdicts, is in the folder (train.py
    writes the metrics JSON last; <folder>/feedback.json holds the spans it was given, and is absent without any).
    A feedback.json that cannot be read (a write cut short) is no such training: the folder is trained again."""
    need = [os.path.join(folder, n) for n in (f"baseline_{model}.joblib", f"baseline_{model}_metrics.json",
                                              "train_files.txt")]
    if not all(os.path.exists(p) for p in need) or _read_names(need[2]) != list(names):
        return False
    fb = os.path.join(folder, "feedback.json")
    had = None
    if os.path.exists(fb):
        try:
            with open(fb, encoding="utf-8") as f:
                had = json.load(f)
        except (OSError, ValueError):
            return False
    return had == (spans if _has_spans(spans) else None)


def _train_py(args: list[Any], what: str) -> None:
    """Run train.py; what it prints goes straight to the console. A failure raises with the end of its stderr."""
    print(f"--- {what}", flush=True)
    r = subprocess.run([sys.executable, TRAIN_PY, *map(str, args)], stderr=subprocess.PIPE, text=True,
                       errors="replace")
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"{what} failed (train.py exit {r.returncode}):\n{tail}")


def _check_bundle(path: str) -> None:
    """A version is registered only when its bundle loads the way the server loads it (spec 8.1). In a process of
    its own, so that this one stays small."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); import train; train.load_bundle(sys.argv[2])"
    r = subprocess.run([sys.executable, "-c", code, PY_DIR, path], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"{path} was written but does not load:\n{tail}")


def _need_early_warning(version: str, block: dict[str, Any] | None) -> None:
    """The serving model always has an early warning (user decision 2026-11-12), so a version without one is not
    registered - the first model included, which no decision would stop. block: the metrics JSON's early_warning."""
    if block and block.get("enabled"):
        return
    if not block:
        why = "the recipe has it turned off: no --early-warning-rule, or --no-early-warning"
    elif block.get("reason"):
        why = block["reason"]
    else:
        why = f"only {(block.get('calibration') or {}).get('n_windows')} calibration windows, no threshold"
    raise RuntimeError(f"{version} was trained but has no early warning ({why}). It is not registered: the loop only "
                       "takes models with an early warning")


def make_train_fn(reg: mr.Registry, data_dir: str, cfg: LoopConfig) -> TrainFn:
    model = model_name(cfg.recipe)

    def one(folder: str, names: list[str], recipe: list[str], seed: int, spans: dict[str, Any] | None) -> None:
        metrics = os.path.join(folder, f"baseline_{model}_metrics.json")
        label = os.path.relpath(folder, reg.root)
        if can_reuse(folder, names, model, spans):
            print(f"--- {label}: already trained on these files, kept", flush=True)
            return
        os.makedirs(folder, exist_ok=True)
        fb = os.path.join(folder, "feedback.json")
        for stale in (metrics, os.path.join(folder, f"baseline_{model}.joblib"),
                      os.path.join(folder, "test_metrics.json"), fb):
            if os.path.exists(stale):
                os.remove(stale)
        shutil.rmtree(os.path.join(folder, "eval"), ignore_errors=True)  # the scores of the bundle that was here
        lst = os.path.join(folder, "train_files.txt")
        _write_names(lst, names)
        extra: list[Any] = []
        if _has_spans(spans):  # D-12: the windows of the confirmed spans are not normal data
            _write_json(fb, spans)
            extra = ["--feedback", fb]
        _train_py([*recipe, "--seed", seed, "--data-dir", data_dir, "--model-dir", folder, "--files-from", lst,
                   "--no-test", *extra],
                  f"training {label} on {len(names)} files (seed {seed})" + (", with operator verdicts" if extra else ""))
        try:
            _check_bundle(os.path.join(folder, f"baseline_{model}.joblib"))
        except RuntimeError:
            os.remove(metrics)  # not a finished training: the next run trains it again
            raise

    def train(version: str, names: list[str], seeds: list[int], spans: dict[str, Any] | None = None) -> dict[str, Any]:
        folder = reg.version_dir(version)
        one(folder, names, list(cfg.recipe), seeds[0], spans)
        with open(os.path.join(folder, f"baseline_{model}_metrics.json"), encoding="utf-8") as f:
            metrics = json.load(f)
        _need_early_warning(version, metrics.get("early_warning"))  # before the noise models; the training is kept
        for s in seeds[1:]:
            one(os.path.join(folder, "seeds", f"s{s}"), names, seed_recipe(cfg.recipe), s, spans)
        return {"created": metrics["created"], "bundle": f"{version}/baseline_{model}.joblib",
                "feedback": (metrics.get("metrics") or {}).get("feedback")}

    return train


def _evaluate_to(out: str, args: list[Any], what: str) -> dict[str, Any]:
    """What train.py --evaluate writes ("test", and with operator verdicts "test_no_feedback" and "feedback"),
    computed once and kept in `out`."""
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        _train_py(["--evaluate", *args, "--eval-out", tmp], what)
        os.replace(tmp, out)
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def make_eval_fn(reg: mr.Registry, data_dir: str, cfg: LoopConfig) -> EvalFn:
    model = model_name(cfg.recipe)

    def evaluate(version: str, seed: int | None, names: list[str], spans: dict[str, Any] | None = None) -> dict[str, Any]:
        """metrics_summary of the "test" block, plus "before" (the summary without the operator verdicts, None when
        no window was confirmed) and "feedback" (train.py's counts, None without spans). The key of the kept result
        holds the spans, so other verdicts are scored again; without spans it is the key from before D-12."""
        folder = reg.version_dir(version)
        if seed is not None:
            folder = os.path.join(folder, "seeds", f"s{seed}")
        use = _has_spans(spans)
        text = "\n".join(sorted(names)) + ("\n" + json.dumps(spans, sort_keys=True) if use else "")
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        out = os.path.join(folder, "eval", f"{key}.json")
        lst = os.path.join(folder, "eval", f"{key}.txt")
        fb = os.path.join(folder, "eval", f"{key}.feedback.json")
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            _write_names(lst, names)
            if use:
                _write_json(fb, spans)
        j = _evaluate_to(out, ["--model", model, "--data-dir", data_dir, "--model-dir", folder,
                               "--eval-files-from", lst, *(["--feedback", fb] if use else [])],
                         f"scoring {os.path.relpath(folder, reg.root)} on {len(names)} shadow files")
        before = j.get("test_no_feedback")
        return {**metrics_summary(j["test"]), "before": metrics_summary(before) if before else None,
                "feedback": j.get("feedback")}

    return evaluate


def make_notify_fn(server_url: str | None) -> NotifyFn | None:
    if not server_url:
        return None

    def notify() -> None:
        import httpx  # only a run with --server-url needs it

        r = httpx.post(server_url.rstrip("/") + "/model/reload", timeout=60)
        r.raise_for_status()
        print(f"server: {r.json()}", flush=True)

    return notify


# ------------------------------------------------------------- report (spec 7, 8.2)
def build_report(reg: mr.Registry, data_dir: str, cfg: LoopConfig) -> dict[str, Any]:
    """Ticks, versions and decisions of the registry, and every version scored on the test split. The test files
    are never used for a decision; they answer whether the loop helped."""
    data = _existing(reg)
    model = model_name(cfg.recipe)
    test_dir = os.path.join(data_dir, "test")
    n_test = len(glob.glob(os.path.join(test_dir, "test_*.parquet")))
    versions: dict[str, Any] = {}
    for v, info in data["versions"].items():
        test = None
        if n_test:
            m = _evaluate_to(os.path.join(reg.version_dir(v), "test_metrics.json"),
                             ["--model", model, "--data-dir", data_dir, "--model-dir", reg.version_dir(v)],
                             f"scoring {v} on the {n_test} test files")["test"]
            ew = m.get("early_warning") or {}
            test = {"auroc": _num(m.get("auroc")), "auroc_gun_mean": _num(m.get("auroc_gun_mean")),
                    "ew_recall": _num(ew.get("recall")), "ew_chance": _num(ew.get("chance")),
                    "ew_false_alarms_per_gun_day": _num(ew.get("false_alarms_per_gun_day"))}
        names = info["train_files"]
        versions[v] = {"cutoff": info["cutoff"], "status": info["status"], "created": info["created"],
                       "n_train_files": len(names),
                       "by_class": {c: sum(n.startswith(c + "_") for n in names) for c in CLASSES}, "test": test}
    first = data["history"][0]["champion"] if data["history"] else None
    final = data["champion"]
    first_vs_final = None
    if first and final:
        a = (versions[first]["test"] or {}).get("auroc_gun_mean")
        b = (versions[final]["test"] or {}).get("auroc_gun_mean")
        first_vs_final = {"first": first, "final": final, "first_auroc_gun_mean": a, "final_auroc_gun_mean": b,
                          "diff": None if a is None or b is None else b - a}
    return {"registry": reg.root, "config": data["config"], "champion": final, "challenger": data["challenger"],
            "test_files": n_test, "note": None if n_test else f"no test_*.parquet in {test_dir}: no test table",
            "ticks": data["ticks"], "versions": versions, "decisions": data["decisions"], "history": data["history"],
            "provisional": data["provisional"], "probations": data["probations"], "first_vs_final": first_vs_final,
            # D-12: the operator verdicts each version was trained with and each decision / probation was counted with
            "feedback": {"versions": {v: info.get("feedback") for v, info in data["versions"].items()},
                         "decisions": [{k: d.get(k) for k in ("at", "champion", "challenger", "feedback", "values",
                                                              "values_before", "swap", "swap_before")}
                                       for d in data["decisions"]],
                         "probations": [{k: p.get(k) for k in ("at", "version", "previous", "feedback", "values",
                                                               "values_before", "revert", "revert_before")}
                                        for p in data["probations"]]}}


def _f(x: Any, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _table(head: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def _decision_markdown(d: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    v, c = d["values"], d["checks"]
    gates = d.get("gates", list(CHECKS))  # a decision recorded before 2026-11-12: all five decided

    def mark(k: str) -> str:
        word = "통과" if c[k] else "실패"
        return word if k in gates else f"참고: {word}"

    margin = cfg.get("auroc_margin", 1.0)
    first = "1 AUROC 차 > 우연 폭" if margin == 1.0 else ("1 AUROC 차 > 0" if margin == 0
                                                         else f"1 AUROC 차 > 우연 폭 x {margin}")
    side = ""
    if "above_noise" in d and v["auroc_a"] is not None and v["auroc_b"] is not None and v["noise"] is not None:
        side = f", {noise_side(v['auroc_b'] - v['auroc_a'], v['noise'])}"
    verdict = "교체" if d["swap"] else "유지"
    rows = [[first, f"B {_f(v['auroc_b'], 4)} - A {_f(v['auroc_a'], 4)}, 폭 {_f(v['noise'], 4)}{side}",
             mark("auroc_margin")],
            ["2 파일 과반에서 B >= A", f"{v['files_b_ge_a']} / {v['files_compared']}", mark("file_majority")],
            ["3 B에 조기 경보가 있고 적중 B >= A", f"B {v['hits_b']}, A {v['hits_a']}", mark("ew_hits")],
            [f"4 오알람 B <= A x {cfg['fa_ratio_max']}", f"B {_f(v['fa_b'], 2)}, A {_f(v['fa_a'], 2)}",
             mark("fa_ratio")],
            [f"5 오알람 B <= {cfg['fa_cap']}", f"B {_f(v['fa_b'], 2)}", mark("fa_cap")]]
    return [f"### {d['at']} · {d['champion']} 대 {d['challenger']} → {verdict}", "",
            f"섀도 파일 {len(d['shadow_files'])}건: {', '.join(d['shadow_files'])}", "",
            *_table(["조건", "값", "결과"], rows), "", f"사유: {d['reason']}", ""]


def _feedback_markdown(fb: dict[str, Any]) -> list[str]:
    """The "사람 판정" section (D-12 spec 6.3): what each version took in, and the decisions the verdicts moved."""
    out = ["## 사람 판정", ""]

    def matched(f: Any) -> bool:
        """Did any verdict land on the files of this record? A store whose cases match no file gives only zeros."""
        return bool(f) and any(f.get(k) for k in ("confirmed", "false_alarm", "unknown"))

    records = [*(fb.get("versions") or {}).values(), *(d.get("feedback") for d in fb.get("decisions") or []),
               *(p.get("feedback") for p in fb.get("probations") or [])]
    versions = {v: f for v, f in (fb.get("versions") or {}).items() if matched(f)}
    decisions = [d for d in fb.get("decisions") or [] if matched(d.get("feedback"))]
    probations = [p for p in fb.get("probations") or [] if matched(p.get("feedback"))]
    if not versions and not decisions and not probations:
        unmatched = max((f.get("unmatched") or 0 for f in records if f), default=0)
        tail = (f" 가져온 판정 가운데 {unmatched}건은 어느 전처리 파일에도 맞지 않았습니다(설비 이름이나 경보 시각이 파일과 "
                "다름).") if unmatched else ""
        return out + ["반영된 사람 판정이 없습니다." + tail, ""]
    out += ["작업자가 닫은 조기 경보 케이스의 판정이다. \"이상 확인\"(해결로 닫혔고 점검에 이상이 있음)인 경보는 그 시각부터 "
            "정해진 시간 동안을 학습과 채점에서 정상으로 치지 않았다. 오알람 · 모름 · 못 맞춤은 기록만 했다. 사람 판정은 "
            "지금 모델이 경보를 낸 구간에만 생기므로 A/B 판정을 지금 모델 쪽으로 기울인다.", "",
            *_table(["버전", "이상 확인", "오알람", "모름", "못 맞춤", "뺀 창 수"],
                    [[v, f.get("confirmed", 0), f.get("false_alarm", 0), f.get("unknown", 0), f.get("unmatched", 0),
                      f.get("windows_excluded", "-")] for v, f in versions.items()]), ""]

    def gain(v: dict[str, Any]) -> str:
        a, b = v.get("auroc_a"), v.get("auroc_b")
        return "-" if a is None or b is None else f"{b - a:+.4f}"

    def word(swap: Any) -> str:
        return "-" if swap is None else ("교체" if swap else "유지")

    moved = [d for d in decisions if d.get("values_before")]
    if moved:
        out += ["섀도 파일에 이상 확인 구간이 있던 A/B 판정이다. 교체 여부는 반영 후 수치로 정했다.", "",
                *_table(["시점", "판정", "AUROC 차 B-A (전 → 후)", "오알람 B (전 → 후)", "결론 (전 → 후)"],
                        [[d["at"], f"{d['champion']} 대 {d['challenger']}",
                          f"{gain(d['values_before'])} → {gain(d['values'])}",
                          f"{_f(d['values_before'].get('fa_b'), 2)} → {_f(d['values'].get('fa_b'), 2)}",
                          f"{word(d.get('swap_before'))} → {word(d.get('swap'))}"] for d in moved]), ""]
    else:
        out += ["섀도 파일에 이상 확인 구간이 걸린 A/B 판정은 없다. 판정 수치는 반영 전후가 같다.", ""]

    def diff(v: dict[str, Any]) -> str:
        a, b = v.get("auroc_prev"), v.get("auroc_new")
        return "-" if a is None or b is None else f"{b - a:+.4f}"

    def end(revert: Any) -> str:
        return "-" if revert is None else ("되돌림" if revert else "확정")

    checked = [p for p in probations if p.get("values_before") and p.get("values")]
    if checked:
        out += ["확인에 쓴 파일에 이상 확인 구간이 있던 잠정 교체 확인이다. 확정 · 되돌림도 반영 후 수치로 정했다. 이 "
                "기간의 판정은 새로 올린 모델의 경보에서 나오므로 \"확정\" 쪽으로 기운다.", "",
                *_table(["시점", "확인한 모델", "AUROC 차 새-이전 (전 → 후)", "결론 (전 → 후)"],
                        [[p["at"], f"{p['version']} (이전 {p['previous']})",
                          f"{diff(p['values_before'])} → {diff(p['values'])}",
                          f"{end(p.get('revert_before'))} → {end(p.get('revert'))}"] for p in checked]), ""]
    return out


def report_markdown(rep: dict[str, Any]) -> str:
    cfg = rep["config"]
    champion, challenger = rep["champion"] or "-", rep["challenger"] or "-"
    if rep.get("provisional"):
        champion += " (잠정)"
    out = ["# D-11 모델 루프 보고서", "",
           f"등록부 `{rep['registry']}` · 현재 모델 {champion} · 도전자 {challenger}", "",
           "## 설정", "",
           *_table(["이름", "값"], [[k, cfg[k]] for k in cfg if k != "recipe"]), "",
           "학습 레시피: `" + " ".join(cfg["recipe"]) + "`", "",
           "## 점검", "",
           *_table(["점검", "한 일"], [[t["at"], ", ".join(t["actions"])] for t in rep["ticks"]]), "",
           "## 버전", "",
           *_table(["버전", "학습 시점", "상태", "학습 파일", *CLASSES],
                   [[v, i["cutoff"], i["status"], i["n_train_files"], *[i["by_class"][c] for c in CLASSES]]
                    for v, i in rep["versions"].items()]), "",
           "## 판정", ""]
    if not rep["decisions"]:
        out += ["판정이 아직 없다.", ""]
    for d in rep["decisions"]:
        out += _decision_markdown(d, cfg)
    if rep.get("probations") or rep.get("provisional"):
        out += ["## 잠정 교체 확인", "",
                "판정으로 교체된 모델은 잠정이다. 교체 뒤에 끝난 파일로 이전 모델과 다시 견주어 확정하거나 되돌린다.", ""]
        rows = []
        for q in rep.get("probations") or []:
            val = q.get("values") or {}
            rows.append([q["at"], q["version"], q["previous"], len(q["files"]), _f(val.get("auroc_new"), 4),
                         _f(val.get("auroc_prev"), 4), _f(val.get("noise"), 4), "되돌림" if q["revert"] else "확정",
                         q["by"], q["reason"]])
        if rows:
            out += [*_table(["시점", "새 모델", "이전 모델", "파일", "AUROC 새", "AUROC 이전", "우연 폭", "결과", "누가",
                             "사유"], rows), ""]
        if rep.get("provisional"):
            pv = rep["provisional"]
            out += [f"{pv['version']}은 {pv['since']}에 {pv['previous']} 대신 들어왔고 아직 확인 전이다.", ""]
    if len(rep.get("history") or []) > 1:
        ways = {"first": "첫 모델", "decision": "판정", "probation": "잠정 교체 되돌림(자동)", "rollback": "사람이 되돌림"}
        out += ["## 현재 모델 이력", "",
                *_table(["시점", "현재 모델", "경위", "누가", "사유"],
                        [[h["at"], h["champion"], ways.get(h["by"], h["by"]), h.get("who", "-"), h.get("reason", "-")]
                         for h in rep["history"]]), ""]
    out += _feedback_markdown(rep.get("feedback") or {})
    out += ["## 테스트", ""]
    if rep["note"]:
        return "\n".join(out + [rep["note"], ""])
    rows = [[v, i["status"], _f(i["test"]["auroc"]), _f(i["test"]["auroc_gun_mean"]), _f(i["test"]["ew_recall"]),
             _f(i["test"]["ew_chance"]), _f(i["test"]["ew_false_alarms_per_gun_day"], 2)]
            for v, i in rep["versions"].items()]
    out += [f"테스트 {rep['test_files']}건은 판정에 쓰지 않았다. 모든 버전을 같은 파일로 채점했다.", "",
            *_table(["버전", "상태", "AUROC", "건별 AUROC 평균", "조기 경보 적중", "우연", "오알람 / 설비 · 일"], rows), ""]
    fv = rep["first_vs_final"]
    if fv:
        out += [f"첫 모델 {fv['first']} → 지금 모델 {fv['final']}: 건별 AUROC 평균 {_f(fv['first_auroc_gun_mean'])} → "
                f"{_f(fv['final_auroc_gun_mean'])} (차 {_f(fv['diff'])})", ""]
    return "\n".join(out)


def write_report(out_dir: str, report: dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_markdown(report))


# ------------------------------------------------------------- command line
def _config(args: argparse.Namespace) -> LoopConfig:
    recipe = DEFAULT_RECIPE
    if args.recipe_file:
        with open(args.recipe_file, encoding="utf-8-sig") as f:
            recipe = tuple(str(a) for a in json.load(f))
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds needs at least two seeds, all different: the spread between the models trained "
                         "with them is the noise a challenger must beat")
    names = [g.strip() for g in args.gates.split(",") if g.strip()]
    unknown = [g for g in names if g not in CHECKS]
    if unknown or not names:
        raise ValueError(f"--gates takes names out of {', '.join(CHECKS)} (got {', '.join(unknown) or 'nothing'})")
    return LoopConfig(initial_min_files=args.initial_min_files, train_min_new=args.train_min_new,
                      shadow_min_files=args.shadow_min_files, seeds=seeds, fa_ratio_max=args.fa_ratio_max,
                      fa_cap=args.fa_cap, auroc_margin=args.auroc_margin,
                      gates=tuple(k for k in CHECKS if k in names), recipe=recipe)


# a registry written before 2026-11-12 has neither setting: it was judged by the first rule
FIRST_RULE = {"auroc_margin": 1.0, "gates": list(CHECKS)}


def _stored_config(data: dict[str, Any]) -> LoopConfig:
    c = {**FIRST_RULE, **data["config"]}
    return LoopConfig(**{**c, "seeds": tuple(c["seeds"]), "gates": tuple(c["gates"]), "recipe": tuple(c["recipe"])})


def _existing(reg: mr.Registry) -> dict[str, Any]:
    data = reg.load()
    if data is None:
        raise mr.RegistryError(f"no registry at {reg.path}")
    return data


def _sync_server(reg: mr.Registry, notify_fn: NotifyFn | None) -> None:
    """With --server-url every run ends by asking the server to reload. The call does nothing when the server already
    serves the champion; a server that missed the notice of a swap (down, unreachable) catches up here."""
    if notify_fn is not None and (reg.load() or {}).get("champion"):
        _notify(notify_fn)


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")


def _rollback(reg: mr.Registry, to: str | None, notify_fn: NotifyFn | None, who: str | None = None,
              reason: str | None = None) -> None:
    """A person puts an earlier champion back. During a provisional swap this is the operator's "take it back now":
    the model that is taken out is rejected and the probation is recorded with who and why."""
    _existing(reg)  # before the lock: a missing registry is not created
    with reg.lock():
        data = _existing(reg)
        target = to or mr.previous_champion(data)
        if target is None:
            raise ValueError("there is no earlier champion to go back to: name a version with --to")
        if target not in data["versions"]:
            raise ValueError(f"unknown version {target} (the registry has {', '.join(data['versions'])})")
        if target == data["champion"]:
            raise ValueError(f"{target} is the champion already")
        if target == data["challenger"]:
            raise ValueError(f"{target} is the challenger: it has not been judged yet and cannot be made the "
                             "champion by hand")
        old, at, prov = data["champion"], _iso(_utc_now()), data["provisional"]
        mr.set_champion(data, target, at=at, by="rollback", who=who, reason=reason)
        if prov is not None and prov["version"] == old:
            mr.reject(data, old)
            mr.record_probation(data, {"at": at, "version": old, "previous": prov["previous"], "files": [],
                                       "by": who or "operator", "values": None, "revert": True,
                                       "reason": reason or "사람이 되돌림"})
            mr.clear_provisional(data)
        reg.save(data)
    print(f"champion {old} -> {target} (rollback)", flush=True)
    _notify(notify_fn)


def _feedback_line(reg: mr.Registry, data_dir: str) -> str | None:
    """The verdict store in one line (D-12), None without a store. The counts need manifest.csv: a case is matched
    to a preprocessed file by its gun and the time of its warning."""
    data = mf.Store(reg.root).load()
    if data is None:
        return None
    text = f"{len(data['cases'])} cases, pulled {data.get('pulled_at') or '-'}"
    try:
        ranges = mf.load_ranges(data_dir)
    except (ValueError, KeyError):
        return text + " (no manifest.csv: not matched to files)"
    counts = mf.collect(data, ranges, list(ranges))["counts"]
    return text + ": " + ", ".join(f"{k} {counts[k]}" for k in mf.VERDICTS)


def _status(reg: mr.Registry, data_dir: str) -> None:
    """The registry in a few lines. A verdict store that cannot be read does not hide them: its error follows."""
    data = reg.load()
    broken = None
    try:
        line = _feedback_line(reg, data_dir)
    except mf.FeedbackError as e:
        line, broken = None, e
    if data is None:
        print(f"no registry at {reg.path}", flush=True)
        if line:
            print(f"{'feedback':<12}{line}")
        if broken is not None:
            raise broken
        return
    s = mr.summary(data, 1)
    for label, v in (("champion", s["champion"]), ("challenger", s["challenger"])):
        text = "-" if v is None else f"{v['version']}  trained at {v['cutoff']} on {v['train_files']} files"
        print(f"{label:<12}{text}")
    if s["provisional"]:
        pv = s["provisional"]
        print(f"{'provisional':<12}{pv['version']} since {pv['since']} (before it: {pv['previous']}); the files that "
              "end after the swap confirm it or take it back")
    last_tick = s["last_tick"] or "-"
    print(f"{'last tick':<12}{last_tick}")
    for d in s["decisions"]:
        verdict = "swap" if d["swap"] else "keep"
        print(f"{'last judged':<12}{d['at']}  {d['champion']} vs {d['challenger']}: {verdict} - {d['reason']}")
    if line:
        print(f"{'feedback':<12}{line}")
    if broken is not None:
        sys.stdout.flush()
        raise broken


def _feedback_pull(reg: mr.Registry, api_url: str) -> None:
    """Fetch the closed early-warning cases from the api and add them to <registry>/feedback.json (D-12 spec 6.1).
    The next training and the next judgement use them; a pull alone trains nothing."""
    with reg.lock():
        store = mf.Store(reg.root)
        store.load()  # an unreadable store stops here, before the api is asked
        n = store.merge(mf.fetch(api_url), api_url, _iso(_utc_now()) + "Z")
        total = len(store.load()["cases"])
    print(f"feedback: {n['new']} new, {n['changed']} changed, {n['same']} same, {n['skipped']} skipped "
          f"({total} cases) <- {api_url}", flush=True)


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", default=os.path.join(PROJECT_ROOT, "models", "registry"))
    common.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "preprocessed"),
                        help="preprocessed files + manifest.csv (their `end` is the calendar of the loop)")
    server = argparse.ArgumentParser(add_help=False)
    server.add_argument("--server-url", default=None,
                        help="tell this serving API (POST /model/reload) when the champion changes")
    d = LoopConfig()
    settings = argparse.ArgumentParser(add_help=False)
    settings.add_argument("--initial-min-files", type=int, default=d.initial_min_files)
    settings.add_argument("--train-min-new", type=int, default=d.train_min_new)
    settings.add_argument("--shadow-min-files", type=int, default=d.shadow_min_files)
    settings.add_argument("--seeds", default=",".join(map(str, d.seeds)),
                          help="first: the version itself, the others: its noise models")
    settings.add_argument("--fa-ratio-max", type=float, default=d.fa_ratio_max)
    settings.add_argument("--fa-cap", type=float, default=d.fa_cap)
    settings.add_argument("--auroc-margin", type=float, default=d.auroc_margin,
                          help="check 1: the gain must exceed this many noise widths (0 = any gain, 1 = above the noise)")
    settings.add_argument("--gates", default=",".join(d.gates),
                          help=f"the checks that decide, out of {','.join(CHECKS)}; the others are only recorded")
    settings.add_argument("--recipe-file", default=None, help="JSON list of train.py arguments (default: production)")
    ap = argparse.ArgumentParser(description="D-11 model loop: retrain, shadow comparison, automatic model swap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tick", parents=[common, settings, server], help="one check")
    p.add_argument("--now", default=None, help="tick time (UTC); default: the current time")
    p = sub.add_parser("replay", parents=[common, settings, server], help="ticks over the existing files, in time order")
    p.add_argument("--start", required=True, help="the first tick is one interval after this")
    p.add_argument("--end", default=None, help="last tick time; default: the first tick after the end of the last file")
    p.add_argument("--interval-days", type=int, default=7)
    p = sub.add_parser("report", parents=[common], help="report.json + report.md, every version on the test split")
    p.add_argument("--out", required=True)
    sub.add_parser("status", parents=[common], help="champion, challenger, last tick, last decision")
    p = sub.add_parser("rollback", parents=[common, server], help="make an earlier champion the champion again")
    p.add_argument("--to", default=None, help="version; default: the champion before the current one")
    p.add_argument("--by", default=None, help="who takes the model back (recorded)")
    p.add_argument("--reason", default=None, help="why (recorded), e.g. what the operators see on the line")
    p = sub.add_parser("feedback", parents=[common], help="operator verdicts (D-12): pull = fetch the closed "
                                                          "early-warning cases from the api into the registry folder")
    p.add_argument("action", choices=["pull"])
    p.add_argument("--api-url", required=True, help="the api that holds the cases (GET /api/ml/feedback)")
    args = ap.parse_args(argv)
    reg = mr.Registry(args.registry)
    try:
        if args.cmd in ("tick", "replay"):
            cfg = _config(args)
            files = load_manifest(args.data_dir)
            train_fn, eval_fn = make_train_fn(reg, args.data_dir, cfg), make_eval_fn(reg, args.data_dir, cfg)
            notify_fn = make_notify_fn(args.server_url)
            # D-12: the operator verdicts in <registry>/feedback.json, if there are any (None per call without it)
            feedback_fn = mf.make_feedback_fn(reg.root, args.data_dir, cfg.feedback_span_s)
            if args.cmd == "tick":
                now = parse_time(args.now) if args.now else _utc_now()
                # the server is told below, swap or not
                actions = tick(reg, files, now, cfg, train_fn, eval_fn, feedback_fn=feedback_fn)
                print(f"{_iso(now)}  {'  '.join(actions)}", flush=True)
            else:
                replay(reg, files, parse_time(args.start), parse_time(args.end) if args.end else None,
                       args.interval_days, cfg, train_fn, eval_fn, notify_fn, feedback_fn)
            _sync_server(reg, notify_fn)
        elif args.cmd == "report":
            write_report(args.out, build_report(reg, args.data_dir, _stored_config(_existing(reg))))
            print(f"wrote {os.path.join(args.out, 'report.md')}")
        elif args.cmd == "status":
            _status(reg, args.data_dir)
        elif args.cmd == "feedback":
            _feedback_pull(reg, args.api_url)
        else:
            _rollback(reg, args.to, make_notify_fn(args.server_url), args.by, args.reason)
    except (mr.RegistryError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2
    except RuntimeError as e:  # a training or a scoring failed: nothing was registered, the same command retries it
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
