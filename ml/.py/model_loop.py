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


@dataclass(frozen=True)
class LoopConfig:
    """Spec 3. The registry stores these values and refuses a run with other ones."""
    initial_min_files: int = 16  # ended files before the first model is trained
    train_min_new: int = 5  # new ended files before a challenger is trained
    shadow_min_files: int = 5  # shadow files before a challenger is judged
    seeds: tuple[int, ...] = (42, 43, 44)  # seeds[0]: the version itself, the others: the noise models
    fa_ratio_max: float = 1.2  # B's early-warning false alarms <= this x A's
    fa_cap: float = 24.0  # ... and <= this per gun-day (B-20)
    recipe: tuple[str, ...] = DEFAULT_RECIPE

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {**d, "seeds": list(self.seeds), "recipe": list(self.recipe)}


# ------------------------------------------------------------- decision rule (spec 5)
CHECKS = ("auroc_margin", "file_majority", "ew_hits", "fa_ratio", "fa_cap")
EPS = 1e-9  # "at most 1.2 x" must not fail on floating-point noise


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


def judge(a: dict[str, Any], b: dict[str, Any], seed_aurocs: list[float | None], cfg: LoopConfig) -> dict[str, Any]:
    """Champion a against challenger b on the same shadow files (metrics_summary dicts). seed_aurocs: the per-gun
    AUROC mean of b and of the models trained on b's data with other seeds; their spread is the noise the gain must
    exceed. Swap only when all five checks hold. A challenger without an early warning never passes check 3: the
    serving model always has one (user decision 2026-11-12)."""
    seeds = [s for s in seed_aurocs if s is not None]
    noise = max(seeds) - min(seeds) if len(seeds) >= 2 else None
    both = [(a["per_file"][f], b["per_file"][f]) for f in a["per_file"]
            if a["per_file"][f] is not None and b["per_file"].get(f) is not None]
    wins = sum(y >= x for x, y in both)
    values = {"auroc_a": a["auroc"], "auroc_b": b["auroc"], "noise": noise, "files_b_ge_a": wins,
              "files_compared": len(both), "hits_a": a["hits"], "hits_b": b["hits"], "fa_a": a["fa"], "fa_b": b["fa"]}
    missing = [k for k in ("auroc_a", "auroc_b", "noise") if values[k] is None]
    if missing:
        return {"values": values, "checks": dict.fromkeys(CHECKS, False), "swap": False,
                "reason": f"판정 불가: {', '.join(missing)} 없음"}
    gain = b["auroc"] - a["auroc"]
    checks = {"auroc_margin": gain > noise,
              "file_majority": wins * 2 > len(both),
              "ew_hits": b["ew"] and b["hits"] >= a["hits"],
              "fa_ratio": (not a["ew"]) or b["fa"] <= a["fa"] * cfg.fa_ratio_max + EPS,
              "fa_cap": b["fa"] <= cfg.fa_cap + EPS}
    detail = {"auroc_margin": f"B-A {gain:.4f} <= 폭 {noise:.4f}",
              "file_majority": f"B>=A {wins}/{len(both)}",
              "ew_hits": f"B {b['hits']} < A {a['hits']}" if b["ew"] else "B에 조기 경보 없음",
              "fa_ratio": f"B {b['fa']:.2f} > A {a['fa']:.2f} x {cfg.fa_ratio_max}",
              "fa_cap": f"B {b['fa']:.2f} > {cfg.fa_cap}"}
    swap = all(checks.values())
    reason = "다섯 조건을 모두 만족" if swap else ", ".join(f"{k}({detail[k]})" for k in CHECKS if not checks[k])
    if not a["ew"]:
        reason += "; A에 조기 경보가 없어 오알람 비율 조건을 건너뜀"
    return {"values": values, "checks": checks, "swap": swap, "reason": reason}


# ------------------------------------------------------------- tick / replay (spec 3)
# (version, train file names, seeds) -> {"created", "bundle"}: seeds[0] trains the version, the others the noise models
TrainFn = Callable[[str, list[str], list[int]], dict[str, str]]
# (version, seed of a noise model or None for the version itself, file names) -> metrics_summary dict
EvalFn = Callable[[str, "int | None", list[str]], dict[str, Any]]
NotifyFn = Callable[[], None]  # called once after the champion changed

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


def tick(reg: mr.Registry, files: list[tuple[str, pd.Timestamp]], now: pd.Timestamp, cfg: LoopConfig,
         train_fn: TrainFn, eval_fn: EvalFn, notify_fn: NotifyFn | None = None) -> list[str]:
    """One check at `now` (spec 3). Returns what it did: "wait_first:N", "first:v001", "defer:N", "swap:A>B",
    "keep:A|B", "train:v00N" or "idle". The registry is saved only after a step is complete, so a failed training
    or scoring leaves it as it was and the next call does that step again."""
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

        if data["champion"] is None:
            if len(ended) < cfg.initial_min_files:
                return quiet(f"wait_first:{len(ended)}")
            v = mr.next_version(data)
            out = train_fn(v, names, list(cfg.seeds[:1]))
            mr.add_version(data, v, cutoff=at, train_files=names, seed=cfg.seeds[0], created=out["created"],
                           bundle=out["bundle"], status="champion")
            mr.set_champion(data, v, at=at, by="first")
            commit(f"first:{v}")
            return done

        if data["challenger"] is not None:
            a, b = data["champion"], data["challenger"]
            # files neither model was trained on. By membership, not by `end > cutoff`: a file that ended before b was
            # trained but reached the folder after it is unseen too (in a replay the two are the same)
            seen = set(data["versions"][b]["train_files"])
            shadow = [f for f in names if f not in seen]
            if len(shadow) < cfg.shadow_min_files:
                return quiet(f"defer:{len(shadow)}")
            ma, mb = eval_fn(a, None, shadow), eval_fn(b, None, shadow)
            seed_aurocs = [mb["auroc"]] + [eval_fn(b, s, shadow)["auroc"] for s in cfg.seeds[1:]]
            decision = {"at": at, "champion": a, "challenger": b, "shadow_files": shadow,
                        **judge(ma, mb, seed_aurocs, cfg)}
            mr.record_decision(data, decision)
            if decision["swap"]:
                mr.set_champion(data, b, at=at, by="decision")
                commit(f"swap:{a}>{b}")
                _notify(notify_fn)
            else:
                mr.reject(data, b)
                commit(f"keep:{a}|{b}")

        seen = set(data["versions"][mr.latest_version(data)]["train_files"])  # of the version trained last
        if sum(f not in seen for f in names) >= cfg.train_min_new:
            v = mr.next_version(data)
            out = train_fn(v, names, list(cfg.seeds))
            mr.add_version(data, v, cutoff=at, train_files=names, seed=cfg.seeds[0], created=out["created"],
                           bundle=out["bundle"], status="challenger")
            mr.set_challenger(data, v)
            commit(f"train:{v}")
        return done or quiet("idle")


def replay(reg: mr.Registry, files: list[tuple[str, pd.Timestamp]], start: pd.Timestamp, end: pd.Timestamp | None,
           interval_days: int, cfg: LoopConfig, train_fn: TrainFn, eval_fn: EvalFn,
           notify_fn: NotifyFn | None = None) -> list[tuple[str, list[str]]]:
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
            actions = tick(reg, files, now, cfg, train_fn, eval_fn, notify_fn)
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


def can_reuse(folder: str, names: list[str], model: str) -> bool:
    """A finished training of exactly these files is in the folder (train.py writes the metrics JSON last)."""
    need = [os.path.join(folder, n) for n in (f"baseline_{model}.joblib", f"baseline_{model}_metrics.json",
                                              "train_files.txt")]
    return all(os.path.exists(p) for p in need) and _read_names(need[2]) == list(names)


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

    def one(folder: str, names: list[str], recipe: list[str], seed: int) -> None:
        metrics = os.path.join(folder, f"baseline_{model}_metrics.json")
        label = os.path.relpath(folder, reg.root)
        if can_reuse(folder, names, model):
            print(f"--- {label}: already trained on these files, kept", flush=True)
            return
        os.makedirs(folder, exist_ok=True)
        for stale in (metrics, os.path.join(folder, f"baseline_{model}.joblib"),
                      os.path.join(folder, "test_metrics.json")):
            if os.path.exists(stale):
                os.remove(stale)
        shutil.rmtree(os.path.join(folder, "eval"), ignore_errors=True)  # the scores of the bundle that was here
        lst = os.path.join(folder, "train_files.txt")
        _write_names(lst, names)
        _train_py([*recipe, "--seed", seed, "--data-dir", data_dir, "--model-dir", folder, "--files-from", lst,
                   "--no-test"], f"training {label} on {len(names)} files (seed {seed})")
        try:
            _check_bundle(os.path.join(folder, f"baseline_{model}.joblib"))
        except RuntimeError:
            os.remove(metrics)  # not a finished training: the next run trains it again
            raise

    def train(version: str, names: list[str], seeds: list[int]) -> dict[str, str]:
        folder = reg.version_dir(version)
        one(folder, names, list(cfg.recipe), seeds[0])
        with open(os.path.join(folder, f"baseline_{model}_metrics.json"), encoding="utf-8") as f:
            metrics = json.load(f)
        _need_early_warning(version, metrics.get("early_warning"))  # before the noise models; the training is kept
        for s in seeds[1:]:
            one(os.path.join(folder, "seeds", f"s{s}"), names, seed_recipe(cfg.recipe), s)
        return {"created": metrics["created"], "bundle": f"{version}/baseline_{model}.joblib"}

    return train


def _evaluate_to(out: str, args: list[Any], what: str) -> dict[str, Any]:
    """The "test" block of train.py --evaluate, computed once and kept in `out`."""
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        _train_py(["--evaluate", *args, "--eval-out", tmp], what)
        os.replace(tmp, out)
    with open(out, encoding="utf-8") as f:
        return json.load(f)["test"]


def make_eval_fn(reg: mr.Registry, data_dir: str, cfg: LoopConfig) -> EvalFn:
    model = model_name(cfg.recipe)

    def evaluate(version: str, seed: int | None, names: list[str]) -> dict[str, Any]:
        folder = reg.version_dir(version)
        if seed is not None:
            folder = os.path.join(folder, "seeds", f"s{seed}")
        key = hashlib.sha1("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:12]
        out = os.path.join(folder, "eval", f"{key}.json")
        lst = os.path.join(folder, "eval", f"{key}.txt")
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            _write_names(lst, names)
        block = _evaluate_to(out, ["--model", model, "--data-dir", data_dir, "--model-dir", folder,
                                   "--eval-files-from", lst],
                             f"scoring {os.path.relpath(folder, reg.root)} on {len(names)} shadow files")
        return metrics_summary(block)

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
                             f"scoring {v} on the {n_test} test files")
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
            "first_vs_final": first_vs_final}


def _f(x: Any, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _table(head: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def _decision_markdown(d: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    v, c = d["values"], d["checks"]
    mark = {True: "통과", False: "실패"}
    verdict = "교체" if d["swap"] else "유지"
    rows = [["1 AUROC 차 > 우연 폭", f"B {_f(v['auroc_b'], 4)} - A {_f(v['auroc_a'], 4)}, 폭 {_f(v['noise'], 4)}",
             mark[c["auroc_margin"]]],
            ["2 파일 과반에서 B >= A", f"{v['files_b_ge_a']} / {v['files_compared']}", mark[c["file_majority"]]],
            ["3 B에 조기 경보가 있고 적중 B >= A", f"B {v['hits_b']}, A {v['hits_a']}", mark[c["ew_hits"]]],
            [f"4 오알람 B <= A x {cfg['fa_ratio_max']}", f"B {_f(v['fa_b'], 2)}, A {_f(v['fa_a'], 2)}",
             mark[c["fa_ratio"]]],
            [f"5 오알람 B <= {cfg['fa_cap']}", f"B {_f(v['fa_b'], 2)}", mark[c["fa_cap"]]]]
    return [f"### {d['at']} · {d['champion']} 대 {d['challenger']} → {verdict}", "",
            f"섀도 파일 {len(d['shadow_files'])}건: {', '.join(d['shadow_files'])}", "",
            *_table(["조건", "값", "결과"], rows), "", f"사유: {d['reason']}", ""]


def report_markdown(rep: dict[str, Any]) -> str:
    cfg = rep["config"]
    champion, challenger = rep["champion"] or "-", rep["challenger"] or "-"
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
    return LoopConfig(initial_min_files=args.initial_min_files, train_min_new=args.train_min_new,
                      shadow_min_files=args.shadow_min_files, seeds=seeds, fa_ratio_max=args.fa_ratio_max,
                      fa_cap=args.fa_cap, recipe=recipe)


def _stored_config(data: dict[str, Any]) -> LoopConfig:
    c = data["config"]
    return LoopConfig(**{**c, "seeds": tuple(c["seeds"]), "recipe": tuple(c["recipe"])})


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


def _rollback(reg: mr.Registry, to: str | None, notify_fn: NotifyFn | None) -> None:
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
        old = data["champion"]
        mr.set_champion(data, target, at=_iso(_utc_now()), by="rollback")
        reg.save(data)
    print(f"champion {old} -> {target} (rollback)", flush=True)
    _notify(notify_fn)


def _status(reg: mr.Registry) -> None:
    data = reg.load()
    if data is None:
        print(f"no registry at {reg.path}")
        return
    s = mr.summary(data, 1)
    for label, v in (("champion", s["champion"]), ("challenger", s["challenger"])):
        text = "-" if v is None else f"{v['version']}  trained at {v['cutoff']} on {v['train_files']} files"
        print(f"{label:<12}{text}")
    last_tick = s["last_tick"] or "-"
    print(f"{'last tick':<12}{last_tick}")
    for d in s["decisions"]:
        verdict = "swap" if d["swap"] else "keep"
        print(f"{'last judged':<12}{d['at']}  {d['champion']} vs {d['challenger']}: {verdict} - {d['reason']}")


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
    args = ap.parse_args(argv)
    reg = mr.Registry(args.registry)
    try:
        if args.cmd in ("tick", "replay"):
            cfg = _config(args)
            files = load_manifest(args.data_dir)
            train_fn, eval_fn = make_train_fn(reg, args.data_dir, cfg), make_eval_fn(reg, args.data_dir, cfg)
            notify_fn = make_notify_fn(args.server_url)
            if args.cmd == "tick":
                now = parse_time(args.now) if args.now else _utc_now()
                actions = tick(reg, files, now, cfg, train_fn, eval_fn)  # the server is told below, swap or not
                print(f"{_iso(now)}  {'  '.join(actions)}", flush=True)
            else:
                replay(reg, files, parse_time(args.start), parse_time(args.end) if args.end else None,
                       args.interval_days, cfg, train_fn, eval_fn, notify_fn)
            _sync_server(reg, notify_fn)
        elif args.cmd == "report":
            write_report(args.out, build_report(reg, args.data_dir, _stored_config(_existing(reg))))
            print(f"wrote {os.path.join(args.out, 'report.md')}")
        elif args.cmd == "status":
            _status(reg)
        else:
            _rollback(reg, args.to, make_notify_fn(args.server_url))
    except (mr.RegistryError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2
    except RuntimeError as e:  # a training or a scoring failed: nothing was registered, the same command retries it
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
