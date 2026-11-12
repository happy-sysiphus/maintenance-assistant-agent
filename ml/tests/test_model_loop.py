"""
Unit tests of the D-11 model loop: the registry (.py/model_registry.py), the decision rule and the tick / replay
logic (.py/model_loop.py). No model is trained here - training and scoring are stand-in functions. A few seconds.
What needs data (train.py options, preprocess.py --split append, the loop end to end, the server) is in
test_pipeline.py.
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".py"))

import model_loop as ml  # noqa: E402
import model_registry as mr  # noqa: E402

CFG = {"initial_min_files": 16, "train_min_new": 5, "shadow_min_files": 5, "seeds": [42, 43, 44],
       "fa_ratio_max": 1.2, "fa_cap": 24.0, "recipe": ["--model", "ensemble"]}


def _add(data, version, status="challenger", cutoff="2021-09-06T00:00:00", n=3):
    mr.add_version(data, version, cutoff=cutoff, train_files=[f"E01_{i}" for i in range(n)], seed=42,
                   created=f"c-{version}", bundle=f"{version}/b.joblib", status=status)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ------------------------------------------------------------------ registry
def test_loop_registry_init_and_reload(tmp_path):
    reg = mr.Registry(str(tmp_path / "reg"))
    assert reg.load() is None, "no registry.json yet"
    data = reg.init(CFG)
    assert reg.path == str(tmp_path / "reg" / "registry.json") and os.path.exists(reg.path)
    assert data == mr.empty(CFG) == reg.load()
    assert data["schema"] == mr.SCHEMA and data["config"] == CFG
    assert (data["champion"], data["challenger"], data["last_tick"]) == (None, None, None)
    assert (data["versions"], data["decisions"], data["history"], data["ticks"]) == ({}, [], [], [])
    assert reg.init(CFG) == data, "the same config gives the stored registry back"


def test_loop_registry_config_mismatch(tmp_path):
    reg = mr.Registry(str(tmp_path / "reg"))
    reg.init(CFG)
    before = _read(reg.path)
    with pytest.raises(mr.RegistryError) as e:
        reg.init({**CFG, "shadow_min_files": 3})
    assert "shadow_min_files" in str(e.value)
    assert _read(reg.path) == before


def test_loop_registry_versions_and_champion(tmp_path):
    reg = mr.Registry(str(tmp_path / "reg"))
    data = mr.empty(CFG)
    assert mr.next_version(data) == "v001" and mr.previous_champion(data) is None
    _add(data, "v001")
    mr.set_champion(data, "v001", at="2021-09-06T00:00:00", by="first")
    assert mr.next_version(data) == "v002" and mr.previous_champion(data) is None
    _add(data, "v002")
    mr.set_challenger(data, "v002")
    assert data["challenger"] == "v002" and data["versions"]["v002"]["status"] == "challenger"
    mr.set_champion(data, "v002", at="2021-09-20T00:00:00", by="decision")
    assert data["champion"] == "v002" and data["challenger"] is None
    assert data["versions"]["v001"]["status"] == "retired" and data["versions"]["v002"]["status"] == "champion"
    assert data["history"] == [{"at": "2021-09-06T00:00:00", "champion": "v001", "by": "first"},
                               {"at": "2021-09-20T00:00:00", "champion": "v002", "by": "decision"}]
    assert mr.previous_champion(data) == "v001"
    assert data["versions"]["v001"] == {"cutoff": "2021-09-06T00:00:00", "train_files": ["E01_0", "E01_1", "E01_2"],
                                        "seed": 42, "created": "c-v001", "bundle": "v001/b.joblib",
                                        "status": "retired"}
    assert reg.version_dir("v001") == str(tmp_path / "reg" / "v001")
    assert reg.bundle_path(data, "v001") == str(tmp_path / "reg" / "v001" / "b.joblib")
    with pytest.raises(ValueError):
        _add(data, "v002")  # a version is registered once
    with pytest.raises(ValueError):
        mr.set_champion(data, "v009", at="x", by="rollback")


def test_loop_registry_reject_and_latest():
    data = mr.empty(CFG)
    assert mr.latest_version(data) is None
    _add(data, "v001")
    mr.set_champion(data, "v001", at="t", by="first")
    _add(data, "v002")
    mr.set_challenger(data, "v002")
    mr.reject(data, "v002")
    assert data["versions"]["v002"]["status"] == "rejected" and data["challenger"] is None
    assert data["champion"] == "v001"
    assert mr.latest_version(data) == "v002", "a rejected version still counts"
    assert mr.next_version(data) == "v003"


def test_loop_registry_atomic_save(tmp_path, monkeypatch):
    reg = mr.Registry(str(tmp_path / "reg"))
    data = reg.init(CFG)
    before = _read(reg.path)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mr.json, "dump", boom)
    _add(data, "v001")
    with pytest.raises(RuntimeError):
        reg.save(data)
    assert _read(reg.path) == before, "a failed write leaves the previous registry"
    assert os.listdir(reg.root) == ["registry.json"], "no temp file left behind"


def test_loop_registry_rejects_broken_json(tmp_path):
    reg = mr.Registry(str(tmp_path / "reg"))
    os.makedirs(reg.root)
    with open(reg.path, "w", encoding="utf-8") as fh:
        fh.write('{"schema": 1,')
    for call in (reg.load, lambda: reg.init(CFG)):
        with pytest.raises(mr.RegistryError) as e:
            call()
        assert reg.path in str(e.value)
    assert _read(reg.path) == '{"schema": 1,', "a broken registry is never overwritten"


def test_loop_registry_lock(tmp_path):
    reg = mr.Registry(str(tmp_path / "reg"))
    lock = tmp_path / "reg" / "registry.lock"
    with reg.lock():
        assert lock.exists()
        with pytest.raises(mr.RegistryError) as e:
            with reg.lock():
                pass
        assert str(lock) in str(e.value)
        assert lock.exists(), "the refused run must not remove the holder's lock"
    assert not lock.exists()


def test_loop_lock_released_on_interrupt(tmp_path):
    reg = mr.Registry(str(tmp_path / "reg"))
    with pytest.raises(KeyboardInterrupt):
        with reg.lock():
            raise KeyboardInterrupt
    assert not (tmp_path / "reg" / "registry.lock").exists()
    with reg.lock():  # and the next run gets in
        pass


def test_loop_registry_summary_limit():
    data = mr.empty(CFG)
    assert mr.summary(data) == {"champion": None, "challenger": None, "last_tick": None, "decisions": []}
    _add(data, "v001", n=16)
    mr.set_champion(data, "v001", at="t", by="first")
    _add(data, "v002", cutoff="2021-09-13T00:00:00", n=33)
    mr.set_challenger(data, "v002")
    for i in range(12):
        mr.record_decision(data, {"at": f"d{i:02d}", "swap": False})
    mr.record_tick(data, "2021-09-20T00:00:00", ["defer:1"])
    s = mr.summary(data, 10)
    assert s["champion"] == {"version": "v001", "cutoff": "2021-09-06T00:00:00", "train_files": 16}
    assert s["challenger"] == {"version": "v002", "cutoff": "2021-09-13T00:00:00", "train_files": 33}
    assert s["last_tick"] == "2021-09-20T00:00:00"
    assert [d["at"] for d in s["decisions"]] == [f"d{i:02d}" for i in range(2, 12)], "the latest 10, newest last"
    assert len(mr.summary(data, 3)["decisions"]) == 3
    assert data["ticks"] == [{"at": "2021-09-20T00:00:00", "actions": ["defer:1"]}]


# ------------------------------------------------------------------ decision rule (spec 5)
def _pf(*aurocs):
    return {f"f{i}": v for i, v in enumerate(aurocs)}


def _m(auroc, per_file=None, hits=1, fa=10.0, ew=True):
    """What metrics_summary gives for one model on the shadow files."""
    return {"auroc": auroc, "per_file": per_file if per_file is not None else _pf(auroc),
            "ew": ew, "hits": hits if ew else 0, "fa": fa if ew else 0.0}


A5, B5 = _pf(0.6, 0.7, 0.7, 0.8, 0.7), _pf(0.7, 0.8, 0.7, 0.9, 0.6)  # B >= A on 4 of 5 (the tie counts for B)
CHECKS = {"auroc_margin", "file_majority", "ew_hits", "fa_ratio", "fa_cap"}
SEEDS_75 = [0.75, 0.74, 0.76]  # B first, then the seed models: noise 0.02


def test_loop_judge_swap():
    d = ml.judge(_m(0.70, A5, hits=2, fa=10.0), _m(0.75, B5, hits=2, fa=11.0), SEEDS_75, ml.LoopConfig())
    assert d["swap"] is True and d["reason"] == "다섯 조건을 모두 만족"
    assert set(d["checks"]) == CHECKS and all(v is True for v in d["checks"].values())
    v = d["values"]
    assert set(v) == {"auroc_a", "auroc_b", "noise", "files_b_ge_a", "files_compared", "hits_a", "hits_b", "fa_a",
                      "fa_b"}
    assert (v["auroc_a"], v["auroc_b"], v["files_b_ge_a"], v["files_compared"]) == (0.70, 0.75, 4, 5)
    assert (v["hits_a"], v["hits_b"], v["fa_a"], v["fa_b"]) == (2, 2, 10.0, 11.0)
    assert v["noise"] == pytest.approx(0.02)


def test_loop_judge_margin_fails():
    d = ml.judge(_m(0.70, A5, hits=2), _m(0.71, B5, hits=2), [0.71, 0.70, 0.72], ml.LoopConfig())
    assert d["checks"]["auroc_margin"] is False and d["swap"] is False
    assert all(d["checks"][k] for k in CHECKS - {"auroc_margin"})
    assert "auroc_margin" in d["reason"]


def test_loop_judge_margin_equal_fails():
    # exact binary fractions: B - A == noise == 0.25 -> not above the noise
    d = ml.judge(_m(0.5, _pf(0.5)), _m(0.75, _pf(0.75)), [0.75, 0.5, 0.75], ml.LoopConfig())
    assert d["values"]["noise"] == 0.25 and d["checks"]["auroc_margin"] is False and d["swap"] is False


def test_loop_judge_majority():
    cfg, seeds = ml.LoopConfig(), [0.80, 0.79, 0.81]
    d = ml.judge(_m(0.70, _pf(0.7, 0.7, 0.7, 0.7, 0.7)), _m(0.80, _pf(0.8, 0.8, 0.6, 0.6, 0.6)), seeds, cfg)
    assert (d["values"]["files_b_ge_a"], d["values"]["files_compared"]) == (2, 5)
    assert d["checks"]["file_majority"] is False and d["swap"] is False and "file_majority" in d["reason"]
    d = ml.judge(_m(0.70, _pf(0.7, 0.7, 0.7, 0.7)), _m(0.80, _pf(0.8, 0.8, 0.6, 0.6)), seeds, cfg)
    assert d["checks"]["file_majority"] is False, "half is not a majority"
    d = ml.judge(_m(0.70, _pf(0.7, 0.7, 0.7)), _m(0.80, _pf(0.7, 0.8, 0.6)), seeds, cfg)
    assert d["values"]["files_b_ge_a"] == 2 and d["checks"]["file_majority"] is True, "a tie counts as B >= A"
    d = ml.judge(_m(0.70, {}), _m(0.80, {}), seeds, cfg)
    assert d["values"]["files_compared"] == 0 and d["checks"]["file_majority"] is False


def test_loop_judge_majority_skips_missing():
    a = _m(0.70, {"f0": 0.7, "f1": None, "f2": 0.7, "f3": 0.7, "f4": 0.7, "only_a": 0.9})
    b = _m(0.80, {"f0": 0.8, "f1": 0.8, "f2": None, "f3": 0.8, "f4": 0.6, "only_b": 0.1})
    d = ml.judge(a, b, [0.80, 0.79, 0.81], ml.LoopConfig())
    assert (d["values"]["files_b_ge_a"], d["values"]["files_compared"]) == (2, 3)
    assert d["checks"]["file_majority"] is True


def test_loop_judge_hits():
    d = ml.judge(_m(0.70, A5, hits=3), _m(0.75, B5, hits=2), SEEDS_75, ml.LoopConfig())
    assert d["checks"]["ew_hits"] is False and d["swap"] is False and "ew_hits" in d["reason"]
    assert all(d["checks"][k] for k in CHECKS - {"ew_hits"})


def test_loop_judge_fa_ratio():
    d = ml.judge(_m(0.70, A5, fa=10.0), _m(0.75, B5, fa=12.0), SEEDS_75, ml.LoopConfig())
    assert d["checks"]["fa_ratio"] is True and d["swap"] is True, "exactly 1.2 x passes"
    d = ml.judge(_m(0.70, A5, fa=10.0), _m(0.75, B5, fa=12.1), SEEDS_75, ml.LoopConfig())
    assert d["checks"]["fa_ratio"] is False and d["swap"] is False and "fa_ratio" in d["reason"]


def test_loop_judge_fa_cap():
    d = ml.judge(_m(0.70, A5, fa=23.0), _m(0.75, B5, fa=24.0), SEEDS_75, ml.LoopConfig())
    assert d["checks"]["fa_cap"] is True and d["swap"] is True
    d = ml.judge(_m(0.70, A5, fa=23.0), _m(0.75, B5, fa=24.1), SEEDS_75, ml.LoopConfig())
    assert d["checks"]["fa_cap"] is False and d["checks"]["fa_ratio"] is True and d["swap"] is False
    assert "fa_cap" in d["reason"]


def test_loop_judge_no_auroc():
    for a, b in ((None, 0.75), (0.70, None)):
        d = ml.judge(_m(a, A5), _m(b, B5), SEEDS_75, ml.LoopConfig())
        assert d["swap"] is False and d["reason"].startswith("판정 불가")
        assert set(d["checks"]) == CHECKS and not any(d["checks"].values())


def test_loop_judge_one_seed():
    d = ml.judge(_m(0.70, A5), _m(0.75, B5), [0.75, None, None], ml.LoopConfig())
    assert d["values"]["noise"] is None and d["swap"] is False and d["reason"].startswith("판정 불가")
    assert "noise" in d["reason"]


def test_loop_judge_a_without_ew():
    d = ml.judge(_m(0.70, A5, ew=False), _m(0.75, B5, hits=1, fa=5.0), SEEDS_75, ml.LoopConfig())
    assert (d["values"]["hits_a"], d["values"]["fa_a"]) == (0, 0.0)
    assert d["checks"]["fa_ratio"] is True and d["checks"]["ew_hits"] is True and d["swap"] is True
    assert "A에 조기 경보가 없어 오알람 비율 조건을 건너뜀" in d["reason"]


def test_loop_judge_b_without_ew():
    d = ml.judge(_m(0.70, A5, hits=1), _m(0.75, B5, ew=False), SEEDS_75, ml.LoopConfig())
    assert d["values"]["hits_b"] == 0 and d["checks"]["ew_hits"] is False and d["swap"] is False


def test_loop_judge_b_without_ew_never_swaps():
    """User decision 2026-11-12: the serving model always has an early warning. A challenger without one fails
    check 3 even when the champion had no hit, or no early warning either."""
    for a in (_m(0.70, A5, hits=0, fa=10.0), _m(0.70, A5, ew=False)):
        d = ml.judge(a, _m(0.75, B5, ew=False), SEEDS_75, ml.LoopConfig())
        assert d["checks"]["ew_hits"] is False and d["swap"] is False
        assert all(d["checks"][k] for k in CHECKS - {"ew_hits"}), "only check 3 stops it"
        assert "ew_hits(B에 조기 경보 없음)" in d["reason"]


def test_loop_metrics_summary():
    nan = float("nan")
    block = {"auroc_gun_mean": 0.71, "auroc": 0.69,
             "per_file": {"E01_0": {"auroc": 0.8}, "E01_1": {"auroc": nan}, "E02_0": {"auroc": 0.6}},
             "early_warning": {"false_alarms_per_gun_day": 8.3,
                               "per_file": {"E01_0": {"hit": 1.0}, "E01_1": {"hit": 0.0}, "E02_0": {"hit": nan}}}}
    assert ml.metrics_summary(block) == {"auroc": 0.71, "per_file": {"E01_0": 0.8, "E01_1": None, "E02_0": 0.6},
                                         "ew": True, "hits": 1, "fa": 8.3}
    plain = {k: v for k, v in block.items() if k != "early_warning"}
    s = ml.metrics_summary({**plain, "auroc_gun_mean": nan})
    assert s["auroc"] is None and (s["ew"], s["hits"], s["fa"]) == (False, 0, 0.0)
    s = ml.metrics_summary({**block, "early_warning": {"false_alarms_per_gun_day": nan, "per_file": {}}})
    assert (s["ew"], s["hits"], s["fa"]) == (True, 0, 0.0)


def test_loop_default_recipe():
    r = list(ml.DEFAULT_RECIPE)
    for flag, value in (("--cv", "4"), ("--early-warning-q", "0.975"), ("--drop-features", "c19"),
                        ("--model", "ensemble"), ("--val-frac", "0.25"), ("--warmup-hours", "6")):
        assert r[r.index(flag) + 1] == value
    assert "--exclude-non-welding" in r and "--seed" not in r
    assert ml.LoopConfig().as_dict() == {"initial_min_files": 16, "train_min_new": 5, "shadow_min_files": 5,
                                         "seeds": [42, 43, 44], "fa_ratio_max": 1.2, "fa_cap": 24.0, "recipe": r}


# ------------------------------------------------------------------ tick / replay (spec 3)
T0 = pd.Timestamp("2021-08-16")  # a Monday
WEEKS = [2, 2, 11, 17, 7, 1, 4, 3, 12, 3, 1]  # train files ending per week from 2021-08-16 (spec 2)


def _files(counts, start=T0):
    """counts[i] files that end on the Tuesday noon of week i."""
    out = []
    for i, n in enumerate(counts):
        end = start + pd.Timedelta(days=7 * i + 1, hours=12)
        out += [(f"w{i:02d}_{j}", end) for j in range(n)]
    return out


def _week(k):
    return T0 + pd.Timedelta(days=7 * k)


class Fakes:
    """Stand-ins for training and scoring: they record their calls; a version scores its entry in aurocs (0.70 when
    it has none), a seed model a little off it."""

    def __init__(self, aurocs=None):
        self.aurocs = aurocs or {}
        self.trains, self.evals, self.notified = [], [], 0
        self.fail_train = self.fail_eval = self.fail_notify = False

    def train(self, version, files, seeds):
        if self.fail_train:
            raise RuntimeError("train.py failed")
        self.trains.append((version, list(files), list(seeds)))
        return {"created": f"c-{version}", "bundle": f"{version}/b.joblib"}

    def eval(self, version, seed, files):
        if self.fail_eval:
            raise RuntimeError("train.py --evaluate failed")
        self.evals.append((version, seed, list(files)))
        a = self.aurocs.get(version, 0.70) + (0.0 if seed is None else 0.001 * (seed - 42))
        return _m(a, {f: a for f in files})

    def notify(self):
        if self.fail_notify:
            raise RuntimeError("server down")
        self.notified += 1

    def fns(self):
        return self.train, self.eval, self.notify


def _reg(tmp_path, name="reg"):
    return mr.Registry(str(tmp_path / name))


def test_loop_spec_timeline(tmp_path):
    files = [("old_0", pd.Timestamp("2019-12-25 14:12:04"))] + _files(WEEKS)
    reg, fk = _reg(tmp_path), Fakes()
    out = ml.replay(reg, files, T0, None, 7, ml.LoopConfig(), *fk.fns())
    assert out == [("2021-08-23T00:00:00", ["wait_first:3"]), ("2021-08-30T00:00:00", ["wait_first:5"]),
                   ("2021-09-06T00:00:00", ["first:v001"]), ("2021-09-13T00:00:00", ["train:v002"]),
                   ("2021-09-20T00:00:00", ["keep:v001|v002", "train:v003"]), ("2021-09-27T00:00:00", ["defer:1"]),
                   ("2021-10-04T00:00:00", ["keep:v001|v003", "train:v004"]), ("2021-10-11T00:00:00", ["defer:3"]),
                   ("2021-10-18T00:00:00", ["keep:v001|v004", "train:v005"]), ("2021-10-25T00:00:00", ["defer:3"]),
                   ("2021-11-01T00:00:00", ["defer:4"])]
    assert [(v, len(f)) for v, f, _ in fk.trains] == [("v001", 16), ("v002", 33), ("v003", 40), ("v004", 45),
                                                      ("v005", 60)]
    data = reg.load()
    assert [len(d["shadow_files"]) for d in data["decisions"]] == [7, 5, 15]
    assert (data["champion"], data["challenger"], data["last_tick"]) == ("v001", "v005", "2021-11-01T00:00:00")
    assert [v["status"] for v in data["versions"].values()] == ["champion", "rejected", "rejected", "rejected",
                                                                "challenger"]
    assert [v["cutoff"] for v in data["versions"].values()][:2] == ["2021-09-06T00:00:00", "2021-09-13T00:00:00"]
    assert [(t["at"], t["actions"]) for t in data["ticks"]] == out
    assert fk.notified == 0 and not os.path.exists(os.path.join(reg.root, "registry.lock"))


def _swap_run(tmp_path, fk, cfg=None):
    """16 files -> v001; 5 more -> v002; 5 more -> the decision."""
    reg = _reg(tmp_path)
    out = ml.replay(reg, _files([16, 5, 5]), T0, None, 7, cfg or ml.LoopConfig(), *fk.fns())
    return reg, out


def test_loop_swap_changes_champion(tmp_path):
    fk = Fakes({"v002": 0.80})
    reg, out = _swap_run(tmp_path, fk)
    assert out == [("2021-08-23T00:00:00", ["first:v001"]), ("2021-08-30T00:00:00", ["train:v002"]),
                   ("2021-09-06T00:00:00", ["swap:v001>v002", "train:v003"])]
    data = reg.load()
    assert data["champion"] == "v002" and data["challenger"] == "v003"
    assert [data["versions"][v]["status"] for v in ("v001", "v002", "v003")] == ["retired", "champion", "challenger"]
    assert data["history"] == [{"at": "2021-08-23T00:00:00", "champion": "v001", "by": "first"},
                               {"at": "2021-09-06T00:00:00", "champion": "v002", "by": "decision"}]
    d, = data["decisions"]
    assert (d["at"], d["champion"], d["challenger"], d["swap"]) == ("2021-09-06T00:00:00", "v001", "v002", True)
    assert d["shadow_files"] == [f"w02_{j}" for j in range(5)] and set(d["checks"]) == CHECKS
    assert d["values"]["auroc_b"] == 0.80 and d["values"]["noise"] == pytest.approx(0.002)
    assert fk.notified == 1


def test_loop_keep_rejects_challenger(tmp_path):
    fk = Fakes()
    reg = _reg(tmp_path)
    cfg = ml.LoopConfig(train_min_new=6)  # 5 shadow files: enough to judge, not enough to train again
    out = ml.replay(reg, _files([16, 6, 5]), T0, None, 7, cfg, *fk.fns())
    assert out[-1] == ("2021-09-06T00:00:00", ["keep:v001|v002"])
    data = reg.load()
    assert data["versions"]["v002"]["status"] == "rejected" and data["challenger"] is None
    assert data["champion"] == "v001" and len(data["history"]) == 1 and data["decisions"][0]["swap"] is False
    assert fk.notified == 0


def test_loop_seed_models_only_for_challengers(tmp_path):
    fk = Fakes()
    _swap_run(tmp_path, fk)
    assert [(v, s) for v, _, s in fk.trains] == [("v001", [42]), ("v002", [42, 43, 44]), ("v003", [42, 43, 44])]


def test_loop_eval_calls(tmp_path):
    fk = Fakes()
    _swap_run(tmp_path, fk)
    shadow = [f"w02_{j}" for j in range(5)]
    assert fk.evals == [("v001", None, shadow), ("v002", None, shadow), ("v002", 43, shadow), ("v002", 44, shadow)]


def test_loop_train_min_new(tmp_path):
    fk, cfg = Fakes(), ml.LoopConfig()
    reg = _reg(tmp_path)
    files = _files([16, 4, 1])
    assert ml.tick(reg, files, _week(1), cfg, *fk.fns()) == ["first:v001"]
    assert ml.tick(reg, files, _week(2), cfg, *fk.fns()) == ["idle"], "4 new files: not yet"
    assert ml.tick(reg, files, _week(3), cfg, *fk.fns()) == ["train:v002"], "5 new files"
    assert [len(f) for _, f, _ in fk.trains] == [16, 21]


def test_loop_same_now_is_noop(tmp_path):
    fk, cfg = Fakes(), ml.LoopConfig()
    reg = _reg(tmp_path)
    files = _files([16, 5])
    assert ml.tick(reg, files, _week(1), cfg, *fk.fns()) == ["first:v001"]
    before = _read(reg.path)
    assert ml.tick(reg, files, _week(1), cfg, *fk.fns()) == ["idle"]
    assert _read(reg.path) == before, "the same tick again changes nothing"
    assert ml.tick(reg, files, _week(2), cfg, *fk.fns()) == ["train:v002"]
    before = _read(reg.path)
    assert ml.tick(reg, files, _week(2), cfg, *fk.fns()) == ["defer:0"]
    assert _read(reg.path) == before and len(fk.trains) == 2


def test_loop_time_backwards(tmp_path):
    fk, cfg = Fakes(), ml.LoopConfig()
    reg = _reg(tmp_path)
    files = _files([16, 5])
    ml.tick(reg, files, _week(2), cfg, *fk.fns())
    before = _read(reg.path)
    with pytest.raises(ValueError) as e:
        ml.tick(reg, files, _week(1), cfg, *fk.fns())
    assert "2021-08-30T00:00:00" in str(e.value)
    assert _read(reg.path) == before and not os.path.exists(os.path.join(reg.root, "registry.lock"))


def test_loop_train_failure_leaves_registry(tmp_path):
    fk, cfg = Fakes(), ml.LoopConfig()
    reg = _reg(tmp_path)
    files = _files([16, 5])
    ml.tick(reg, files, _week(1), cfg, *fk.fns())
    before = _read(reg.path)
    fk.fail_train = True
    with pytest.raises(RuntimeError):
        ml.tick(reg, files, _week(2), cfg, *fk.fns())
    assert _read(reg.path) == before, "a failed training registers nothing"
    assert not os.path.exists(os.path.join(reg.root, "registry.lock"))
    fk.fail_train = False
    assert ml.tick(reg, files, _week(2), cfg, *fk.fns()) == ["train:v002"], "the next run takes the same version"


def test_loop_eval_failure_keeps_challenger(tmp_path):
    fk, cfg = Fakes(), ml.LoopConfig()
    reg = _reg(tmp_path)
    files = _files([16, 5, 5])
    ml.tick(reg, files, _week(1), cfg, *fk.fns())
    ml.tick(reg, files, _week(2), cfg, *fk.fns())
    before = _read(reg.path)
    fk.fail_eval = True
    with pytest.raises(RuntimeError):
        ml.tick(reg, files, _week(3), cfg, *fk.fns())
    data = reg.load()
    assert _read(reg.path) == before and data["decisions"] == [] and data["challenger"] == "v002"
    fk.fail_eval = False
    assert ml.tick(reg, files, _week(3), cfg, *fk.fns()) == ["keep:v001|v002", "train:v003"]


def test_loop_train_failure_after_a_decision_resumes(tmp_path):
    """The decision is saved before the next challenger is trained; a failure there must not lose either."""
    fk, cfg = Fakes({"v002": 0.80}), ml.LoopConfig()
    reg = _reg(tmp_path)
    files = _files([16, 5, 5])
    ml.tick(reg, files, _week(1), cfg, *fk.fns())
    ml.tick(reg, files, _week(2), cfg, *fk.fns())

    def train_v3_fails(version, names, seeds):
        if version == "v003":
            raise RuntimeError("train.py failed")
        return fk.train(version, names, seeds)

    with pytest.raises(RuntimeError):
        ml.tick(reg, files, _week(3), cfg, train_v3_fails, fk.eval, fk.notify)
    data = reg.load()
    assert data["champion"] == "v002" and data["challenger"] is None and len(data["decisions"]) == 1
    out = ml.replay(reg, files, T0, None, 7, cfg, *fk.fns())  # the same command again
    assert out == [("2021-09-06T00:00:00", ["train:v003"])]
    data = reg.load()
    assert data["ticks"][-1] == {"at": "2021-09-06T00:00:00", "actions": ["swap:v001>v002", "train:v003"]}
    assert data["challenger"] == "v003" and len(data["decisions"]) == 1 and fk.notified == 1


def test_loop_notify_failure_is_warning(tmp_path, capsys):
    fk = Fakes({"v002": 0.80})
    fk.fail_notify = True
    reg, out = _swap_run(tmp_path, fk)
    assert out[-1][1] == ["swap:v001>v002", "train:v003"] and reg.load()["champion"] == "v002"
    assert "server down" in capsys.readouterr().out


def test_loop_replay_resumes(tmp_path):
    files = [("old_0", pd.Timestamp("2019-12-25 14:12:04"))] + _files(WEEKS)
    cfg = ml.LoopConfig()
    whole, fk1 = _reg(tmp_path, "whole"), Fakes()
    ml.replay(whole, files, T0, None, 7, cfg, *fk1.fns())
    parts, fk2 = _reg(tmp_path, "parts"), Fakes()
    first = ml.replay(parts, files, T0, pd.Timestamp("2021-09-13"), 7, cfg, *fk2.fns())
    assert [at for at, _ in first] == ["2021-08-23T00:00:00", "2021-08-30T00:00:00", "2021-09-06T00:00:00",
                                       "2021-09-13T00:00:00"]
    rest = ml.replay(parts, files, T0, None, 7, cfg, *fk2.fns())  # from the start again
    assert rest[0][0] == "2021-09-20T00:00:00" and len(rest) == 7, "the ticks already done are skipped"
    assert fk2.trains == fk1.trains and fk2.evals == fk1.evals
    assert parts.load() == whole.load()


def test_loop_now_forms():
    t = pd.Timestamp("2021-09-20")
    for s in ("2021-09-20", "2021-09-20T00:00:00", "2021-09-20T00:00:00Z", "2021-09-20T09:00:00+09:00"):
        got = ml.parse_time(s)
        assert got == t and got.tzinfo is None, s


def test_loop_manifest_skips_dropped(tmp_path):
    rows = [{"file": "E02_1", "class": "E02", "start": "2021-09-01 02:00:00", "end": "2021-09-08 02:10:00", "dropped": ""},
            {"file": "E02_0", "class": "E02", "start": None, "end": None, "dropped": "missing rate 81%"},
            {"file": "E01_0", "class": "E01", "start": "2021-08-17 21:45:37", "end": "2021-08-24 21:55:37", "dropped": ""},
            {"file": "E01_1", "class": "E01", "start": "2021-08-17 21:45:37", "end": "2021-08-24 21:55:37", "dropped": None}]
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    assert ml.load_manifest(str(tmp_path)) == [("E01_0", pd.Timestamp("2021-08-24 21:55:37")),
                                               ("E01_1", pd.Timestamp("2021-08-24 21:55:37")),
                                               ("E02_1", pd.Timestamp("2021-09-08 02:10:00"))]
    pd.DataFrame(rows + [rows[0]]).to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError) as e:
        ml.load_manifest(str(tmp_path))
    assert "E02_1" in str(e.value)
    with pytest.raises(ValueError) as e:
        ml.load_manifest(str(tmp_path / "nowhere"))
    assert "manifest.csv" in str(e.value)


# ------------------------------------------------------------------ runner pieces and CLI (no training)
def test_loop_seed_recipe():
    assert ml.seed_recipe(ml.DEFAULT_RECIPE) == list(ml.DEFAULT_RECIPE[:-8]) + ["--no-early-warning"]
    assert ml.DEFAULT_RECIPE[-8:] == ("--cv", "4", "--early-warning-rule", "model_ng_k1", "--early-warning-q", "0.975",
                                      "--early-warning-cooldown-min", "60")
    assert "--window" in ml.seed_recipe(ml.DEFAULT_RECIPE) and "60" in ml.seed_recipe(ml.DEFAULT_RECIPE)
    assert ml.seed_recipe(("--cv", "4", "--model", "iforest", "--no-early-warning", "--window", "60")) == \
        ["--model", "iforest", "--window", "60", "--no-early-warning"]
    assert ml.model_name(ml.DEFAULT_RECIPE) == "ensemble" and ml.model_name(("--window", "60")) == "ensemble"
    assert ml.model_name(("--window", "60", "--model", "iforest")) == "iforest"


def test_loop_can_reuse(tmp_path):
    d, names = tmp_path / "v001", ["E01_0", "E02_0"]
    d.mkdir()
    assert ml.can_reuse(str(d), names, "ensemble") is False
    (d / "baseline_ensemble.joblib").write_bytes(b"x")
    (d / "baseline_ensemble_metrics.json").write_text("{}", encoding="utf-8")
    assert ml.can_reuse(str(d), names, "ensemble") is False, "no file list: an unknown training"
    (d / "train_files.txt").write_text("E01_0\nE02_0\n", encoding="utf-8")
    assert ml.can_reuse(str(d), names, "ensemble") is True
    assert ml.can_reuse(str(d), ["E01_0"], "ensemble") is False, "trained on other files"
    assert ml.can_reuse(str(d), names, "iforest") is False
    (d / "baseline_ensemble_metrics.json").unlink()
    assert ml.can_reuse(str(d), names, "ensemble") is False, "the metrics JSON is written last: training did not end"


def _history_registry(tmp_path):
    """v001 first champion, v002 rejected, v003 swapped in, v004 waiting as the challenger."""
    reg = mr.Registry(str(tmp_path / "reg"))
    data = reg.init(ml.LoopConfig().as_dict())
    _add(data, "v001", status="champion")
    mr.set_champion(data, "v001", at="2021-09-06T00:00:00", by="first")
    _add(data, "v002")
    mr.set_challenger(data, "v002")
    mr.reject(data, "v002")
    _add(data, "v003")
    mr.set_challenger(data, "v003")
    mr.set_champion(data, "v003", at="2021-10-04T00:00:00", by="decision")
    _add(data, "v004")
    mr.set_challenger(data, "v004")
    mr.record_decision(data, {"at": "2021-10-04T00:00:00", "champion": "v001", "challenger": "v003", "swap": True,
                              "reason": "다섯 조건을 모두 만족"})
    mr.record_tick(data, "2021-10-18T00:00:00", ["train:v004"])
    reg.save(data)
    return reg


def test_loop_cli_status_without_registry(tmp_path, capsys):
    assert ml.main(["status", "--registry", str(tmp_path / "none")]) == 0
    assert "no registry" in capsys.readouterr().out
    assert not (tmp_path / "none").exists(), "looking must not create anything"


def test_loop_cli_status(tmp_path, capsys):
    reg = _history_registry(tmp_path)
    assert ml.main(["status", "--registry", reg.root]) == 0
    out = capsys.readouterr().out
    assert "v003" in out and "v004" in out and "2021-10-18T00:00:00" in out and "다섯 조건을 모두 만족" in out


def test_loop_cli_rollback_without_history(tmp_path, capsys):
    reg = mr.Registry(str(tmp_path / "reg"))
    data = reg.init(ml.LoopConfig().as_dict())
    _add(data, "v001", status="champion")
    mr.set_champion(data, "v001", at="t", by="first")
    reg.save(data)
    before = _read(reg.path)
    assert ml.main(["rollback", "--registry", reg.root]) == 2
    assert "--to" in capsys.readouterr().err
    assert _read(reg.path) == before and os.listdir(reg.root) == ["registry.json"]
    assert ml.main(["rollback", "--registry", str(tmp_path / "none")]) == 2


def test_loop_cli_rollback(tmp_path, capsys):
    reg = _history_registry(tmp_path)
    assert ml.main(["rollback", "--registry", reg.root]) == 0
    data = reg.load()
    assert data["champion"] == "v001" and data["challenger"] == "v004", "the challenger keeps waiting"
    assert [data["versions"][v]["status"] for v in ("v001", "v002", "v003", "v004")] == \
        ["champion", "rejected", "retired", "challenger"]
    assert data["history"][-1]["champion"] == "v001" and data["history"][-1]["by"] == "rollback"
    assert len(data["history"]) == 3 and "v003 -> v001" in capsys.readouterr().out


def test_loop_cli_rollback_to(tmp_path, capsys):
    reg = _history_registry(tmp_path)
    before = _read(reg.path)
    for target, word in (("v004", "challenger"), ("v009", "v009"), ("v003", "already")):
        assert ml.main(["rollback", "--registry", reg.root, "--to", target]) == 2, target
        assert word in capsys.readouterr().err
        assert _read(reg.path) == before
    assert ml.main(["rollback", "--registry", reg.root, "--to", "v002"]) == 0, "a rejected version can be put back"
    data = reg.load()
    assert data["champion"] == "v002" and data["versions"]["v003"]["status"] == "retired"


def _manifest(tmp_path):
    d = tmp_path / "pre"
    d.mkdir()
    pd.DataFrame([{"file": "E01_0", "end": "2021-08-24 21:55:37", "dropped": ""}]).to_csv(d / "manifest.csv", index=False)
    return str(d)


def test_loop_cli_broken_registry(tmp_path, capsys):
    reg = mr.Registry(str(tmp_path / "reg"))
    os.makedirs(reg.root)
    with open(reg.path, "w", encoding="utf-8") as fh:
        fh.write('{"schema": 1,')
    for cmd in (["tick", "--now", "2021-09-06", "--data-dir", _manifest(tmp_path)], ["status"], ["rollback"]):
        assert ml.main([cmd[0], "--registry", reg.root, *cmd[1:]]) == 2, cmd
        assert reg.path in capsys.readouterr().err
    assert _read(reg.path) == '{"schema": 1,' and os.listdir(reg.root) == ["registry.json"]


def test_loop_cli_settings_refused(tmp_path, capsys):
    data_dir = _manifest(tmp_path)
    reg = str(tmp_path / "reg")
    assert ml.main(["tick", "--now", "2021-09-06", "--registry", reg, "--data-dir", data_dir, "--seeds", "42"]) == 2
    assert "two seeds" in capsys.readouterr().err, "one seed gives no noise width: no challenger could ever win"
    assert ml.main(["tick", "--now", "2021-09-06", "--registry", reg, "--data-dir", str(tmp_path / "nowhere")]) == 2
    assert "manifest.csv" in capsys.readouterr().err
    assert ml.main(["tick", "--now", "2021-09-06", "--registry", reg, "--data-dir", data_dir]) == 0
    assert "wait_first:1" in capsys.readouterr().out
    assert ml.main(["tick", "--now", "2021-09-07", "--registry", reg, "--data-dir", data_dir,
                    "--shadow-min-files", "3"]) == 2
    assert "shadow_min_files" in capsys.readouterr().err, "a registry keeps the settings it was made with"


# ------------------------------------------------------------------ found by the final review
def test_loop_late_file_still_counts(tmp_path):
    """A file that ended before a training tick but reached the folder after it was not trained on: it is shadow
    evidence and counts as new. (In a replay every file is there from the start, so `end` alone decides.)"""
    fk, cfg = Fakes(), ml.LoopConfig()
    reg = _reg(tmp_path)
    early = _files([16, 5])
    assert ml.tick(reg, early, _week(1), cfg, *fk.fns()) == ["first:v001"]
    assert ml.tick(reg, early, _week(2), cfg, *fk.fns()) == ["train:v002"]
    late = ("late_0", _week(2) - pd.Timedelta(hours=1))  # ended an hour before v002 was trained, arrived after
    week2 = [(f"w02_{j}", _week(2) + pd.Timedelta(days=1)) for j in range(4)]
    files = sorted([*early, late, *week2], key=lambda x: (x[1], x[0]))
    assert "late_0" not in reg.load()["versions"]["v002"]["train_files"]
    assert ml.tick(reg, files, _week(3), cfg, *fk.fns()) == ["keep:v001|v002", "train:v003"]
    data = reg.load()
    assert sorted(data["decisions"][0]["shadow_files"]) == ["late_0", "w02_0", "w02_1", "w02_2", "w02_3"]
    assert "late_0" in data["versions"]["v003"]["train_files"] and len(data["versions"]["v003"]["train_files"]) == 26


def test_loop_cli_tick_tells_the_server_every_time(tmp_path, monkeypatch, capsys):
    """With --server-url every tick ends by asking the server to reload (a no-op when it already serves the
    champion): a server that missed the notice of a swap catches up at the next tick."""
    calls = []
    monkeypatch.setattr(ml, "make_notify_fn", lambda url: (lambda: calls.append(url)) if url else None)
    reg = _history_registry(tmp_path)  # champion v003, challenger v004
    data_dir = _manifest(tmp_path)  # one file, among v004's training files: nothing to judge
    args = ["tick", "--now", "2021-10-25", "--registry", reg.root, "--data-dir", data_dir]
    assert ml.main(args) == 0 and calls == [], "no --server-url, no call"
    assert ml.main([*args, "--server-url", "http://ml:8000"]) == 0
    assert "defer:0" in capsys.readouterr().out and calls == ["http://ml:8000"]
    fresh = str(tmp_path / "fresh")  # no champion yet: there is nothing the server could load
    assert ml.main(["tick", "--now", "2021-10-25", "--registry", fresh, "--data-dir", data_dir,
                    "--server-url", "http://ml:8000"]) == 0
    assert "wait_first:1" in capsys.readouterr().out and calls == ["http://ml:8000"]


def _fake_training(monkeypatch, early_warning):
    """make_train_fn without train.py: the "training" writes an empty bundle and a metrics JSON with this
    early_warning block. Returns the list of trainings that were started."""
    import json
    started = []

    def fake_train(args, what):
        args = [str(x) for x in args]
        folder = args[args.index("--model-dir") + 1]
        started.append(what)
        open(os.path.join(folder, "baseline_ensemble.joblib"), "wb").close()
        with open(os.path.join(folder, "baseline_ensemble_metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({"created": "2026-11-12T00:00:00", "early_warning": early_warning}, fh)

    monkeypatch.setattr(ml, "_train_py", fake_train)
    monkeypatch.setattr(ml, "_check_bundle", lambda path: None)
    return started


@pytest.mark.parametrize("block, why", [
    (None, "turned off"),
    ({"enabled": False, "reason": "no gun-held-out scores (no --cv, no val files)"}, "no gun-held-out scores"),
    ({"enabled": False, "threshold": None, "calibration": {"n_windows": 12}}, "12 calibration windows"),
])
def test_loop_version_without_early_warning_is_not_registered(tmp_path, monkeypatch, block, why):
    """User decision 2026-11-12: a model without an early warning never enters the registry - not as the first model
    either, which no decision would stop. Its training is kept, so the next tick does not train it again."""
    started = _fake_training(monkeypatch, block)
    reg, cfg, files = _reg(tmp_path), ml.LoopConfig(initial_min_files=2), _files([2])
    train_fn = ml.make_train_fn(reg, str(tmp_path / "data"), cfg)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="v001 .*no early warning") as e:
            ml.tick(reg, files, _week(1), cfg, train_fn, Fakes().eval)
        assert why in str(e.value)
        data = reg.load()
        assert data["champion"] is None and data["versions"] == {} and data["ticks"] == []
    assert len(started) == 1, "the same files are not trained again"


def test_loop_challenger_without_early_warning_gets_no_seed_models(tmp_path, monkeypatch):
    """The check comes before the noise models: nothing is trained for a version that cannot be registered."""
    started = _fake_training(monkeypatch, None)
    train_fn = ml.make_train_fn(_reg(tmp_path), str(tmp_path / "data"), ml.LoopConfig())
    with pytest.raises(RuntimeError, match="v002 .*no early warning"):
        train_fn("v002", ["a", "b"], [42, 43, 44])
    assert len(started) == 1 and "seed 42" in started[0]


def test_loop_version_with_early_warning_is_registered(tmp_path, monkeypatch):
    started = _fake_training(monkeypatch, {"enabled": True, "threshold": 0.97})
    reg, cfg = _reg(tmp_path), ml.LoopConfig(initial_min_files=2)
    train_fn = ml.make_train_fn(reg, str(tmp_path / "data"), cfg)
    assert ml.tick(reg, _files([2]), _week(1), cfg, train_fn, Fakes().eval) == ["first:v001"]
    assert reg.load()["champion"] == "v001" and len(started) == 1
