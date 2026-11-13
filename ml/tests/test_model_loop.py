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
                                        "status": "retired", "feedback": None}
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
    assert mr.summary(data) == {"champion": None, "challenger": None, "last_tick": None, "decisions": [],
                                "provisional": None, "probations": []}
    _add(data, "v001", n=16)
    mr.set_champion(data, "v001", at="t", by="first")
    _add(data, "v002", cutoff="2021-09-13T00:00:00", n=33)
    mr.set_challenger(data, "v002")
    for i in range(12):
        mr.record_decision(data, {"at": f"d{i:02d}", "swap": False})
    mr.record_tick(data, "2021-09-20T00:00:00", ["defer:1"])
    s = mr.summary(data, 10)
    assert s["champion"] == {"version": "v001", "cutoff": "2021-09-06T00:00:00", "train_files": 16, "feedback": None}
    assert s["challenger"] == {"version": "v002", "cutoff": "2021-09-13T00:00:00", "train_files": 33,
                               "feedback": None}
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
# the rule of 2026-11-11: the gain must exceed the noise and all five checks decide (the default is looser since
# 2026-11-12, see the end of this file)
STRICT = ml.LoopConfig(auroc_margin=1.0, gates=ml.CHECKS)


def test_loop_judge_swap():
    d = ml.judge(_m(0.70, A5, hits=2, fa=10.0), _m(0.75, B5, hits=2, fa=11.0), SEEDS_75, STRICT)
    assert d["swap"] is True and d["reason"] == "다섯 조건을 모두 만족"
    assert set(d["checks"]) == CHECKS and all(v is True for v in d["checks"].values())
    v = d["values"]
    assert set(v) == {"auroc_a", "auroc_b", "noise", "files_b_ge_a", "files_compared", "hits_a", "hits_b", "fa_a",
                      "fa_b"}
    assert (v["auroc_a"], v["auroc_b"], v["files_b_ge_a"], v["files_compared"]) == (0.70, 0.75, 4, 5)
    assert (v["hits_a"], v["hits_b"], v["fa_a"], v["fa_b"]) == (2, 2, 10.0, 11.0)
    assert v["noise"] == pytest.approx(0.02)


def test_loop_judge_margin_fails():
    d = ml.judge(_m(0.70, A5, hits=2), _m(0.71, B5, hits=2), [0.71, 0.70, 0.72], STRICT)
    assert d["checks"]["auroc_margin"] is False and d["swap"] is False
    assert all(d["checks"][k] for k in CHECKS - {"auroc_margin"})
    assert "auroc_margin" in d["reason"]


def test_loop_judge_margin_equal_fails():
    # exact binary fractions: B - A == noise == 0.25 -> not above the noise
    d = ml.judge(_m(0.5, _pf(0.5)), _m(0.75, _pf(0.75)), [0.75, 0.5, 0.75], STRICT)
    assert d["values"]["noise"] == 0.25 and d["checks"]["auroc_margin"] is False and d["swap"] is False


def test_loop_judge_majority():
    cfg, seeds = STRICT, [0.80, 0.79, 0.81]
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
    d = ml.judge(a, b, [0.80, 0.79, 0.81], STRICT)
    assert (d["values"]["files_b_ge_a"], d["values"]["files_compared"]) == (2, 3)
    assert d["checks"]["file_majority"] is True


def test_loop_judge_hits():
    d = ml.judge(_m(0.70, A5, hits=3), _m(0.75, B5, hits=2), SEEDS_75, STRICT)
    assert d["checks"]["ew_hits"] is False and d["swap"] is False and "ew_hits" in d["reason"]
    assert all(d["checks"][k] for k in CHECKS - {"ew_hits"})


def test_loop_judge_fa_ratio():
    d = ml.judge(_m(0.70, A5, fa=10.0), _m(0.75, B5, fa=12.0), SEEDS_75, STRICT)
    assert d["checks"]["fa_ratio"] is True and d["swap"] is True, "exactly 1.2 x passes"
    d = ml.judge(_m(0.70, A5, fa=10.0), _m(0.75, B5, fa=12.1), SEEDS_75, STRICT)
    assert d["checks"]["fa_ratio"] is False and d["swap"] is False and "fa_ratio" in d["reason"]


def test_loop_judge_fa_cap():
    d = ml.judge(_m(0.70, A5, fa=23.0), _m(0.75, B5, fa=24.0), SEEDS_75, STRICT)
    assert d["checks"]["fa_cap"] is True and d["swap"] is True
    d = ml.judge(_m(0.70, A5, fa=23.0), _m(0.75, B5, fa=24.1), SEEDS_75, STRICT)
    assert d["checks"]["fa_cap"] is False and d["checks"]["fa_ratio"] is True and d["swap"] is False
    assert "fa_cap" in d["reason"]


def test_loop_judge_no_auroc():
    for a, b in ((None, 0.75), (0.70, None)):
        d = ml.judge(_m(a, A5), _m(b, B5), SEEDS_75, STRICT)
        assert d["swap"] is False and d["reason"].startswith("판정 불가")
        assert set(d["checks"]) == CHECKS and not any(d["checks"].values())


def test_loop_judge_one_seed():
    d = ml.judge(_m(0.70, A5), _m(0.75, B5), [0.75, None, None], STRICT)
    assert d["values"]["noise"] is None and d["swap"] is False and d["reason"].startswith("판정 불가")
    assert "noise" in d["reason"]


def test_loop_judge_a_without_ew():
    d = ml.judge(_m(0.70, A5, ew=False), _m(0.75, B5, hits=1, fa=5.0), SEEDS_75, STRICT)
    assert (d["values"]["hits_a"], d["values"]["fa_a"]) == (0, 0.0)
    assert d["checks"]["fa_ratio"] is True and d["checks"]["ew_hits"] is True and d["swap"] is True
    assert "A에 조기 경보가 없어 오알람 비율 조건을 건너뜀" in d["reason"]


def test_loop_judge_b_without_ew():
    d = ml.judge(_m(0.70, A5, hits=1), _m(0.75, B5, ew=False), SEEDS_75, STRICT)
    assert d["values"]["hits_b"] == 0 and d["checks"]["ew_hits"] is False and d["swap"] is False


def test_loop_judge_b_without_ew_never_swaps():
    """User decision 2026-11-12: the serving model always has an early warning. A challenger without one fails
    check 3 even when the champion had no hit, or no early warning either."""
    for a in (_m(0.70, A5, hits=0, fa=10.0), _m(0.70, A5, ew=False)):
        d = ml.judge(a, _m(0.75, B5, ew=False), SEEDS_75, STRICT)
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
                                         "seeds": [42, 43, 44], "fa_ratio_max": 1.2, "fa_cap": 24.0,
                                         "auroc_margin": 0.0, "gates": ["auroc_margin", "ew_hits", "fa_cap"],
                                         "feedback_span_s": 3600, "recipe": r}


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

    def __init__(self, aurocs=None, before=None, fb_block=None):
        self.aurocs = aurocs or {}
        self.trains, self.evals, self.notified = [], [], 0
        self.fail_train = self.fail_eval = self.fail_notify = False
        # D-12: the spans of the operator verdicts arrive as a fourth argument, only when there is a verdict store.
        # before: version -> its AUROC without the verdicts; fb_block: what train.py reports about the spans
        self.before, self.fb_block = before or {}, fb_block
        self.train_spans, self.eval_spans = {}, []

    def train(self, version, files, seeds, *spans):
        if self.fail_train:
            raise RuntimeError("train.py failed")
        self.trains.append((version, list(files), list(seeds)))
        out = {"created": f"c-{version}", "bundle": f"{version}/b.joblib"}
        if spans:
            self.train_spans[version] = spans[0]
            out["feedback"] = self.fb_block if spans[0]["confirmed"] else None  # no span: no --feedback
        return out

    def eval(self, version, seed, files, *spans):
        if self.fail_eval:
            raise RuntimeError("train.py --evaluate failed")
        self.evals.append((version, seed, list(files)))
        off = 0.0 if seed is None else 0.001 * (seed - 42)
        a = self.aurocs.get(version, 0.70) + off
        m = _m(a, {f: a for f in files})
        if spans:
            self.eval_spans.append((version, seed, spans[0]))
            b = self.before.get(version)
            m["before"] = None if b is None else _m(b + off, {f: b + off for f in files})
            m["feedback"] = self.fb_block if spans[0]["confirmed"] else None
        return m

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


# ------------------------------------------------------------------ relaxed rule + provisional swap (2026-11-12)
GATES = ["auroc_margin", "ew_hits", "fa_cap"]


def test_loop_default_rule_swaps_on_a_small_gain():
    """User decision 2026-11-12: a small gain is enough. Checks 2 and 4 are still computed and recorded, but they
    no longer stop a swap; the record says whether the gain is above the noise."""
    a = _m(0.70, _pf(0.7, 0.7, 0.7), hits=1, fa=2.0)
    b = _m(0.71, _pf(0.8, 0.6, 0.6), hits=1, fa=5.0)  # B >= A on 1 of 3 files, 2.5 x the false alarms
    d = ml.judge(a, b, [0.71, 0.70, 0.72], ml.LoopConfig())
    assert d["swap"] is True and d["gates"] == GATES and d["above_noise"] is False
    assert d["checks"] == {"auroc_margin": True, "file_majority": False, "ew_hits": True, "fa_ratio": False,
                           "fa_cap": True}
    assert "우연 폭 안" in d["reason"] and "file_majority" in d["reason"] and "fa_ratio" in d["reason"]
    strict = ml.judge(a, b, [0.71, 0.70, 0.72], STRICT)
    assert strict["swap"] is False and strict["gates"] == list(ml.CHECKS)


def test_loop_default_rule_still_stops_these():
    cfg = ml.LoopConfig()
    for b, check in ((_m(0.70, B5), "auroc_margin"),  # no gain at all
                     (_m(0.69, B5), "auroc_margin"),
                     (_m(0.75, B5, hits=0), "ew_hits"),
                     (_m(0.75, B5, ew=False), "ew_hits"),
                     (_m(0.75, B5, fa=24.1), "fa_cap")):
        d = ml.judge(_m(0.70, A5, hits=1, fa=23.0), b, SEEDS_75, cfg)
        assert d["swap"] is False and d["checks"][check] is False and check in d["reason"], check
    d = ml.judge(_m(0.70, A5), _m(0.75, B5), SEEDS_75, cfg)
    assert d["swap"] is True and d["above_noise"] is True and "우연 폭 밖" in d["reason"]


def test_loop_config_gates():
    d = ml.LoopConfig().as_dict()
    assert d["auroc_margin"] == 0.0 and d["gates"] == GATES
    old = {k: v for k, v in d.items() if k not in ("auroc_margin", "gates")}  # a registry made before 2026-11-12
    cfg = ml._stored_config({"config": old})
    assert cfg.auroc_margin == 1.0 and cfg.gates == ml.CHECKS, "it was judged by the strict rule: read it that way"
    assert STRICT.as_dict()["gates"] == list(ml.CHECKS)


def test_loop_cli_gates(tmp_path, capsys):
    data_dir = _manifest(tmp_path)
    args = ["tick", "--now", "2021-09-06", "--data-dir", data_dir]
    assert ml.main([*args, "--registry", str(tmp_path / "bad"), "--gates", "auroc_margin,nonsense"]) == 2
    assert "nonsense" in capsys.readouterr().err
    assert ml.main([*args, "--registry", str(tmp_path / "strict"), "--auroc-margin", "1",
                    "--gates", "fa_cap,auroc_margin,file_majority,ew_hits,fa_ratio"]) == 0
    stored = mr.Registry(str(tmp_path / "strict")).load()["config"]
    assert stored["auroc_margin"] == 1.0 and stored["gates"] == list(ml.CHECKS), "kept in the order of the checks"


def test_loop_registry_provisional_fields(tmp_path):
    reg = mr.Registry(str(tmp_path / "reg"))
    data = reg.init(dict(CFG))
    assert data["provisional"] is None and data["probations"] == []
    s = mr.summary(data)
    assert s["provisional"] is None and s["probations"] == []
    old = {k: v for k, v in data.items() if k not in ("provisional", "probations")}  # written before 2026-11-12
    reg.save(old)
    again = reg.load()
    assert again["provisional"] is None and again["probations"] == []


def test_loop_swap_is_provisional(tmp_path):
    fk = Fakes({"v002": 0.80})
    reg, _ = _swap_run(tmp_path, fk)
    data = reg.load()
    assert data["provisional"] == {"version": "v002", "previous": "v001", "since": "2021-09-06T00:00:00",
                                   "shadow_files": [f"w02_{j}" for j in range(5)]}
    assert mr.summary(data)["provisional"]["version"] == "v002"


def _provisional_run(tmp_path, later):
    """v002 replaces v001 at week 3 (provisional), v003 is trained in the same tick; `later` files end in week 4."""
    fk = Fakes({"v002": 0.80})
    reg = _reg(tmp_path)
    files = _files([16, 5, 5, later])
    for k in (1, 2, 3):
        ml.tick(reg, files, _week(k), ml.LoopConfig(), *fk.fns())
    assert reg.load()["provisional"]["version"] == "v002" and fk.notified == 1
    return reg, fk, files


def test_loop_provisional_confirmed(tmp_path):
    reg, fk, files = _provisional_run(tmp_path, 5)
    n = len(fk.evals)
    assert ml.tick(reg, files, _week(4), ml.LoopConfig(), *fk.fns()) == ["confirm:v002", "keep:v002|v003", "train:v004"]
    fresh = [f"w03_{j}" for j in range(5)]
    assert fk.evals[n:n + 4] == [("v001", None, fresh), ("v002", None, fresh), ("v002", 43, fresh), ("v002", 44, fresh)], \
        "only files that ended after the decision: its own shadow files are no new evidence"
    data = reg.load()
    assert data["provisional"] is None and data["champion"] == "v002" and fk.notified == 1
    p, = data["probations"]
    assert (p["at"], p["version"], p["previous"], p["files"], p["revert"], p["by"]) == \
        ("2021-09-13T00:00:00", "v002", "v001", fresh, False, "auto")
    assert p["values"]["auroc_new"] == 0.80 and p["values"]["auroc_prev"] == 0.70
    assert p["values"]["noise"] == pytest.approx(0.002)


def test_loop_provisional_reverted(tmp_path):
    reg, fk, files = _provisional_run(tmp_path, 5)
    fk.aurocs["v002"] = 0.60  # on the files after the swap the new champion is clearly worse than the old one
    assert ml.tick(reg, files, _week(4), ml.LoopConfig(), *fk.fns()) == ["revert:v002>v001", "keep:v001|v003",
                                                                         "train:v004"]
    data = reg.load()
    assert data["champion"] == "v001" and data["provisional"] is None and fk.notified == 2, "the server is told"
    assert [data["versions"][v]["status"] for v in ("v001", "v002")] == ["champion", "rejected"]
    assert data["history"][-1] == {"at": "2021-09-13T00:00:00", "champion": "v001", "by": "probation"}
    p, = data["probations"]
    assert p["revert"] is True and p["by"] == "auto" and "폭" in p["reason"]
    d = data["decisions"][-1]
    assert (d["champion"], d["challenger"]) == ("v001", "v003"), "the challenger meets the champion that came back"


def test_loop_provisional_within_noise_is_kept(tmp_path):
    reg, fk, files = _provisional_run(tmp_path, 5)
    fk.aurocs["v002"] = 0.699  # 0.001 below the old champion, the noise is 0.002
    assert ml.tick(reg, files, _week(4), ml.LoopConfig(), *fk.fns())[0] == "confirm:v002"
    assert reg.load()["probations"][0]["revert"] is False


def test_loop_provisional_waits_for_files(tmp_path):
    reg, fk, files = _provisional_run(tmp_path, 4)
    n = len(fk.evals)
    assert ml.tick(reg, files, _week(4), ml.LoopConfig(), *fk.fns()) == ["defer:4"]
    data = reg.load()
    assert data["provisional"]["version"] == "v002" and data["probations"] == [] and len(fk.evals) == n


def test_loop_provisional_without_auroc_waits(tmp_path):
    reg, fk, files = _provisional_run(tmp_path, 5)
    plain = fk.eval  # v001 gets no AUROC, e.g. no pre-failure window in these files: nothing to compare
    fk.eval = lambda v, s, f: {**plain(v, s, f), "auroc": None} if v == "v001" else plain(v, s, f)
    out = ml.tick(reg, files, _week(4), ml.LoopConfig(), fk.train, fk.eval, fk.notify)
    assert not any(a.startswith(("confirm", "revert")) for a in out), out
    data = reg.load()
    assert data["provisional"]["version"] == "v002" and data["probations"] == [] and data["champion"] == "v002"


def test_loop_cli_rollback_of_a_provisional_champion(tmp_path, capsys):
    """The operator does not have to wait for the next judgement: who and why are recorded, the model that was
    taken out is rejected."""
    reg, fk, _ = _provisional_run(tmp_path, 0)
    assert ml.main(["rollback", "--registry", reg.root, "--by", "kim", "--reason", "알림이 너무 잦음"]) == 0
    data = reg.load()
    assert data["champion"] == "v001" and data["provisional"] is None and data["challenger"] == "v003"
    assert data["versions"]["v002"]["status"] == "rejected"
    h = data["history"][-1]
    assert (h["champion"], h["by"], h["who"], h["reason"]) == ("v001", "rollback", "kim", "알림이 너무 잦음")
    p, = data["probations"]
    assert (p["version"], p["previous"], p["revert"], p["by"], p["reason"]) == \
        ("v002", "v001", True, "kim", "알림이 너무 잦음")
    assert "v002 -> v001" in capsys.readouterr().out


def test_loop_cli_rollback_records_who_and_why(tmp_path):
    reg = _history_registry(tmp_path)  # v003 is a confirmed champion: nothing provisional
    assert ml.main(["rollback", "--registry", reg.root, "--by", "lee", "--reason", "점검"]) == 0
    data = reg.load()
    assert data["history"][-1] == {"at": data["history"][-1]["at"], "champion": "v001", "by": "rollback",
                                   "who": "lee", "reason": "점검"}
    assert data["versions"]["v003"]["status"] == "retired" and data["probations"] == []


def test_loop_cli_status_shows_provisional(tmp_path, capsys):
    reg, _, _ = _provisional_run(tmp_path, 0)
    assert ml.main(["status", "--registry", reg.root]) == 0
    out = capsys.readouterr().out
    assert "provisional" in out and "v002" in out and "v001" in out


def test_loop_report_shows_gates_probation_and_history(tmp_path):
    reg, fk, files = _provisional_run(tmp_path, 5)
    ml.tick(reg, files, _week(4), ml.LoopConfig(), *fk.fns())
    ml.main(["rollback", "--registry", reg.root, "--by", "kim", "--reason", "알림이 너무 잦음"])
    rep = ml.build_report(reg, str(tmp_path / "no-data"), ml.LoopConfig())
    assert rep["provisional"] is None and len(rep["probations"]) == 1 and len(rep["history"]) == 3
    md = ml.report_markdown(rep)
    assert "## 잠정 교체 확인" in md and "확정" in md
    assert "## 현재 모델 이력" in md and "kim" in md and "알림이 너무 잦음" in md
    assert "참고" in md and "1 AUROC 차 > 0" in md, "checks that do not decide are marked as reference"


def test_loop_reason_says_which_side_of_the_noise():
    """A challenger that is lower by more than the noise is not "within the noise"."""
    d = ml.judge(_m(0.75, A5), _m(0.70, B5), [0.70, 0.69, 0.71], ml.LoopConfig())
    assert d["swap"] is False and d["above_noise"] is False
    assert "우연 폭보다 크게 낮음" in d["reason"] and "우연 폭 안" not in d["reason"]
    row = ml._decision_markdown({"at": "t", "champion": "v001", "challenger": "v002", "shadow_files": ["f0"], **d},
                                ml.LoopConfig().as_dict())
    assert any("우연 폭보다 크게 낮음" in line for line in row) and not any("우연 폭 안" in line for line in row)
    assert ml.noise_side(0.01, 0.02) == "우연 폭 안" and ml.noise_side(-0.02, 0.02) == "우연 폭 안"
    assert ml.noise_side(0.03, 0.02) == "우연 폭 밖" and ml.noise_side(None, 0.02) is None


# ------------------------------------------------------------------ D-12: operator verdicts in the tick
# (docs/superpowers/specs/2026-11-12-feedback-labels-design.md 6; the verdict rule itself is in test_feedback.py)
SPAN = ["2021-08-20T10:00:00", "2021-08-20T11:00:00"]
FB_BLOCK = {"spans": 1, "windows_excluded": 58, "windows_already_not_normal": 2, "false_alarm_conflicts": 0}
NO_SPANS = {"confirmed": {}, "false_alarm": {}}


def _fb(*confirmed, false_alarm=()):
    """A feedback_fn: one confirmed span on each of these files, as model_feedback.collect gives it for `names`."""
    def fn(names):
        spans = {"confirmed": {f: [SPAN] for f in confirmed if f in names},
                 "false_alarm": {f: [SPAN] for f in false_alarm if f in names}}
        return {"spans": spans, "counts": {"confirmed": len(spans["confirmed"]), "false_alarm": len(spans["false_alarm"]),
                                           "unknown": 0, "unmatched": 0},
                "event_ids": [f"ev-{f}" for f in spans["confirmed"]]}
    return fn


def test_loop_config_has_feedback_span():
    assert ml.LoopConfig().feedback_span_s == 3600 and ml.LoopConfig().as_dict()["feedback_span_s"] == 3600


def test_loop_registry_old_config_still_opens(tmp_path):
    """A registry written before D-12 has no feedback_span_s: it is read as the default, and left as it is."""
    now = ml.LoopConfig().as_dict()
    old = {k: v for k, v in now.items() if k != "feedback_span_s"}
    reg = _reg(tmp_path)
    reg.init(old)
    assert reg.init(now)["config"] == old and "feedback_span_s" not in reg.load()["config"]
    with pytest.raises(mr.RegistryError) as e:
        reg.init({**now, "feedback_span_s": 1800})
    assert "feedback_span_s" in str(e.value)
    new = _reg(tmp_path, "new")
    new.init(now)
    assert new.init(old)["config"] == now, "and the other way round"
    assert mr.CONFIG_DEFAULTS == {"feedback_span_s": 3600}


def test_loop_tick_without_feedback_fn(tmp_path):
    fk = Fakes({"v002": 0.80})
    reg, _ = _swap_run(tmp_path, fk)
    assert fk.train_spans == {} and fk.eval_spans == [], "three arguments, as before D-12"
    data = reg.load()
    assert [v["feedback"] for v in data["versions"].values()] == [None, None, None]
    d, = data["decisions"]
    assert (d["feedback"], d["values_before"], d["swap_before"]) == (None, None, None)


def test_loop_tick_feedback_fn_returns_none(tmp_path):
    """No verdict store: the function gives None and the tick is what it was."""
    fk = Fakes({"v002": 0.80})
    reg = _reg(tmp_path)
    asked = []
    out = ml.replay(reg, _files([16, 5, 5]), T0, None, 7, ml.LoopConfig(), *fk.fns(),
                    feedback_fn=lambda names: asked.append(list(names)))
    assert [a for _, a in out] == [["first:v001"], ["train:v002"], ["swap:v001>v002", "train:v003"]]
    assert fk.train_spans == {} and fk.eval_spans == [] and [len(n) for n in asked] == [16, 21, 5, 26]
    data = reg.load()
    assert [v["feedback"] for v in data["versions"].values()] == [None, None, None]
    assert data["decisions"][0]["feedback"] is None and data["decisions"][0]["values_before"] is None


def test_loop_tick_passes_spans_to_training(tmp_path):
    fk = Fakes(fb_block=FB_BLOCK)
    reg = _reg(tmp_path)
    ml.replay(reg, _files([16, 5]), T0, None, 7, ml.LoopConfig(), *fk.fns(), feedback_fn=_fb("w00_0"))
    assert fk.train_spans == {"v001": {"confirmed": {"w00_0": [SPAN]}, "false_alarm": {}},
                              "v002": {"confirmed": {"w00_0": [SPAN]}, "false_alarm": {}}}, "v002 has the file too"
    assert [(v, s) for v, _, s in fk.trains] == [("v001", [42]), ("v002", [42, 43, 44])]
    data = reg.load()
    assert data["versions"]["v001"]["feedback"] == {"confirmed": 1, "false_alarm": 0, "unknown": 0, "unmatched": 0,
                                                    "event_ids": ["ev-w00_0"], **FB_BLOCK}
    assert mr.summary(data)["champion"]["feedback"]["windows_excluded"] == 58


def test_loop_tick_passes_spans_to_scoring(tmp_path):
    fk = Fakes({"v002": 0.80}, fb_block=FB_BLOCK)
    reg = _reg(tmp_path)
    ml.replay(reg, _files([16, 5, 5]), T0, None, 7, ml.LoopConfig(), *fk.fns(), feedback_fn=_fb("w02_1"))
    spans = {"confirmed": {"w02_1": [SPAN]}, "false_alarm": {}}
    assert fk.eval_spans == [("v001", None, spans), ("v002", None, spans), ("v002", 43, spans), ("v002", 44, spans)], \
        "champion, challenger and the noise models are counted on the same spans"
    data = reg.load()
    d, = data["decisions"]
    assert d["feedback"] == {"confirmed": 1, "false_alarm": 0, "unknown": 0, "unmatched": 0, "event_ids": ["ev-w02_1"],
                             **FB_BLOCK}
    assert fk.train_spans["v001"] == NO_SPANS, "a store without a verdict on these files: empty spans"
    assert data["versions"]["v001"]["feedback"] == {"confirmed": 0, "false_alarm": 0, "unknown": 0, "unmatched": 0,
                                                    "event_ids": []}
    assert fk.train_spans["v003"] == spans, "the shadow file is a training file of the next version"


def test_loop_decision_records_before_and_after(tmp_path):
    """The verdicts decide (the figures after them), and what the figures were without them is kept beside."""
    fk = Fakes({"v002": 0.80}, before={"v001": 0.70, "v002": 0.69})
    reg = _reg(tmp_path)
    ml.replay(reg, _files([16, 5, 5]), T0, None, 7, ml.LoopConfig(), *fk.fns(), feedback_fn=_fb("w02_1"))
    d, = reg.load()["decisions"]
    assert d["swap"] is True and d["swap_before"] is False
    assert d["values"]["auroc_b"] == 0.80 and d["values_before"]["auroc_b"] == 0.69
    assert d["values_before"]["auroc_a"] == 0.70 and d["values_before"]["noise"] == pytest.approx(0.002)
    assert set(d["values_before"]) == set(d["values"])


def test_loop_decision_without_before_figures(tmp_path):
    """A store, but no confirmed span on the shadow files: scoring gives no "before", nothing is invented."""
    fk = Fakes({"v002": 0.80})
    reg = _reg(tmp_path)
    ml.replay(reg, _files([16, 5, 5]), T0, None, 7, ml.LoopConfig(), *fk.fns(), feedback_fn=_fb("w00_0"))
    d, = reg.load()["decisions"]
    assert d["feedback"]["confirmed"] == 0 and (d["values_before"], d["swap_before"]) == (None, None)


def test_loop_feedback_alone_does_not_train(tmp_path):
    fk = Fakes()
    reg = _reg(tmp_path)
    files = _files([16, 2])
    out = ml.replay(reg, files, T0, None, 7, ml.LoopConfig(), *fk.fns(), feedback_fn=_fb("w00_0", "w01_0"))
    assert [a for _, a in out] == [["first:v001"], ["idle"]] and [v for v, _, _ in fk.trains] == ["v001"]


def test_loop_tick_stops_on_broken_feedback(tmp_path):
    fk = Fakes()
    reg = _reg(tmp_path)
    files = _files([16, 5])
    ml.tick(reg, files, _week(1), ml.LoopConfig(), *fk.fns())
    before = _read(reg.path)

    def broken(names):
        raise ValueError("cannot read reg/feedback.json: Expecting value")

    with pytest.raises(ValueError, match="feedback.json"):
        ml.tick(reg, files, _week(2), ml.LoopConfig(), *fk.fns(), feedback_fn=broken)
    assert _read(reg.path) == before and [v for v, _, _ in fk.trains] == ["v001"], "no training without the verdicts"
    assert not os.path.exists(os.path.join(reg.root, "registry.lock"))
    assert ml.tick(reg, files, _week(2), ml.LoopConfig(), *fk.fns()) == ["train:v002"], "the next call goes on"


def test_loop_probation_records_feedback(tmp_path):
    fk = Fakes({"v002": 0.80}, before={"v001": 0.70, "v002": 0.80}, fb_block=FB_BLOCK)
    reg = _reg(tmp_path)
    files = _files([16, 5, 5, 5])
    for k in (1, 2, 3):
        ml.tick(reg, files, _week(k), ml.LoopConfig(), *fk.fns())
    out = ml.tick(reg, files, _week(4), ml.LoopConfig(), *fk.fns(), feedback_fn=_fb("w03_0"))
    assert out[0] == "confirm:v002"
    p, = reg.load()["probations"]
    assert p["feedback"]["confirmed"] == 1 and p["feedback"]["windows_excluded"] == 58
    assert p["revert"] is False and p["revert_before"] is False
    assert p["values_before"]["auroc_new"] == 0.80 and set(p["values_before"]) == set(p["values"])
    plain = _reg(tmp_path, "plain")
    fk2 = Fakes({"v002": 0.80})
    for k in (1, 2, 3, 4):
        ml.tick(plain, files, _week(k), ml.LoopConfig(), *fk2.fns())
    q, = plain.load()["probations"]
    assert (q["feedback"], q["values_before"], q["revert_before"]) == (None, None, None)


def test_loop_registry_brief_has_feedback():
    data = mr.empty(CFG)
    mr.add_version(data, "v001", cutoff="t", train_files=["a"], seed=42, created="c", bundle="v001/b.joblib",
                   status="champion", feedback={"confirmed": 1})
    mr.set_champion(data, "v001", at="t", by="first")
    assert data["versions"]["v001"]["feedback"] == {"confirmed": 1}
    assert mr.summary(data)["champion"] == {"version": "v001", "cutoff": "t", "train_files": 1,
                                            "feedback": {"confirmed": 1}}
    del data["versions"]["v001"]["feedback"]  # a registry written before D-12
    assert mr.summary(data)["champion"]["feedback"] is None


# ------------------------------------------------------------------ D-12: train.py calls, feedback pull, status, report
import json  # noqa: E402

import model_feedback as mfb  # noqa: E402


def _case(eid, gun, ws, outcome="resolved", abnormal=1):
    """One item of GET /api/ml/feedback."""
    return {"case_id": "CASE-0001", "outcome": outcome, "closed_at": "2026-11-12T08:00:00+00:00",
            "resolution_status": "not_linked", "selected_event_id": None,
            "checks": {"abnormal": abnormal, "normal": 0, "unchecked": 0},
            "events": [{"event_id": eid, "representative": True, "gun_id": gun, "window_start": ws,
                        "detected_at": ws, "detector_created": "2026-11-01T18:46:47"}]}


CASES = [_case("e1", "w00_0", "2021-08-15T10:00:00Z"), _case("e2", "w01_0", "2021-08-20T10:00:00Z", "false_alarm"),
         _case("e3", "test_0", "2021-08-20T10:00:00Z")]


def _range_manifest(folder, files):
    """A manifest.csv for these (file, end) pairs: every file starts 7 days before it ends."""
    os.makedirs(folder, exist_ok=True)
    rows = [f"{f},{(e - pd.Timedelta(days=7)).isoformat(sep=' ')},{e.isoformat(sep=' ')}," for f, e in files]
    with open(os.path.join(folder, "manifest.csv"), "w", encoding="utf-8") as fh:
        fh.write("file,start,end,dropped\n" + "\n".join(rows) + "\n")
    return str(folder)


def test_loop_can_reuse_needs_same_feedback(tmp_path):
    """A training that was left in the folder is taken over only when it saw the same verdicts."""
    d, names = tmp_path / "v001", ["E01_0", "E02_0"]
    d.mkdir()
    (d / "baseline_ensemble.joblib").write_bytes(b"x")
    (d / "baseline_ensemble_metrics.json").write_text("{}", encoding="utf-8")
    (d / "train_files.txt").write_text("E01_0\nE02_0\n", encoding="utf-8")
    a = {"confirmed": {"E01_0": [SPAN]}, "false_alarm": {}}
    b = {"confirmed": {"E01_0": [["2021-08-21T10:00:00", "2021-08-21T11:00:00"]]}, "false_alarm": {}}
    assert ml.can_reuse(str(d), names, "ensemble") and ml.can_reuse(str(d), names, "ensemble", None)
    assert ml.can_reuse(str(d), names, "ensemble", NO_SPANS), "no span at all is the same as no verdicts"
    assert ml.can_reuse(str(d), names, "ensemble", a) is False, "trained without the verdict"
    (d / "feedback.json").write_text(json.dumps(a), encoding="utf-8")
    assert ml.can_reuse(str(d), names, "ensemble", a) is True
    assert ml.can_reuse(str(d), names, "ensemble", b) is False, "trained with another verdict"
    assert ml.can_reuse(str(d), names, "ensemble", None) is False and not ml.can_reuse(str(d), names, "ensemble", NO_SPANS)
    assert ml.can_reuse(str(d), ["E01_0"], "ensemble", a) is False, "and the files still count"


def test_loop_cli_feedback_pull(tmp_path, monkeypatch, capsys):
    asked = []
    monkeypatch.setattr(mfb, "fetch", lambda url, timeout=30.0: asked.append(url) or CASES[:2])
    root = str(tmp_path / "reg")
    assert ml.main(["feedback", "pull", "--api-url", "http://api:9000", "--registry", root]) == 0
    out = capsys.readouterr().out
    assert "2 new" in out and "(2 cases)" in out and "http://api:9000" in out and asked == ["http://api:9000"]
    data = mfb.Store(root).load()
    assert set(data["cases"]) == {"e1", "e2"} and data["source"] == "http://api:9000"
    assert ml.main(["feedback", "pull", "--api-url", "http://api:9000", "--registry", root]) == 0
    again = capsys.readouterr().out
    assert "0 new" in again and "2 same" in again and "(2 cases)" in again
    assert not os.path.exists(os.path.join(root, "registry.lock")) and not os.path.exists(os.path.join(root, "registry.json"))


def test_loop_cli_feedback_pull_api_down(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    mfb.Store(root).merge(CASES[:1], "http://api:9000", "2026-11-12T09:00:00Z")
    before = _read(os.path.join(root, "feedback.json"))

    def down(url, timeout=30.0):
        raise mfb.FeedbackError(f"no answer from {url}/api/ml/feedback: refused")

    monkeypatch.setattr(mfb, "fetch", down)
    assert ml.main(["feedback", "pull", "--api-url", "http://api:9000", "--registry", root]) == 2
    assert "http://api:9000" in capsys.readouterr().err
    assert _read(os.path.join(root, "feedback.json")) == before
    assert not os.path.exists(os.path.join(root, "registry.lock"))


def test_loop_cli_status_shows_feedback(tmp_path, capsys):
    fk = Fakes({"v002": 0.80})
    reg, _ = _swap_run(tmp_path, fk)
    data_dir = _range_manifest(tmp_path / "data", _files([16, 5, 5]))
    assert ml.main(["status", "--registry", reg.root, "--data-dir", data_dir]) == 0
    assert "feedback" not in capsys.readouterr().out, "no verdict store: no line"
    mfb.Store(reg.root).merge(CASES, "http://api:9000", "2026-11-12T09:00:00Z")
    assert ml.main(["status", "--registry", reg.root, "--data-dir", data_dir]) == 0
    line, = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("feedback")]
    assert "3 cases" in line and "2026-11-12T09:00:00Z" in line
    assert "confirmed 1" in line and "false_alarm 1" in line and "unknown 0" in line and "unmatched 1" in line
    assert ml.main(["status", "--registry", reg.root, "--data-dir", str(tmp_path / "nowhere")]) == 0
    line, = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("feedback")]
    assert "3 cases" in line and "confirmed" not in line, "no manifest: the cases cannot be matched to files"


def test_loop_report_feedback_section(tmp_path):
    fk = Fakes({"v002": 0.80}, before={"v001": 0.70, "v002": 0.69}, fb_block=FB_BLOCK)
    reg = _reg(tmp_path)
    ml.replay(reg, _files([16, 5, 5]), T0, None, 7, ml.LoopConfig(), *fk.fns(), feedback_fn=_fb("w00_0", "w02_1"))
    rep = ml.build_report(reg, str(tmp_path / "no-data"), ml.LoopConfig())
    assert rep["feedback"]["versions"]["v001"]["confirmed"] == 1 and list(rep["feedback"]["versions"]) == [
        "v001", "v002", "v003"]
    d, = rep["feedback"]["decisions"]
    assert (d["at"], d["champion"], d["challenger"], d["swap"], d["swap_before"]) == \
        ("2021-09-06T00:00:00", "v001", "v002", True, False)
    assert d["values_before"]["auroc_b"] == 0.69 and d["feedback"]["confirmed"] == 1
    md = ml.report_markdown(rep)
    assert "## 사람 판정" in md and md.index("## 사람 판정") < md.index("## 테스트")
    part = md.split("## 사람 판정")[1].split("## 테스트")[0]
    assert "| v001 | 1 | 0 | 0 | 0 | 58 |" in part, "version, confirmed, false alarm, unknown, unmatched, windows"
    assert "유지 → 교체" in part and "-0.0100 → +0.1000" in part, "what the verdicts changed"
    plain = _reg(tmp_path, "plain")
    ml.replay(plain, _files([16, 5, 5]), T0, None, 7, ml.LoopConfig(), *Fakes({"v002": 0.80}).fns())
    rep = ml.build_report(plain, str(tmp_path / "no-data"), ml.LoopConfig())
    assert rep["feedback"]["versions"] == {"v001": None, "v002": None, "v003": None}
    assert "반영된 사람 판정이 없습니다." in ml.report_markdown(rep)


def test_loop_eval_fn_passes_the_verdicts(tmp_path, monkeypatch):
    """make_eval_fn without train.py (the scoring behind success criterion 2): with a span the scoring gets
    --feedback and a kept result of its own, and what train.py writes as "test_no_feedback" comes back as "before";
    without one the call and the kept result are those from before D-12."""
    calls = []

    def fake_train(args, what):
        args = [str(x) for x in args]
        calls.append(args)
        fb = args[args.index("--feedback") + 1] if "--feedback" in args else None
        auroc = 0.75 if fb else 0.70
        out = {"test": {"auroc_gun_mean": auroc, "per_file": {"w01_0": {"auroc": auroc}}}}
        if fb:
            with open(fb, encoding="utf-8") as fh:
                spans = json.load(fh)
            out["feedback"] = dict(FB_BLOCK)
            if spans["confirmed"]:
                out["test_no_feedback"] = {"auroc_gun_mean": 0.70, "per_file": {"w01_0": {"auroc": 0.70}}}
        with open(args[args.index("--eval-out") + 1], "w", encoding="utf-8") as fh:
            json.dump(out, fh)

    monkeypatch.setattr(ml, "_train_py", fake_train)
    ev = ml.make_eval_fn(_reg(tmp_path), str(tmp_path / "data"), ml.LoopConfig())
    names = ["w01_0", "w01_1"]
    a = {"confirmed": {"w01_0": [SPAN]}, "false_alarm": {}}
    b = {"confirmed": {}, "false_alarm": {"w01_0": [SPAN]}}

    plain = ev("v001", None, names)
    assert plain["auroc"] == 0.70 and plain["before"] is None and plain["feedback"] is None
    assert "--feedback" not in calls[0]
    assert ev("v001", None, names, None) == plain and ev("v001", None, names, NO_SPANS) == plain
    assert len(calls) == 1, "no verdict: the kept result from before D-12 is used"

    got = ev("v001", None, names, a)
    assert len(calls) == 2 and "--feedback" in calls[1]
    with open(calls[1][calls[1].index("--feedback") + 1], encoding="utf-8") as fh:
        assert json.load(fh) == a, "train.py gets exactly these spans"
    assert got["auroc"] == 0.75 and got["before"]["auroc"] == 0.70 and got["feedback"] == FB_BLOCK
    assert ev("v001", None, names, a) == got and len(calls) == 2, "the same verdicts: kept"

    fa = ev("v001", None, names, b)
    assert len(calls) == 3, "other verdicts are scored again"
    assert fa["before"] is None and fa["feedback"] == FB_BLOCK, "no confirmed window: nothing to compare with"
    assert ev("v001", None, names) == plain and len(calls) == 3, "and the plain result is still there"

    seed = ev("v001", 43, names, a)
    assert os.path.join("v001", "seeds", "s43") in calls[3][calls[3].index("--model-dir") + 1]
    assert seed["before"]["auroc"] == 0.70


# ------------------------------------------------------------------ minors of the final review (2026-11-13)
def test_loop_swap_before_unknown_without_seed_figures(tmp_path):
    """The figures without the verdicts exist for A and B but not for the noise models: there is no width, so what
    the conclusion would have been cannot be said - None, not "keep"."""
    fk = Fakes({"v002": 0.80}, before={"v001": 0.70, "v002": 0.69})

    def ev(version, seed, files, *spans):
        m = fk.eval(version, seed, files, *spans)
        if seed is not None:
            m["before"] = None
        return m

    reg = _reg(tmp_path)
    ml.replay(reg, _files([16, 5, 5]), T0, None, 7, ml.LoopConfig(), fk.train, ev, fk.notify, feedback_fn=_fb("w02_1"))
    d, = reg.load()["decisions"]
    assert d["swap"] is True and d["values_before"]["auroc_b"] == 0.69 and d["values_before"]["noise"] is None
    assert d["swap_before"] is None


def test_loop_can_reuse_unreadable_feedback(tmp_path):
    """A <version>/feedback.json that cannot be read (a write cut short) is no finished training: train again."""
    d, names = tmp_path / "v001", ["E01_0"]
    d.mkdir()
    (d / "baseline_ensemble.joblib").write_bytes(b"x")
    (d / "baseline_ensemble_metrics.json").write_text("{}", encoding="utf-8")
    (d / "train_files.txt").write_text("E01_0\n", encoding="utf-8")
    (d / "feedback.json").write_text('{"confirmed": {"E01_0"', encoding="utf-8")
    assert ml.can_reuse(str(d), names, "ensemble", {"confirmed": {"E01_0": [SPAN]}, "false_alarm": {}}) is False
    assert ml.can_reuse(str(d), names, "ensemble") is False


def test_loop_cli_status_with_broken_feedback(tmp_path, capsys):
    """An unreadable verdict store: the registry's state is shown first, then the error (exit 2)."""
    reg, _ = _swap_run(tmp_path, Fakes({"v002": 0.80}))
    with open(os.path.join(reg.root, "feedback.json"), "w", encoding="utf-8") as fh:
        fh.write("{")
    assert ml.main(["status", "--registry", reg.root, "--data-dir", str(tmp_path / "nowhere")]) == 2
    got = capsys.readouterr()
    assert "champion" in got.out and "last tick" in got.out and "feedback.json" in got.err
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    with open(os.path.join(empty, "feedback.json"), "w", encoding="utf-8") as fh:
        fh.write("{")
    assert ml.main(["status", "--registry", empty]) == 2
    got = capsys.readouterr()
    assert "no registry at" in got.out and "feedback.json" in got.err


def test_loop_report_feedback_nothing_matched(tmp_path):
    """A verdict store whose cases match no file: the section says so in a sentence, not in a table of zeros."""
    def fn(names):
        return {"spans": {"confirmed": {}, "false_alarm": {}},
                "counts": {"confirmed": 0, "false_alarm": 0, "unknown": 0, "unmatched": 2}, "event_ids": []}

    reg = _reg(tmp_path)
    ml.replay(reg, _files([16, 5, 5]), T0, None, 7, ml.LoopConfig(), *Fakes({"v002": 0.80}).fns(), feedback_fn=fn)
    md = ml.report_markdown(ml.build_report(reg, str(tmp_path / "no-data"), ml.LoopConfig()))
    part = md.split("## 사람 판정")[1].split("## 테스트")[0]
    assert "반영된 사람 판정이 없습니다." in part and "| 버전 |" not in part
    assert "2건" in part, "the cases that matched no file are named"


def test_loop_report_feedback_shows_probation(tmp_path):
    """A provisional swap that was checked on files with a confirmed span: before and after, as for a decision."""
    fk = Fakes({"v002": 0.80}, before={"v001": 0.70, "v002": 0.80}, fb_block=FB_BLOCK)
    reg = _reg(tmp_path)
    files = _files([16, 5, 5, 5])
    for k in (1, 2, 3):
        ml.tick(reg, files, _week(k), ml.LoopConfig(), *fk.fns())
    ml.tick(reg, files, _week(4), ml.LoopConfig(), *fk.fns(), feedback_fn=_fb("w03_0"))
    rep = ml.build_report(reg, str(tmp_path / "no-data"), ml.LoopConfig())
    p, = rep["feedback"]["probations"]
    assert (p["version"], p["previous"], p["revert"], p["revert_before"]) == ("v002", "v001", False, False)
    part = ml.report_markdown(rep).split("## 사람 판정")[1].split("## 테스트")[0]
    assert "v002 (이전 v001)" in part and "확정 → 확정" in part and "+0.1000 → +0.1000" in part
