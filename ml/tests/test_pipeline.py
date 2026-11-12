"""
End-to-end smoke test of the pipeline on synthetic data:

    CSV (train + test) -> preprocess.py -> preprocess.py --split test -> train.py -> train.py --evaluate
        -> main.py (FastAPI) /predict on the raw test CSV rows

It checks that every stage accepts the previous stage's output and that the API reproduces the
offline preprocessing (same score for the same window). It runs in ~30 s and touches only a
temporary directory - never train/, preprocessed/ or models/.
"""
import importlib
import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".py")
sys.path.insert(0, PY)

CLASSES = ["E01", "E02", "E03", "E04"]
TERMINAL = {"E01": "E012", "E02": "E016", "E03": "E028", "E04": "E029"}
HOURS = 2  # per synthetic file
PRE_FAILURE_S = 600  # short pre-failure window so 2 h files still hold plenty of normal windows


def synth_file(path, cls, seed, start="2021-09-01T00:00:00Z"):
    """1 Hz file that ends at its failure: gap of 90 s (splits a segment), a cap-dressing block
    (c16 = 0), drifting balance pressure in the last 10 min and the class terminal code at the end."""
    rng = np.random.default_rng(seed)
    n = HOURS * 3600
    t = pd.date_range(start, periods=n, freq="s")
    ttf = np.arange(n)[::-1]
    df = pd.DataFrame({"time": t.strftime("%Y-%m-%dT%H:%M:%SZ")})
    df["c1"] = rng.normal(2, 0.5, n).round(2)
    df["c2"] = np.where(rng.random(n) < 0.3, rng.normal(2400, 50, n), 0).round(0)
    df["c3"] = rng.normal(140, 2, n).round(1)
    df["c4"] = (rng.random(n) < 0.3).astype(int)
    df["c5"] = rng.normal(480, 3, n) + np.where(ttf < PRE_FAILURE_S, 40 * (1 - ttf / PRE_FAILURE_S), 0)
    df["c6"] = rng.normal(-65, 3, n).round(0)
    df["c7"], df["c8"], df["c9"] = 222.3, 4309, 132  # gun constants
    df["c10"] = np.where(rng.random(n) < 0.9, "on", "off")
    df["c11"] = np.cumsum(rng.random(n) < 0.3)
    df["c12"] = np.cumsum(rng.random(n) < 0.5)
    df["c13"] = 14.99
    df["c14"] = 2200
    df["c15"] = 150
    df["c16"] = 1.6
    df["c17"] = 1
    df["c18"] = (rng.random(n) < 0.3).astype(int)
    df["c19"] = rng.normal(1.02e9, 5e4, n).round(0)
    df["error"] = "0"
    df.loc[(ttf < 300), "error"] = TERMINAL[cls]  # terminal code in the last 5 min
    df.loc[3000:3299, "error"] = "E003"  # a long-lived minor state in the middle
    # cap dressing block -> non_welding. Deliberately starts mid-minute (t+1530 s) so that one
    # 60 s window is half covered: that is what distinguishes the "mean" aggregation from "max".
    df.loc[1530:1799, "c16"] = 0.0
    df = df.drop(index=range(2400, 2490))  # 90 s gap -> new segment
    df = df.drop(index=[100, 101, 500])  # short gaps -> ffill
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("rsw")
    train_dir, test_dir = ws / "train", ws / "test"
    train_dir.mkdir(), test_dir.mkdir()
    seed = 0
    for cls in CLASSES:
        for gun in range(2):
            synth_file(train_dir / f"{cls}_{gun}.csv", cls, seed, start=f"2021-09-{1 + seed:02d}T00:00:00Z")
            seed += 1
    synth_file(test_dir / "test_0.csv", "E04", 99, start="2021-10-01T00:00:00Z")
    synth_file(test_dir / "test_1.csv", "E02", 98, start="2021-10-02T00:00:00Z")
    return {"ws": ws, "train": train_dir, "test": test_dir, "pre": ws / "preprocessed", "models": ws / "models"}


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test serves from its own state dir (main.py persists guns / handoffs there) and pushes nowhere."""
    monkeypatch.setenv("RSW_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("RSW_MOCK_RAG_OUT", str(tmp_path / "rag_received"))
    monkeypatch.delenv("RSW_RAG_URL", raising=False)
    monkeypatch.delenv("RSW_MODEL_REGISTRY", raising=False)  # D-11: a registry set in the shell would win over RSW_MODEL_PATH


def run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"{' '.join(map(str, args))}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}"
    return r.stdout


# ------------------------------------------------------------------ stages
def test_1_preprocess_train(workspace):
    out = run(os.path.join(PY, "preprocess.py"), "--input-dir", workspace["train"], "--out-dir", workspace["pre"],
              "--pre-failure-window", str(PRE_FAILURE_S))
    assert "files kept 8/8" in out
    pre = workspace["pre"]
    assert (pre / "scaler.json").exists() and (pre / "preprocess_config.json").exists()
    df = pd.read_parquet(pre / "E01_0.parquet")
    assert df["segment_id"].nunique() == 2, "the 90 s gap must split the file into two segments"
    assert df["non_welding"].sum() == 270
    assert df["label"].sum() == PRE_FAILURE_S + 1 and df["terminal_code"].sum() == 300
    assert "c11" not in df.columns and "c12" not in df.columns, "counters are replaced by deltas"
    assert {"welds_delta", "pos_delta", "welds_10min", "error_share_10min"} <= set(df.columns)
    normal = df[(df["label"] == 0) & (df["error_active"] == 0)]
    assert abs(normal["c5"].mean()) < 0.2, "z-score is fitted on normal rows of all files"


def test_2_preprocess_test_split(workspace):
    out = run(os.path.join(PY, "preprocess.py"), "--split", "test", "--input-dir", workspace["test"],
              "--train-out-dir", workspace["pre"])
    assert "files kept 2/2" in out
    m = pd.read_csv(workspace["pre"] / "test" / "manifest.csv")
    assert dict(zip(m["file"], m["class"])) == {"test_0": "E04", "test_1": "E02"}
    assert (m["class_source"] == "terminal code E029").iloc[0]
    cfg = json.load(open(workspace["pre"] / "test" / "preprocess_config.json", encoding="utf-8"))
    assert cfg["pre_failure_window"] == PRE_FAILURE_S, "unset options are inherited from the train run"
    assert not (workspace["pre"] / "test" / "scaler.json").exists(), "the test split never writes a scaler"
    tr, te = pd.read_parquet(workspace["pre"] / "E04_0.parquet"), pd.read_parquet(workspace["pre"] / "test" / "test_0.parquet")
    assert list(tr.columns) == list(te.columns)
    # same scaler as train: a constant sensor maps to the same z-value in both splits
    assert np.isclose(tr["c13"].iloc[0], te["c13"].iloc[0])


def test_2b_window_columns(workspace):
    """Windowing must give the 24 features a _mean/_std pair and leave metadata names alone.
    Regression: `non_welding` is aggregated with "mean" and used to become `non_welding_mean`,
    which broke --exclude-non-welding and dropped the column from the score CSVs."""
    from train import feature_columns, model_input_columns, window_features

    df = pd.read_parquet(workspace["pre"] / "E01_0.parquet")
    feat = feature_columns(df)
    w = window_features(df, 60, feat)
    assert len(feat) == 24 and len(model_input_columns(feat)) == 48
    assert set(model_input_columns(feat)) <= set(w.columns)
    for c in ("non_welding", "error_active", "terminal_code", "label", "ttf_s", "n"):
        assert c in w.columns, f"{c} must keep its own name"
        assert f"{c}_mean" not in w.columns
    nw = w["non_welding"]
    assert nw.min() == 0.0 and nw.max() == 1.0, "non_welding is a share over the window"
    assert (nw == 0.5).sum() == 1, "the half-covered window must be 0.5, i.e. a mean and not a max"
    assert (nw == 1.0).sum() == 4
    assert w["n"].max() == 60 and w["n"].min() >= 30


def test_3_train_and_evaluate(workspace):
    # 30 min warm-up so that the 2 h synthetic files exercise the per-gun normalisation + threshold
    # the IsolationForest path (--model iforest): fast, and what the serving tests below load; the ensemble (the
    # production default) goes through the same code in test_3d
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--model-dir", workspace["models"],
              "--val-frac", "0.5", "--window", "60", "--warmup-hours", "0.5", "--model", "iforest")
    assert "val: AUROC" in out and "test: AUROC" in out and "gun-normalised" in out
    metrics = json.load(open(workspace["models"] / "baseline_iforest_metrics.json", encoding="utf-8"))
    assert metrics["train_files"] and metrics["val_files"]
    assert not set(metrics["train_files"]) & set(metrics["val_files"]), "split is by file"
    assert metrics["metrics"]["test"]["files"] == ["test_0.parquet", "test_1.parquet"]
    assert metrics["metrics"]["test"]["file_class"] == {"test_0": "E04", "test_1": "E02"}
    assert len(metrics["feature_reference"]) == len(metrics["model_cols"])
    assert metrics["preprocess_config"]["gap_fill_limit"] == 60
    assert metrics["preprocess_config"]["outlier_hi"] is None, "c16 > x rule is off by default"
    assert metrics["alarm_max_non_welding"] == 0.5 and "held_windows" in metrics["metrics"]["val"]
    assert metrics["dropped_features"] == ["c19"] and len(metrics["model_cols"]) == 46, "c19 is a time counter (leak)"
    gn = metrics["gun_norm"]
    assert gn["mode"] == "center" and gn["warmup_s"] == 1800 and gn["threshold_q"] == 0.99
    assert "c7" in gn["columns"] and "hour_sin" not in gn["columns"] and "weld_duty_10min" not in gn["columns"]
    assert metrics["metrics"]["test"]["files_gun_normalised"] == 2 and metrics["metrics"]["test"]["files_gun_threshold"] == 2
    for v in metrics["metrics"]["test"]["per_file"].values():
        assert v["gun_norm"] == "gun" and v["threshold"] >= metrics["threshold"]
    assert metrics["metrics"]["val"]["auroc"] > 0.5, "the drifting c5 in the pre-failure window must be detectable"
    # terminal-code rule, evaluated as main.py fires it: one trigger per file (the code starts 300 s before failure)
    rule = metrics["metrics"]["test"]["rule_terminal_code"]
    assert rule["flag"] == "terminal_any" and rule["n_triggers"] == 2 and rule["files_hit"] == 2
    assert rule["files_with_false_trigger"] == 0 and 4 <= rule["rule_lead_min_median"] <= 5
    assert rule["n_repeats"] == 0 and rule["repeat_s"] == 86400
    assert metrics["rule"] == {"codes": ["E012", "E016", "E028", "E029"], "cooldown_s": 1800, "repeat_s": 86400,
                               "per_code": True}
    prof = metrics["metrics"]["test"]["rule_terminal_code_profile"]
    assert prof["files_hit"] == 2 and prof["out_of_profile_triggers"] == 0, "own-class codes only in the synthetic files"
    # per-gun AUROC next to the pooled one
    assert all(v["auroc"] == v["auroc"] for v in metrics["metrics"]["test"]["per_file"].values())
    assert metrics["metrics"]["test"]["auroc_gun_mean"] > 0.5

    out = run(os.path.join(PY, "train.py"), "--evaluate", "--data-dir", workspace["pre"], "--model-dir", workspace["models"],
              "--model", "iforest")
    assert "saved" in out
    tm = json.load(open(workspace["models"] / "baseline_iforest_test_metrics.json", encoding="utf-8"))
    assert tm["test"]["auroc"] == pytest.approx(metrics["metrics"]["test"]["auroc"])
    scores = pd.read_csv(workspace["models"] / "scores" / "test_0_iforest.csv", index_col=0, parse_dates=True)
    assert {"score", "alarm", "threshold", "ttf_s", "label", "error_active", "non_welding", "warmup"} <= set(scores.columns)
    assert scores["warmup"].iloc[0] == 1 and scores["warmup"].iloc[-1] == 0
    assert np.allclose(scores.loc[scores["warmup"] == 1, "threshold"], metrics["threshold"])
    assert (scores.loc[scores["warmup"] == 0, "threshold"] >= metrics["threshold"] - 1e-9).all()

    # without gun normalisation the bundle carries gun_norm = None and plain global thresholds
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--val-frac", "0.5", "--gun-norm", "none",
              "--no-test", "--model-dir", workspace["ws"] / "models_global", "--model", "iforest")
    m2 = json.load(open(workspace["ws"] / "models_global" / "baseline_iforest_metrics.json", encoding="utf-8"))
    assert m2["gun_norm"] is None and m2["metrics"]["val"]["files_gun_normalised"] == 0

    # --exclude-non-welding must run (it used to raise KeyError: 'non_welding')
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--val-frac", "0.5",
              "--exclude-non-welding", "--no-test", "--model-dir", workspace["ws"] / "models_xnw", "--model", "iforest")
    assert "fitting iforest" in out and "val: AUROC" in out

    out = run(os.path.join(PY, "train.py"), "--score", str(workspace["pre"] / "test" / "test_1.parquet"),
              "--model-dir", workspace["models"], "--model", "iforest")
    assert "windows, alarm rate" in out


def test_3b_rule_triggers():
    """The rule fires on an episode start (0 -> 1) and then keeps quiet for the cooldown."""
    from train import rule_triggers

    flags = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    ttf = np.arange(len(flags))[::-1] * 60.0
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=1800)).tolist() == [1]
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=60)).tolist() == [1, 4, 7]
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=240)).tolist() == [1, 7]


def test_3b2_rule_triggers_per_code():
    """C-6: per code, a switch of terminal code is a new onset and the cooldown does not cross codes."""
    from train import rule_triggers

    code = np.array([0, 2, 2, 4, 4, 0, 2, 0, 4])  # E016, switch to E029, E016 again, E029 again
    ttf = np.arange(len(code))[::-1] * 60.0
    flags = (code > 0).astype(int)
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=1800)).tolist() == [1], "shared cooldown: one trigger"
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=1800, code=code)).tolist() == [1, 3]
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=60, code=code)).tolist() == [1, 3, 6, 8]


def test_3c_rule_repeats():
    """A trigger whose code already triggered less than repeat_s earlier is a repeat; another code is not."""
    from train import rule_repeats

    trig = np.array([1, 0, 1, 0, 1, 1], dtype=bool)
    code = np.array([4, 0, 4, 0, 1, 4])
    ttf = np.array([100, 90, 60, 50, 40, 0]) * 3600.0  # hours before failure
    assert np.flatnonzero(rule_repeats(trig, code, ttf, repeat_s=24 * 3600)).tolist() == []  # 40 h and 60 h apart
    assert np.flatnonzero(rule_repeats(trig, code, ttf, repeat_s=48 * 3600)).tolist() == [2]
    assert np.flatnonzero(rule_repeats(trig, code, ttf, repeat_s=61 * 3600)).tolist() == [2, 5]


def test_3d_ensemble_model(workspace):
    """--model ensemble (IsolationForest + LightGBM, B-7) trains, evaluates and serves through the same paths."""
    pytest.importorskip("lightgbm")
    d_ens = workspace["ws"] / "models_ens"
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--model-dir", d_ens, "--model", "ensemble",
              "--val-frac", "0.5", "--warmup-hours", "0.5", "--exclude-non-welding")
    assert "val: AUROC" in out and "test: AUROC" in out
    m = json.load(open(d_ens / "baseline_ensemble_metrics.json", encoding="utf-8"))
    assert m["model_type"] == "ensemble" and 0 < m["threshold"] < 1, "score = weighted CDF average in [0, 1]"
    import joblib

    model = joblib.load(d_ens / "baseline_ensemble.joblib")["model"]
    assert len(model.if_idx) == 46 - 10, "the forest leaves out hour_* and c7-c9"
    assert np.isfinite(model.fit_scores_).all() and model.lgbm_weight == 0.5

    os.environ["RSW_MODEL_PATH"] = str(d_ens / "baseline_ensemble.joblib")
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    with TestClient(main.app) as client:
        assert client.get("/model").json()["model"]["type"] == "ensemble"
        r = client.post("/predict", json={"gun_id": "G1", "readings": raw.iloc[:1200].to_dict("records")})
        assert r.status_code == 200, r.text
        res = r.json()
        d = main.app.state.detector
        off = pd.read_parquet(workspace["pre"] / "test" / "test_0.parquet")
        vec_off = d.window_vector(off.loc[pd.Timestamp(res["window_start"]):pd.Timestamp(res["window_end"])])
        assert res["anomaly_score"] == pytest.approx(float(main.trainlib.anomaly_score(d.model, vec_off[None])[0]), abs=1e-4)
        assert len(res["contributing_features"]) == 8
    os.environ.pop("RSW_MODEL_PATH")


def test_3e_ensemble_components():
    """B-9: any mix of iforest / lgbm / lgbm6h / et, equal-weight CDF mean, gun-level OOF CDFs; pre-P8 bundles
    (one LightGBM as `lgbm`, grids grid_if / grid_lgbm) still load and score the same."""
    pytest.importorskip("lightgbm")
    import pickle

    import train as T

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6)).astype(np.float32)
    Xp = (rng.normal(size=(150, 6)) + 1.0).astype(np.float32)
    g, gp, ttf = rng.integers(0, 8, 2000), rng.integers(0, 8, 150), rng.integers(0, 7 * 86400, 2000)
    e = T.EnsembleDetector(components=("iforest", "lgbm", "lgbm6h", "et"), neg_sub=800).fit(X, Xp, g, gp, ttf)
    assert set(e.sup) == {"lgbm", "lgbm6h", "et"} and set(e.grids) == set(e.components)
    s = T.anomaly_score(e, np.vstack([X, Xp]))
    assert s[2000:].mean() > s[:2000].mean() and 0.9 < float(np.quantile(e.fit_scores_, 0.99)) < 1.1
    assert np.allclose(pickle.loads(pickle.dumps(e)).score_samples(X[:50]), e.score_samples(X[:50]))
    with pytest.raises(ValueError):
        T.EnsembleDetector(components=("iforest", "knn"))
    with pytest.raises(ValueError):
        T.EnsembleDetector(components=("lgbm6h",)).fit(X, Xp, g, gp)  # no ttf
    # a pre-P8 bundle's state
    two = T.EnsembleDetector().fit(X, Xp, g, gp)
    old = {k: v for k, v in two.__dict__.items() if k not in ("components", "neg_sub", "sup", "grids")}
    old.update(lgbm=two.sup["lgbm"], grid_if=two.grids["iforest"], grid_lgbm=two.grids["lgbm"])
    legacy = T.EnsembleDetector.__new__(T.EnsembleDetector)
    legacy.__setstate__(old)
    assert legacy.components == ("iforest", "lgbm") and np.allclose(legacy.score_samples(X[:50]), two.score_samples(X[:50]))


def test_4_api_matches_offline(workspace):
    os.environ["RSW_MODEL_PATH"] = str(workspace["models"] / "baseline_iforest.joblib")
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    with TestClient(main.app) as client:
        card = client.get("/model").json()
        assert card["model"]["n_features"] == 46 and card["test"]["auroc"] is not None
        assert card["dropped_features"] == ["c19"] and "c19_mean" not in card["features"]
        assert card["preprocess"]["gap_fill_limit_s"] == 60
        assert card["model"]["gun_norm"] == "center" and card["model"]["gun_warmup_s"] == 1800
        assert card["gun_norm"]["threshold_q"] == 0.99

        # first 30 s -> warming up
        r = client.post("/predict", json={"gun_id": "G1", "readings": raw.iloc[:30].to_dict("records")})
        assert r.status_code == 202 and r.json()["status"] == "warming_up"
        # 20 minutes of history -> scored; window = the latest complete clock minute
        r = client.post("/predict", json={"gun_id": "G1", "readings": raw.iloc[30:1200].to_dict("records")})
        assert r.status_code == 200, r.text
        res = r.json()
        for k in ("is_anomaly", "anomaly_score", "threshold", "score_z", "severity", "sustained_alarm",
                  "consecutive_alarms", "contributing_features", "context", "model"):
            assert k in res
        assert res["n_samples"] == 60 and res["severity"] in ("normal", "warning", "critical")
        assert res["context"]["known_code_class_hint"] is None
        assert len(res["contributing_features"]) == 8 and res["contributing_features"][0]["reference"] != 0
        # 20 min < the 30 min warm-up: the model raises no alarm yet (bundle warmup_alarms false)
        assert card["model"]["warmup_alarms"] is False
        assert res["alarm_held"] is True and res["hold_reason"].startswith("gun warm-up")
        assert card["model"]["alarm_max_non_welding"] == 0.5
        assert res["gun_norm"] == "warming_up" and res["gun_threshold"] is None, "20 min < 30 min warm-up"
        assert res["threshold"] == pytest.approx(card["model"]["threshold"])

        # the API's online preprocessing must reproduce the offline pipeline on the same window
        d = main.app.state.detector
        off = pd.read_parquet(workspace["pre"] / "test" / "test_0.parquet")
        w_end = pd.Timestamp(res["window_end"])
        assert pd.Timestamp(res["window_start"]) == w_end - pd.Timedelta(seconds=59) and w_end.second == 59
        vec_off = d.window_vector(off.loc[w_end - pd.Timedelta(seconds=59):w_end])
        score_off = float(main.trainlib.anomaly_score(d.model, vec_off[None])[0])
        assert res["anomaly_score"] == pytest.approx(score_off, abs=1e-4)

        # past the warm-up: the gun statistics are fixed, later windows are re-normalised and judged
        # against the gun's own threshold - exactly as train.window_file() / gun_thresholds() do offline
        # (up to the 90 s gap: raw row 2390 ~ t = 2393 s > the 1800 s warm-up)
        r = client.post("/predict", json={"gun_id": "G1", "readings": raw.iloc[1200:2390].to_dict("records")})
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["gun_norm"] == "gun" and res["gun_threshold"] is not None
        assert res["alarm_held"] is False and res["hold_reason"] is None, "past the warm-up the model may alarm again"
        assert res["threshold"] == pytest.approx(res["gun_threshold"]) and res["threshold"] >= card["model"]["threshold"]
        gs = {g["gun_id"]: g for g in client.get("/guns").json()}["G1"]
        assert gs["gun_norm"] == "gun" and gs["warmup_rows"] >= 1700
        gn, cols = d.gun_norm, d.norm_cols
        stats = main.trainlib.gun_norm_stats(off, cols, gn)
        assert stats is not None and stats["n"] >= 1000
        off_n = main.trainlib.apply_gun_norm(off, cols, stats)
        w_end = pd.Timestamp(res["window_end"])
        vec_off = d.window_vector(off_n.loc[pd.Timestamp(res["window_start"]):w_end])
        score_off = float(main.trainlib.anomaly_score(d.model, vec_off[None])[0])
        assert res["anomaly_score"] == pytest.approx(score_off, abs=1e-3)
        w_off, calib = main.trainlib.window_file(off, 60, d.feat_cols, gn, cols)
        thr_off = main.trainlib.gun_thresholds(d.model, d.model_cols, calib, d.threshold, gn["threshold_q"],
                                                 d.alarm_max_non_welding)["test_0"]
        assert res["gun_threshold"] == pytest.approx(thr_off, abs=2e-3)
        # gun constants (c7-c9) are centred away: their window means are 0 after the warm-up
        assert all(c["value"] == 0 for c in res["contributing_features"] if c["sensor"] in ("c7", "c8", "c9"))

        # the failure end: terminal code -> class hint for the ontology stage
        r = client.post("/predict", json={"gun_id": "G2", "readings": raw.iloc[-1500:-750].to_dict("records")})
        assert r.status_code == 200 and r.json()["rule_triggered"] is False
        r = client.post("/predict", json={"gun_id": "G2", "readings": raw.iloc[-750:].to_dict("records")})
        assert r.status_code == 200 and r.json()["context"]["known_code_class_hint"] == "E04"
        assert r.json()["context"]["latest_error_code"] == "E029"
        # terminal-code rule: severity is critical whatever the model says
        assert r.json()["rule_triggered"] is True and r.json()["severity"] == "critical"
        assert r.json()["severity_source"] in ("rule", "model+rule") and r.json()["critical_in_request"] is True
        assert pd.Timestamp(r.json()["rule_trigger_time"]) == pd.Timestamp(raw["time"].iloc[-300]).tz_localize(None)
        assert r.json()["rule_code"] == "E029" and r.json()["rule_class_hint"] == "E04"
        # ML -> RAG handoff rides on the rule trigger and lands in the outbox (pull; no RSW_RAG_URL in tests)
        ho = r.json()["handoff"]
        assert ho is not None and ho["schema_version"] == "1.4" and ho["gun_id"] == "G2"
        assert ho["event_type"] == "critical" and ho["early_warning"] is None
        # v1.1 (FE requests 3-5): UTC times with Z, controller code runs, sensor trend, Korean names
        assert ho["detected_at"].endswith("Z") and ho["window_start"].endswith("Z")
        assert ho["trigger"]["rule_trigger_time"].endswith("Z")
        runs = ho["error_timeline"]
        assert any(x["code"] == "E029" and x["class_hint"] == "E04" and x["duration_s"] >= 1 for x in runs)
        assert all(x["start"].endswith("Z") and x["end"] >= x["start"] for x in runs)
        assert all(len(s["points"]) <= 30 and s["sensor_name_ko"] and all(p["t"].endswith("Z") for p in s["points"])
                   for s in ho["sensor_trend"])
        assert all(c["sensor_name_ko"] for c in r.json()["contributing_features"])
        assert ho["fault_class"]["code"] == "E04" and ho["fault_class"]["basis"] == "rule_trigger"
        assert ho["trigger"]["source"] in ("rule", "model+rule") and ho["trigger"]["rule_code"] == "E029"
        assert ho["fault_class"]["terminal_code"] == "E029" and "E04" in ho["summary_ko"]
        assert "situation_ids" not in ho and "symptoms" not in ho, "v1.2: no situation-doc mapping in ML"
        assert "hypotheses" not in ho and "search" not in ho, "causes / manuals are the ontology's job"
        recs = client.get("/handoffs", params={"gun_id": "G2"}).json()
        assert recs[-1]["handoff"]["event_id"] == ho["event_id"] and recs[-1]["delivery"] == "pull_only"
        assert client.get(f"/handoffs/{ho['event_id']}").status_code == 200
        assert client.get("/handoffs/nope").status_code == 404
        prev = client.post("/handoffs/preview", json={k: v for k, v in r.json().items() if k != "handoff"})
        assert prev.status_code == 200 and prev.json()["fault_class"]["code"] == "E04"
        assert prev.json()["fault_class"] == ho["fault_class"]
        n_off = len(main.trainlib.window_features(off.loc[pd.Timestamp(raw["time"].iloc[-750]).tz_localize(None):],
                                                  60, d.feat_cols))
        assert r.json()["windows_scored"] == n_off, "every clock minute of the chunk with >= 30 samples"
        assert res["rule_triggered"] is False and res["severity_source"] in ("none", "model")
        assert res["handoff"] is None or res["critical_in_request"], "handoffs only on critical events"

        # a window inside the cap-dressing block: alarms are held, never counted, severity stays normal
        r = client.post("/predict", json={"gun_id": "G3", "readings": raw.iloc[1400:1700].to_dict("records")})
        assert r.status_code == 200, r.text
        held = r.json()
        assert held["context"]["non_welding_share"] == 1.0 and held["alarm_held"] is True
        assert held["hold_reason"].startswith("non_welding_share 1.00 > gate 0.5")
        assert held["severity"] == "normal" and held["consecutive_alarms"] == 0

        # row-based warm-up: it ends right after the 900th normal welding row, online as offline
        d.gun_norm = {**gn, "warmup_rows": 900}
        try:
            for a in range(0, 2390, 300):
                client.post("/predict", json={"gun_id": "G4", "readings": raw.iloc[a:min(a + 300, 2390)].to_dict("records")})
            g4 = main.app.state.guns["G4"]
            st = main.trainlib.gun_norm_stats(off, cols, d.gun_norm)
            assert g4.norm_status == "gun" and g4.norm["t_end"] == st["t_end"] < off.index[0] + pd.Timedelta(seconds=1800)
            assert np.allclose(g4.norm["mean"], st["mean"], atol=1e-5)
            _, calib4 = main.trainlib.window_file(off, 60, d.feat_cols, d.gun_norm, cols)
            thr4 = main.trainlib.gun_thresholds(d.model, d.model_cols, calib4, d.threshold, gn["threshold_q"],
                                                d.alarm_max_non_welding).get("test_0")
            assert g4.gun_threshold == pytest.approx(thr4, abs=2e-3)
        finally:
            d.gun_norm = gn

        assert {g["gun_id"] for g in client.get("/guns").json()} == {"G1", "G2", "G3", "G4"}
        assert client.delete("/guns/G1").status_code == 204
        assert client.delete("/guns/G1").status_code == 404


def test_4b_handoff_timeline_keeps_the_triggering_code(workspace):
    """C-7: a rule fired in the minute that is not complete yet (after `window_end`) still shows its code in the
    handoff's `error_timeline`. With 1-min streaming this is the usual case; the timeline used to stop at window_end."""
    main = load_main(workspace)
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    part = raw.iloc[:900].copy()
    t = pd.to_datetime(part["time"], utc=True)
    part.loc[(t >= t.iloc[-1].floor("min")).to_numpy(), "error"] = "E029"  # only the minute still in progress
    with TestClient(main.app) as client:
        r = client.post("/predict", json={"gun_id": "TL", "readings": part.to_dict("records")}).json()
    assert r["rule_triggered"] is True and r["rule_code"] == "E029"
    assert pd.Timestamp(r["rule_trigger_time"]) > pd.Timestamp(r["window_end"]), "trigger after the last full window"
    runs = r["handoff"]["error_timeline"]
    assert any(x["code"] == "E029" and x["class_hint"] == "E04" for x in runs), runs
    os.environ.pop("RSW_MODEL_PATH")


def card_repeat(client):
    return client.get("/model").json()["model"]["rule_repeat_s"]


def test_5_api_chunks_sustain_rule(workspace):
    """Chunk limit, time-based sustained alarm, and the edge-triggered terminal-code rule with cooldown."""
    os.environ["RSW_MODEL_PATH"] = str(workspace["models"] / "baseline_iforest.joblib")
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})

    def post(client, gun, part):
        return client.post("/predict", json={"gun_id": gun, "readings": part.to_dict("records")})

    with TestClient(main.app) as client:
        # a chunk larger than the buffer minus 10 min of history is refused, not silently truncated
        assert main.MAX_CHUNK == 1200
        assert post(client, "big", raw.iloc[:1201]).status_code == 422

        d = main.app.state.detector
        thr = d.threshold
        d.threshold = -1e9  # every window alarms (all guns below are still in their warm-up -> global threshold)
        d.warmup_alarms = True  # ... and may alarm there (the bundle holds warm-up alarms: history.md §17)
        try:
            # one 1170-row chunk: every window is scored, the run is long enough -> sustained
            res = post(client, "S1", raw.iloc[:1170]).json()
            assert res["windows_scored"] == 19 and res["consecutive_alarms"] == 19
            assert res["sustained_alarm"] is True and res["severity"] == "critical" and res["alarm_duration_s"] >= 1100
            # 10 s chunks: a clock window is scored once, when complete - sustained after
            # sustain x window = 180 s of continuous alarm, whatever the call rate
            first, sustained_at = None, None
            for i in range(0, 400, 10):
                r = post(client, "S2", raw.iloc[i: i + 10])
                if r.status_code != 200:
                    continue
                res = r.json()
                first = first or pd.Timestamp(res["window_start"])
                if res["sustained_alarm"]:
                    sustained_at = pd.Timestamp(res["window_end"])
                    assert res["consecutive_alarms"] == 3
                    break
            assert sustained_at is not None and 179 <= (sustained_at - first).total_seconds() < 190
        finally:
            d.threshold, d.warmup_alarms = thr, False

        # the rule fires once per episode start and is quiet during the cooldown, even if the code comes back
        part = raw.iloc[:600].copy()
        part.loc[150:199, "error"] = "E029"  # a cross-class E029 flap long before the failure
        part.loc[250:299, "error"] = "E029"
        triggers, active = [], 0
        for i in range(0, 600, 60):
            r = post(client, "R1", part.iloc[i: i + 60])
            if r.status_code == 200:
                res = r.json()
                triggers.append(res["rule_triggered"])
                active += res["rule_code_active"]
                if res["rule_triggered"]:
                    assert res["critical_in_request"] and res["severity"] == "critical" and res["context"]["known_code_class_hint"] == "E04"
                elif not res["sustained_alarm"]:
                    assert res["severity"] != "critical", "a persisting terminal code alone is not critical"
        assert sum(triggers) == 1 and active >= 1

        # the same code firing again after the cooldown but within rule_repeat_s is a repeat: reported, not critical
        assert card_repeat(client) == 86400
        part = raw.iloc[:2280].copy()
        part.loc[150:199, "error"] = "E029"
        part.loc[2100:2149, "error"] = "E029"  # 32.5 min later: past the 30-min cooldown
        fired = []
        for i in range(0, 2280, 60):
            r = post(client, "R2", part.iloc[i: i + 60])
            if r.status_code == 200 and r.json()["rule_triggered"]:
                fired.append(r.json())
        assert [f["rule_repeat"] for f in fired] == [False, True]
        assert fired[0]["severity"] == "critical" and fired[0]["handoff"] is not None
        rep = fired[1]
        assert rep["severity_source"] in ("none", "model") and rep["handoff"] is None
        assert rep["severity"] == "critical" if rep["critical_in_request"] else rep["severity"] == "warning"

        # a cap-dressing block longer than the 30-min buffer: the non-welding rows keep carrying the last welding
        # values (regression: the buffer held no welding row any more -> NaN features -> HTTP 500 on real test_0)
        assert post(client, "CD", raw.iloc[:1200]).status_code == 200
        for a, b in ((1200, 2390), (2390, 3590)):
            r = post(client, "CD", raw.iloc[a:b].assign(c16=0.0))
            assert r.status_code == 200, r.text
        res = r.json()
        assert res["context"]["non_welding_share"] == 1.0 and res["alarm_held"] is True
        assert np.isfinite(res["anomaly_score"])
        # a stream that starts in non-welding has nothing to carry yet -> 202, not 500
        r = post(client, "CD0", raw.iloc[:120].assign(c16=0.0))
        assert r.status_code == 202 and "no welding row" in r.json()["message"]

        # replay client: the whole 2 h test file in 5-min chunks -> one rule event (the terminal code at the end)
        from replay import replay_file

        s = replay_file(client, str(workspace["test"] / "test_0.csv"), "RP", chunk_s=300)
        assert s["requests"] == 24 and s["rule_triggers"] == 1
        rule_events = [e for e in s["critical_events"] if e["rule_trigger_time"]]
        assert len(rule_events) == 1 and rule_events[0]["class_hint"] == "E04" and "rule" in rule_events[0]["severity_source"]
        assert all(e["handoff_event_id"] for e in s["critical_events"]), "every critical event carries a handoff"
        assert s["windows_scored"] >= 100 and s["last"]["gun_norm"] == "gun"
        with pytest.raises(ValueError):
            replay_file(client, str(workspace["test"] / "test_0.csv"), "RP", chunk_s=1201)

        # cause chart feed: every scored window of the replay, in order, with the rule event and the focus explanation
        tr = client.get("/guns/RP/trace", params={"minutes": 360}).json()
        pts = tr["points"]
        assert tr["n_points"] == len(pts) >= 100 and tr["window_s"] == 60
        ends = pd.to_datetime([p["window_end"] for p in pts])
        assert ends.is_monotonic_increasing and ends.is_unique
        assert all(pd.Timestamp(p["window_end"]).second == 59 or p is pts[-1] for p in pts), "clock-aligned windows"
        # C-2: the served windows ARE the offline windows - same minutes, same scores (train.score_frame)
        off = main.trainlib.score_frame(d.bundle, pd.read_parquet(workspace["pre"] / "test" / "test_0.parquet"))
        on = pd.Series([p["score"] for p in pts], index=pd.to_datetime([p["window_start"] for p in pts]).floor("60s"))
        assert on.index.isin(off.index).all() and len(on) >= len(off) - 2, "at most the partial last minute(s) missing"
        assert np.allclose(on.to_numpy(), off.loc[on.index, "score"].to_numpy(), atol=1e-4)
        # C-4: the daily counters count every served window once, with the offline verdicts
        st = client.get("/guns/RP/stats", params={"days": 31}).json()
        tot, offc = st["total"], off.loc[on.index]
        ok = offc["error_active"] == 0
        assert tot["windows"] == len(pts) and tot["normal"] == int(ok.sum())
        assert tot["normal_alarms"] == int((ok & offc["alarm"].astype(bool)).sum())
        assert tot["alarms"] == int(offc["alarm"].astype(bool).sum()) and tot["rule_triggers"] == 1
        assert tot["critical_events"] == len(s["critical_events"]) and st["drift_warning"] == []
        assert [g["gun_id"] for g in client.get("/stats").json()] == sorted(g["gun_id"] for g in client.get("/guns").json())
        assert pts[-1]["score"] == pytest.approx(s["last"]["anomaly_score"], abs=1e-6), "last point = last response"
        assert pts[0]["gun_norm"] == "warming_up" and pts[-1]["gun_norm"] == "gun"
        rules = [e for e in tr["events"] if e["kind"] == "rule"]
        assert len(rules) == 1 and rules[0]["code"] == "E029" and rules[0]["class_hint"] == "E04"
        assert any("E029" in p["error_codes"] for p in pts[-5:])
        assert tr["focus"]["severity"] == "critical", "the rule trigger is the latest non-normal window"
        feats = [f["feature"] for f in tr["features"]]
        assert 1 <= len(feats) <= 4 and not any(f.startswith("hour_") for f in feats)
        assert all(f["sensor_name_ko"] for f in tr["features"]), "FE request 5: Korean names on the trace too"
        assert all(set(p["deviation"]) == set(feats) for p in pts)
        lo, hi = tr["axes"]["score"]
        assert lo < min(p["threshold"] for p in pts) < hi and tr["axes"]["deviation"] == [-4.0, 4.0]
        # the deviation is the model input minus the normal reference - the same numbers as contributing_features
        c = tr["focus"]["contributing_features"][0]
        fp = [p for p in pts if p["window_end"] == tr["focus"]["window_end"]]
        if fp and c["feature"] in fp[0]["deviation"]:
            assert fp[0]["deviation"][c["feature"]] == pytest.approx(c["value"] - c["reference"], abs=1e-3)
        one = client.get("/guns/RP/trace", params={"minutes": 30, "features": ["c5_mean", "c2_std"]}).json()
        assert [f["feature"] for f in one["features"]] == ["c5_mean", "c2_std"] and one["n_points"] <= 30
        assert client.get("/guns/RP/trace", params={"features": ["nope_mean"]}).status_code == 422
        assert client.get("/guns/NOPE/trace").status_code == 404
        view = client.get("/guns/RP/trace/view")
        assert view.status_code == 200 and "../trace?minutes=" in view.text


def test_6_serving_edge_cases(workspace):
    """Regressions of the 2026-09-29 review (history.md §14): rule during 202, stray timestamps, raw-unit context
    after the warm-up, carry at the buffer start, rule before a gap, trace cadence, code switches, input checks."""
    os.environ["RSW_MODEL_PATH"] = str(workspace["models"] / "baseline_iforest.joblib")
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    t = pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None)

    def post(client, gun, part):
        return client.post("/predict", json={"gun_id": gun, "readings": part.to_dict("records")})

    with TestClient(main.app) as client:
        # a terminal code that starts while the response is 202 (new segment after a long gap) is not lost
        assert post(client, "P", raw.iloc[:300]).status_code == 200
        after_gap = raw.iloc[400:430].assign(error="E029")  # 100 s gap -> new segment of 30 rows -> 202
        r = post(client, "P", after_gap)
        assert r.status_code == 202
        r = post(client, "P", raw.iloc[430:500].assign(error="E029"))
        assert r.status_code == 200 and r.json()["rule_triggered"] is True and r.json()["rule_code"] == "E029"
        assert pd.Timestamp(r.json()["rule_trigger_time"]) == t.iloc[400]

        # a request that spans a long gap scores the complete minutes BEFORE the gap too (it used to score its last
        # segment only: here 3 rows after the synthetic 90 s gap -> 202, and 20 windows lost)
        assert post(client, "SEG", raw.iloc[:1200]).status_code == 200
        r = post(client, "SEG", raw.iloc[1200:2400])  # rows up to t = 2399 s, then 3 rows after the gap (t >= 2490 s)
        assert r.status_code == 200, r.text
        off = pd.read_parquet(workspace["pre"] / "test" / "test_0.parquet")
        t0 = off.index[0]
        n_off = len(main.trainlib.window_features(off.loc[t0 + pd.Timedelta(seconds=1200):t0 + pd.Timedelta(seconds=2399)], 60,
                                                   main.app.state.detector.feat_cols))
        assert r.json()["windows_scored"] == n_off == 20
        assert r.json()["window_end"].endswith("00:39:59"), "the latest complete window is the last minute before the gap"

        # a switch from one terminal code to another: per-code rule (C-6, bundle default) -> a new trigger; the shared
        # cooldown of older bundles (train.rule_triggers on terminal_any) -> the same episode
        r = post(client, "SW", raw.iloc[:100].assign(error="E016"))
        assert r.json()["rule_code"] == "E016"
        r = post(client, "SW", raw.iloc[100:200].assign(error="E029"))
        assert r.json()["rule_triggered"] is True and r.json()["rule_code"] == "E029", "per-code: the switch fires"
        d = main.app.state.detector
        d.rule_per_code = False
        try:
            post(client, "SW0", raw.iloc[:100].assign(error="E016"))
            r = post(client, "SW0", raw.iloc[100:200].assign(error="E029"))
            assert r.json()["rule_triggered"] is False, "shared cooldown: E016 -> E029 is the same episode"
        finally:
            d.rule_per_code = True

        # a terminal code before a long gap in the same chunk is still seen by the rule
        chunk = pd.concat([raw.iloc[:120].assign(error=["0"] * 60 + ["E012"] * 60), raw.iloc[300:420]])
        r = post(client, "GAP", chunk)
        assert r.status_code == 200 and r.json()["rule_code"] == "E012"

        # stray timestamps: a chunk longer than the buffer, and a clock that went back, are refused (not 30 min blind)
        stray = raw.iloc[600:610].copy()
        stray.loc[stray.index[-1], "time"] = (t.iloc[609] + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert post(client, "P", stray).status_code == 422
        assert post(client, "P", raw.iloc[:10]).status_code == 200  # 10 min older than the newest row is fine...
        assert post(client, "OLD", raw.iloc[2400:2410]).status_code == 202
        assert post(client, "OLD", raw.iloc[:10]).status_code == 409  # ...40 min older is not
        # NaN readings are rejected, sub-second timestamps are floored onto the 1 Hz grid
        bad = raw.iloc[:60].to_dict("records")
        bad[5]["c5"] = float("nan")
        body = json.dumps({"gun_id": "NAN", "readings": bad})  # Python's json writes NaN, as a sloppy client would
        assert client.post("/predict", content=body, headers={"content-type": "application/json"}).status_code == 422
        frac = raw.iloc[:120].assign(time=(t.iloc[:120] + pd.Timedelta(milliseconds=300)).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        assert post(client, "FRAC", frac).status_code == 200

        # trace cadence: 1-row requests score overlapping windows, the trace keeps one point per 60 s
        post(client, "ONE", raw.iloc[:120])
        for i in range(120, 240):
            post(client, "ONE", raw.iloc[i:i + 1])
        pts = client.get("/guns/ONE/trace", params={"minutes": 360}).json()["points"]
        ends = pd.to_datetime([p["window_end"] for p in pts])
        assert len(pts) <= 5 and (ends[1:] - ends[:-1] >= pd.Timedelta(seconds=60)).all()

        # after the warm-up (30 min here) the context is in raw units: welds per 60 s window, not a gun-centred z
        for a, b in ((0, 600), (600, 1200), (1200, 1800), (1800, 2380)):  # the synthetic 90 s gap starts at row 2390
            r = post(client, "W", raw.iloc[a:b])
        res = r.json()
        assert res["gun_norm"] == "gun"
        ws, we = pd.Timestamp(res["window_start"]), pd.Timestamp(res["window_end"])
        c11 = raw["c11"].set_axis(t)
        welds = c11.loc[ws - pd.Timedelta(seconds=1):we].diff().clip(lower=0).sum()
        assert abs(res["context"]["welds_in_window"] - welds) <= 1.0
        assert 0 <= res["context"]["welds_10min"] <= 600

    # drift signal (C-4): > 3 % normal alarms after the warm-up on 3 consecutive days, or 2 critical rule triggers a day
    g = main.GunState()
    for day, alarms in (("2021-10-01", 20), ("2021-10-02", 30), ("2021-10-03", 25)):
        main.count(g, pd.Timestamp(day), normal_after_warmup=500, normal_alarms_after_warmup=alarms)
    assert len(main.drift_reasons(g)) == 1 and "3 consecutive days" in main.drift_reasons(g)[0]  # 4 %, 6 %, 5 %
    main.count(g, pd.Timestamp("2021-10-05"), normal_after_warmup=500, normal_alarms_after_warmup=40)
    assert main.drift_reasons(g) == [], "10-03 -> 10-05 is not 3 consecutive days"
    main.count(g, pd.Timestamp("2021-10-05"), rule_triggers=3, rule_repeats=1)
    assert main.drift_reasons(g) == ["2 critical terminal-code rule triggers on 2021-10-05"]
    few = main.GunState()
    for day in ("2021-10-01", "2021-10-02", "2021-10-03"):
        main.count(few, pd.Timestamp(day), normal_after_warmup=100, normal_alarms_after_warmup=50)
    assert main.drift_reasons(few) == [], "less than 6 h of normal windows a day is not evidence"
    d = main.app.state.detector
    buf = raw.iloc[:200].copy()
    buf["time"] = t.iloc[:200].values
    buf["c10"] = 1.0
    buf.loc[buf.index[:50], "c16"] = 0.0  # starts inside a cap-dressing block
    buf.loc[buf.index[50:], "c1"] = -3.0
    carry = {c: float(buf[c].iloc[60]) for c in main.VALUE_COLS + [main.BINARY_COL]} | {"c1": 5.0}
    f = d.features(buf, carry).f
    c1 = f["c1"] * d.scaler["c1"]["std"] + d.scaler["c1"]["mean"]
    assert c1.iloc[:50].round(3).eq(5.0).all() and c1.iloc[50:].round(3).eq(-3.0).all()


def load_main(workspace, model="baseline_iforest.joblib"):
    os.environ["RSW_MODEL_PATH"] = str(workspace["models"] / model)
    sys.path.insert(0, ROOT)
    return importlib.reload(importlib.import_module("main"))


def wait_for(cond, timeout=15.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_7_state_survives_restart(workspace, tmp_path, monkeypatch):
    """C-1: a restart mid-stream keeps the gun statistics / threshold, the rule cooldown and the 24 h repeat history;
    a restart under another model sends the gun back to warming up; DELETE removes the state file."""
    main = load_main(workspace)
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    col = raw.columns.get_loc("error")
    # E029 fires (critical), fires again after the cooldown (repeat), is quiet in the cooldown right AFTER the
    # restart (only the persisted last trigger knows), fires once more (repeat - only the persisted history knows)
    for a, b in ((100, 150), (2100, 2150), (2600, 2650), (4000, 4050)):
        raw.iloc[a:b, col] = "E029"
    keys = ("window_end", "gun_norm", "gun_threshold", "threshold", "rule_triggered", "rule_repeat", "rule_code")

    def stream(state, restart_at=None):
        monkeypatch.setenv("RSW_STATE_DIR", str(tmp_path / state))
        out, client = [], TestClient(main.app)
        client.__enter__()
        try:
            for a in range(0, 5000, 300):
                if a == restart_at:
                    client.__exit__(None, None, None)
                    client = TestClient(main.app)
                    client.__enter__()
                    g = client.get("/guns").json()[0]
                    assert g["gun_norm"] == "gun" and g["gun_threshold"] is not None, "restored after the warm-up"
                r = client.post("/predict", json={"gun_id": "K", "readings": raw.iloc[a:a + 300].to_dict("records")})
                if r.status_code == 200:
                    out.append({k: r.json()[k] for k in keys})
        finally:
            client.__exit__(None, None, None)
        return out

    once, twice = stream("once"), stream("twice", restart_at=2400)
    assert once == twice
    fired = [(x["rule_code"], x["rule_repeat"]) for x in twice if x["rule_triggered"]]
    assert fired == [("E029", False), ("E029", True), ("E029", True)], fired

    # the state file holds no rows; under another model the gun warms up again (its threshold is the old model's)
    snap = json.load(open(tmp_path / "twice" / "guns" / "K.json", encoding="utf-8"))
    assert snap["norm_status"] == "gun" and snap["rule_last_by_code"]["E029"] and "rows" not in snap
    # the daily counters (C-4) carry on across the restart. The buffer is not persisted, so the minute in progress at
    # the restart loses its first rows and may fall under the 30-sample minimum: at most one window fewer
    ref = json.load(open(tmp_path / "once" / "guns" / "K.json", encoding="utf-8"))["stats"]
    for k in ("windows", "normal"):
        assert 0 <= sum(c[k] for c in ref.values()) - sum(c[k] for c in snap["stats"].values()) <= 1, k
    for k in ("rule_triggers", "rule_repeats", "critical_events"):
        assert sum(c[k] for c in snap["stats"].values()) == sum(c[k] for c in ref.values()), k
    assert sum(c["rule_triggers"] for c in snap["stats"].values()) == 3
    import joblib

    b = joblib.load(workspace["models"] / "baseline_iforest.joblib")
    b["created"] = "another model"
    joblib.dump(b, tmp_path / "other.joblib")
    monkeypatch.setenv("RSW_MODEL_PATH", str(tmp_path / "other.joblib"))
    with TestClient(main.app) as client:
        g = client.get("/guns").json()[0]
        assert g["gun_norm"] == "warming_up" and g["gun_threshold"] is None
        assert client.delete("/guns/K").status_code == 204
    assert not (tmp_path / "twice" / "guns" / "K.json").exists()
    os.environ.pop("RSW_MODEL_PATH")


def test_8_outbox_retry_restart_idempotent(workspace, tmp_path, monkeypatch):
    """D-1: handoffs made while the RAG is down are retried with backoff, survive a restart, and reach the RAG
    exactly once when it is back; a 4xx is not retried; a resend after the restart makes no new event."""
    import httpx
    import mock_rag

    main = load_main(workspace)
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RSW_RAG_URL", "http://rag.test/diagnose")
    monkeypatch.setenv("RSW_RAG_BACKOFF_S", "0.05,0.1")
    monkeypatch.setenv("RSW_RAG_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("RSW_OUTBOX_POLL_S", "0.02")
    mock_rag.RECEIVED.clear()
    mock_rag.ANSWERED.clear()
    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    part = raw.iloc[:600].copy()
    part.iloc[150:200, part.columns.get_loc("error")] = "E029"

    def post(client, gun):
        return client.post("/predict", json={"gun_id": gun, "readings": part.to_dict("records")}).json()

    def down(request):
        raise httpx.ConnectError("RAG down")

    main.app.state.rag_transport = httpx.MockTransport(down)
    with TestClient(main.app) as client:
        ids = [post(client, g)["handoff"]["event_id"] for g in ("H1", "H2")]
        assert wait_for(lambda: all(r["delivery"] == "failed" for r in client.get("/handoffs").json()))
        recs = client.get("/handoffs").json()
        assert [r["handoff"]["event_id"] for r in recs] == ids and all(r["attempts"] == 3 for r in recs)
        assert "ConnectError" in recs[0]["delivery_detail"] and "after 3 attempts" in recs[0]["delivery_detail"]
    assert sorted(os.listdir(tmp_path / "state" / "outbox")) == sorted(f"{i}.json" for i in ids)

    main.app.state.rag_transport = httpx.ASGITransport(app=mock_rag.app)
    with TestClient(main.app) as client:
        assert wait_for(lambda: all(r["delivery"] == "delivered" for r in client.get("/handoffs").json()))
        assert client.get("/health").json()["outbox"]["delivered"] == 2
        # the same readings again after the restart: the persisted rule state knows them, no second event
        assert post(client, "H1")["handoff"] is None and len(client.get("/handoffs").json()) == 2
    got = [h["event_id"] for h in mock_rag.RECEIVED]
    assert sorted(got) == sorted(ids), "every event exactly once"
    rec = json.load(open(tmp_path / "state" / "outbox" / f"{ids[0]}.json", encoding="utf-8"))
    assert rec["delivery"] == "delivered" and rec["rag_response"]["generator"] == "mock"
    # the receiver drops a duplicate event_id (the push retries until it sees a 2xx)
    with TestClient(mock_rag.app) as rag:
        again = rag.post("/diagnose", json=rec["handoff"]).json()
    assert again["duplicate"] is True and len(mock_rag.RECEIVED) == 2

    # a 4xx is a contract problem: failed at once, not retried, not requeued by a restart
    main.app.state.rag_transport = httpx.MockTransport(lambda request: httpx.Response(422, json={"detail": "bad"}))
    with TestClient(main.app) as client:
        eid = post(client, "H3")["handoff"]["event_id"]
        assert wait_for(lambda: client.get(f"/handoffs/{eid}").json()["delivery"] == "failed")
        assert client.get(f"/handoffs/{eid}").json()["attempts"] == 1
    with TestClient(main.app) as client:
        time.sleep(0.2)
        r = client.get(f"/handoffs/{eid}").json()
        assert r["delivery"] == "failed" and r["delivery_detail"].startswith("rejected") and r["attempts"] == 1
    os.environ.pop("RSW_MODEL_PATH")


def test_8b_push_sends_nan_as_null(workspace):
    """C-7: a handoff holding NaN / inf still reaches the receiver, with null in those fields. pydantic's JSON mode
    keeps NaN as a float and httpx refuses it in `json=`, so every attempt used to fail and the event was lost."""
    import asyncio
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import httpx

    main = load_main(workspace)
    doc = json.load(open(os.path.join(ROOT, "docs", "rag_handoff.example.json"), encoding="utf-8"))
    ho = main.RagHandoff(**doc)
    ho.trigger.anomaly_score, ho.trigger.score_z = float("nan"), float("inf")
    now = datetime.now(timezone.utc)
    rec = main.HandoffRecord(handoff=ho, created_at=now, delivery="pending", next_attempt_at=now)
    seen = []

    def receive(request):
        seen.append((request.headers.get("content-type", ""), request.headers.get("idempotency-key"),
                     json.loads(request.content)))
        return httpx.Response(200, json={"event_id": doc["event_id"], "status": "ok"})

    app = SimpleNamespace(state=SimpleNamespace(
        rag_url="http://api.test/diagnose", rag_transport=httpx.MockTransport(receive), outbox=main.Outbox(None),
        rag_backoff_s=(0.05,), rag_max_attempts=3))
    asyncio.run(main.push_handoff(app, rec))
    assert (rec.delivery, rec.attempts) == ("delivered", 1), rec.delivery_detail
    ctype, key, body = seen[0]
    assert ctype.startswith("application/json") and key == doc["event_id"]
    assert body["trigger"]["anomaly_score"] is None and body["trigger"]["score_z"] is None
    assert body["event_id"] == doc["event_id"] and body["gun_id"] == doc["gun_id"]
    os.environ.pop("RSW_MODEL_PATH")



def test_9_restart_hold_and_critical_sustain(workspace, monkeypatch):
    """C-5: no model alarm for 30 min after an idle block (offline train.restart_flags = online, one state machine),
    and the model alone is critical only after critical_sustain (10) windows; sustained_alarm keeps sustain (3)."""
    from train import RESTART_HOLD_DEFAULT, restart_flags

    def frame(duty, t=None):
        t = pd.date_range("2021-09-01", periods=len(duty), freq="60s") if t is None else t
        return pd.DataFrame({"weld_duty_10min_mean": duty, "non_welding": 0.0, "file": "g"}, index=t)

    # 5 working, 30 idle, 40 working windows -> the first 30 working windows after the block are held
    held = restart_flags(frame(np.r_[np.full(5, 0.3), np.zeros(30), np.full(40, 0.3)]), RESTART_HOLD_DEFAULT)
    assert np.flatnonzero(held).tolist() == list(range(35, 65))
    # 29 idle windows, then work: not a block
    assert not restart_flags(frame(np.r_[np.full(5, 0.3), np.zeros(29), np.full(10, 0.3)]), RESTART_HOLD_DEFAULT).any()
    # a data gap counts as idle: 5 working windows, 40 min without any window, then work -> held
    t = pd.DatetimeIndex(list(pd.date_range("2021-09-01", periods=5, freq="60s"))
                         + list(pd.date_range("2021-09-01 00:45", periods=40, freq="60s")))
    assert np.flatnonzero(restart_flags(frame(np.full(45, 0.3), t), RESTART_HOLD_DEFAULT)).tolist() == list(range(5, 35))
    # an idle window inside a running hold is held too
    duty = np.r_[np.full(2, 0.3), np.zeros(30), np.full(5, 0.3), [0.0], np.full(5, 0.3)]
    assert restart_flags(frame(duty), RESTART_HOLD_DEFAULT)[37]

    monkeypatch.setenv("RSW_MODEL_PATH", str(workspace["models"] / "baseline_iforest.joblib"))
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    raw = raw.iloc[:-600].copy()  # no terminal code: the model alone
    t0 = pd.Timestamp(raw["time"].iloc[0]).tz_localize(None)
    sec = (pd.to_datetime(raw["time"]).dt.tz_localize(None) - t0).dt.total_seconds().to_numpy()
    raw.loc[(sec >= 2000) & (sec < 4700), "c2"] = 0  # 45 min without pressing (welding setpoint kept: not cap dressing)
    with TestClient(main.app) as client:
        d = main.app.state.detector
        card = client.get("/model").json()["model"]
        assert card["critical_sustain"] == 10 and card["sustain"] == 3
        assert card["restart_hold_s"] == 1800 and card["restart_idle_windows"] == 30
        thr, wa = d.threshold, d.warmup_alarms
        d.threshold = -1e9  # every window alarms unless held
        try:
            out = []
            for a in range(0, len(raw), 60):
                r = client.post("/predict", json={"gun_id": "RH", "readings": raw.iloc[a:a + 60].to_dict("records")})
                if r.status_code == 200:
                    out.append(r.json())
        finally:
            d.threshold = thr
        res = pd.DataFrame(out).drop_duplicates("window_start")
        res["start"] = pd.to_datetime(res["window_start"])
        rs = res[res["hold_reason"].fillna("").str.startswith("restart")]
        assert len(rs) >= 25, "a 30 min hold after the idle block"
        assert rs["start"].min() > t0 + pd.Timedelta(seconds=4700), "the hold starts when welding resumes"
        assert (rs["start"].max() - rs["start"].min()).total_seconds() < 1800
        assert (rs["severity"] == "normal").all() and (rs["consecutive_alarms"] == 0).all()
        # online = offline: the same state machine on the served windows
        g = main.app.state.guns["RH"]
        assert g.restart["restart_at"] is not None and g.restart["restart_at"] == rs["start"].min().floor("60s")
        # an alarm run (no hold): sustained (warning) after 3 windows, critical after 10
        d.threshold, d.warmup_alarms = -1e9, True
        try:
            out = []
            for a in range(0, 1200, 60):
                r = client.post("/predict", json={"gun_id": "CS", "readings": raw.iloc[a:a + 60].to_dict("records")})
                if r.status_code == 200:
                    out.append(r.json())
        finally:
            d.threshold, d.warmup_alarms = thr, wa
        run = pd.DataFrame(out).drop_duplicates("window_start")
        sus = run[run["sustained_alarm"] & (run["alarm_duration_s"] < 600)]
        assert len(sus) >= 5 and (sus["severity"] == "warning").all() and not sus["critical_in_request"].any()
        assert not sus["model_critical"].any()
        crit = run[run["severity"] == "critical"]
        assert len(crit) >= 3 and (crit["alarm_duration_s"] >= 600).all() and crit["critical_in_request"].all()
        assert crit["model_critical"].all()
        assert (crit["severity_source"] == "model").all() and crit["handoff"].iloc[0] is not None
        # the chart marks critical-length runs, not 3-window ones
        pts = client.get("/guns/CS/trace", params={"minutes": 60}).json()["points"]
        assert any(p["sustained"] and not p["critical"] for p in pts) and any(p["critical"] for p in pts)
        # the restart state survives a restart of the server (snapshot / restore)
        snap = g.snapshot("RH", d.info.created)
        g2 = main.GunState.restore(snap, gun_norm=True, model_created=d.info.created)
        assert g2.restart == g.restart and g2.restart_held == g.restart_held


def test_10_rule_per_code_and_profile(workspace, monkeypatch):
    """C-6 per-code rule online (E016 -> E029 in one chunk: two triggers, a handoff each) and the C-3 gun profile: a
    code outside the profile fires the rule but only warns, no handoff, and is no fault class of a model handoff
    either; [] = no class expected; the profile survives a restart and DELETE; old state files keep /stats working."""
    monkeypatch.setenv("RSW_MODEL_PATH", str(workspace["models"] / "baseline_iforest.joblib"))
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    part = raw.iloc[:900].copy()
    part.loc[300:339, "error"] = "E016"
    part.loc[340:379, "error"] = "E029"  # a switch inside the episode, in the same 60 s chunk: a new code

    def stream(client, gun, data=part):
        out = []
        for a in range(0, len(data), 60):
            r = client.post("/predict", json={"gun_id": gun, "readings": data.iloc[a:a + 60].to_dict("records")})
            if r.status_code == 200 and r.json()["rule_triggered"]:
                out.append(r.json())
        return out

    def handoff_codes(client, gun):
        return [h["handoff"]["trigger"]["rule_code"] for h in client.get("/handoffs", params={"gun_id": gun}).json()]

    with TestClient(main.app) as client:
        d = main.app.state.detector
        assert d.rule_per_code and client.get("/model").json()["model"]["rule_per_code"] is True
        fired = stream(client, "PC")
        assert [f["rule_code"] for f in fired] == ["E016"] and fired[0]["severity"] == "critical"
        assert handoff_codes(client, "PC") == ["E016", "E029"], "the second trigger of the chunk is handed off now"
        st = client.get("/guns/PC/stats", params={"days": 31}).json()["total"]
        assert st["rule_triggers"] == 2 and st["critical_events"] == 2
        assert [e["kind"] for e in client.get("/guns/PC/trace").json()["events"]].count("rule") == 2

        assert client.get("/guns/PF/profile").status_code == 404
        r = client.put("/guns/PF/profile", json={"expected_fault_classes": ["E02", "E02"]})
        assert r.status_code == 200 and r.json() == {"expected_fault_classes": ["E02"]}
        fired = stream(client, "PF")
        assert [(f["rule_code"], f["rule_out_of_profile"]) for f in fired] == [("E016", False)]
        assert handoff_codes(client, "PF") == ["E016"], "E029 is outside the profile: no handoff"
        st = client.get("/guns/PF/stats", params={"days": 31}).json()["total"]
        assert st["rule_triggers"] == 2 and st["rule_out_of_profile"] == 1
        kinds = [e["kind"] for e in client.get("/guns/PF/trace").json()["events"]]
        assert "rule_out_of_profile" in kinds
        # an out-of-profile code that is still active is no fault class of a (model) handoff either
        body = {k: v for k, v in fired[0].items() if k != "handoff"}
        body.update(rule_triggered=False, rule_code=None, rule_class_hint=None,
                    context={**fired[0]["context"], "latest_error_code": "E029", "known_code_class_hint": "E04",
                             "known_code_in_profile": False})
        res = client.post("/handoffs/preview", json=body)
        assert res.status_code == 200 and res.json()["fault_class"] is None
        assert {g["gun_id"]: g for g in client.get("/guns").json()}["PF"]["expected_fault_classes"] == ["E02"]
        g = main.app.state.guns["PF"]
        g2 = main.GunState.restore(g.snapshot("PF", d.info.created), gun_norm=True, model_created="another model")
        assert g2.profile == ["E02"], "the profile is configuration: kept across restarts and models"
        # DELETE (re-warm after maintenance, replay.py) keeps the profile; keep_profile=false drops the gun entirely
        assert client.delete("/guns/PF").status_code == 204
        assert client.get("/guns/PF/profile").json() == {"expected_fault_classes": ["E02"]}
        assert client.delete("/guns/PF", params={"keep_profile": False}).status_code == 204
        assert client.get("/guns/PF/profile").status_code == 404

        # [] = no class expected: every terminal code only warns
        assert client.put("/guns/PE/profile", json={"expected_fault_classes": []}).json() == {"expected_fault_classes": []}
        stream(client, "PE")
        assert handoff_codes(client, "PE") == []
        assert client.get("/guns/PE/stats").json()["total"]["rule_out_of_profile"] == 2
        assert client.put("/guns/PE/profile", json={"expected_fault_classes": None}).json() == {"expected_fault_classes": None}

        # a state file written before the rule_out_of_profile counter existed: /stats still answers
        snap = main.app.state.guns["PC"].snapshot("OLD", d.info.created)
        snap["stats"] = {day: {k: v for k, v in c.items() if k != "rule_out_of_profile"} for day, c in snap["stats"].items()}
        main.app.state.guns["OLD"] = main.GunState.restore(snap, gun_norm=True, model_created=d.info.created)
        assert client.get("/guns/OLD/stats").status_code == 200 and client.get("/stats").status_code == 200


def test_11_sensor_features_per_minute(workspace, monkeypatch, caplog):
    """D-6: every scored minute yields one sensor-features-v4 document (schema-valid; all 10 measurement features sorted
    by |z| with z + raw value; on-share of the 0/1 states; setpoint mean / std raw + z; the minute's controller code +
    fault class), in the /predict response and from GET /guns/{id}/sensor-features. ML pushes it nowhere: a leftover
    RSW_RAG_SENSOR_URL only logs a warning (the push was removed, history.md 37)."""
    import jsonschema

    monkeypatch.setenv("RSW_RAG_SENSOR_URL", "http://rag.test/sensor-features")
    caplog.set_level(logging.WARNING, logger="rsw-api")
    main = load_main(workspace)
    from fastapi.testclient import TestClient

    schema = json.load(open(os.path.join(ROOT, "docs", "sensor_features.schema.json"), encoding="utf-8"))
    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    docs = []
    with TestClient(main.app) as client:
        d = main.app.state.detector
        for a in range(0, len(raw), 60):
            r = client.post("/predict", json={"gun_id": "SF", "readings": raw.iloc[a:a + 60].to_dict("records")})
            if r.status_code == 200:
                docs += r.json()["sensor_features"]
        got = client.get("/guns/SF/sensor-features", params={"minutes": 5}).json()
        assert got == docs[-5:], "GET serves the same documents"
        assert not {"rag_sensor_url", "sensor_queue", "sensor_pushed"} & set(client.get("/health").json())
    assert "RSW_RAG_SENSOR_URL is ignored" in caplog.text
    times = [x["observed_at"] for x in docs]
    assert times == sorted(set(times)) and len(docs) == len(main.app.state.guns["SF"].trace), "one per scored minute"
    for x in docs:
        jsonschema.validate(x, schema)
        zs = [abs(m["z"]) for m in x["measurements"] if m["z"] is not None]
        assert len(x["measurements"]) == 10 and zs == sorted(zs, reverse=True)
        assert x["reference_id"] == f"{d.info.name}@{d.info.created}"
    # z = window feature - reference (the trace / handoff value); value = raw mean / std of the minute's CSV rows
    p = main.app.state.guns["SF"].trace[-1]
    top = docs[-1]["measurements"][0]
    j = d.model_cols.index(top["feature"])
    assert top["z"] == round(float(p["vec"][j] - d.reference[j]), 3)
    t = pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None)
    last = raw[(t >= p["window_start"]) & (t <= p["window_end"])]
    c2 = next(m for m in docs[-1]["measurements"] if m["feature"] == "c2_mean")
    assert c2["value"] == round(float(last["c2"].mean()), 3) and docs[-1]["n_samples"] == len(last)
    # states: c10 is "on" 90 % of the synthetic rows, c4 / c18 are 1 for ~30 %
    on = {s["signal"]: [x["states"][k]["on_share"] for x in docs] for k, s in enumerate(docs[0]["states"])}
    assert 0.75 < np.mean(on["c10"]) < 1 and 0.2 < np.mean(on["c4"]) < 0.4 and 0.2 < np.mean(on["c18"]) < 0.4
    # setpoints: constant c13 -> std 0; the cap dressing block (c16 1.6 -> 0 -> 1.6, starts mid-minute at 00:25:30 and
    # ends on a minute boundary) -> mean 0 inside, std > 0 only in the minute it starts
    sp = {x["observed_at"]: {s["setpoint"]: s for s in x["setpoints"]} for x in docs}
    assert all(v["c13"]["std"] == 0 and v["c13"]["mean"] == 14.99 for v in sp.values())
    c16 = [(v["c16"]["mean"], v["c16"]["std"]) for v in sp.values()]
    assert sum(m == 0 and s_ == 0 for m, s_ in c16) >= 3 and sum(s_ > 0 for _, s_ in c16) == 1
    assert all(m["feature"][:2] != "c4" and not m["feature"].startswith("c10") for x in docs for m in x["measurements"])
    # controller codes: the long minor state (no class) and the terminal code at the end (E029 -> E04)
    assert any(x["error_code"] == "E003" and x["fault_class"] is None for x in docs)
    assert docs[-1]["error_code"] == "E029" and docs[-1]["fault_class"] == "E04"
    assert docs[0]["gun_norm"] == "warming_up" and docs[-1]["gun_norm"] == "gun"
    os.environ.pop("RSW_MODEL_PATH")


# ------------------------------------------------------------------ C-8 early warning
def stream_chunks(client, gun, data, step=60):
    """POST data in step-row chunks; the 200 responses, oldest first."""
    out = []
    for a in range(0, len(data), step):
        r = client.post("/predict", json={"gun_id": gun, "readings": data.iloc[a:a + step].to_dict("records")})
        if r.status_code == 200:
            out.append(r.json())
    return out


def test_12_ew_step_matches_b18():
    """C-8: the shared early-warning state machine (train.early_warning_flags window by window - what main.py runs)
    equals B-18's vectorised run logic: sustain consecutive condition windows of one gun without a gap."""
    from train import early_warning_flags

    rng = np.random.default_rng(3)
    t = pd.date_range("2021-09-01", periods=400, freq="60s")
    t = t.delete(np.sort(rng.choice(np.arange(10, 390), 25, replace=False)))  # gaps
    w = pd.DataFrame({"file": np.where(np.arange(len(t)) < 200, "a", "b"), "non_welding": rng.random(len(t)),
                      "error_active": 0.0, "terminal_any": 0.0,
                      "warmup": (rng.random(len(t)) < 0.1).astype(float)}, index=t)
    s = rng.random(len(t))
    g = w["file"].to_numpy()
    brk = np.r_[True, (g[1:] != g[:-1]) | (np.diff(t.asi8) != 60 * 10**9)]
    idx = np.arange(len(t))
    for kind in ("model_ng", "model"):
        c = (s >= 0.4) & (w["warmup"].to_numpy() == 0)
        if kind == "model":
            c &= w["non_welding"].to_numpy() <= 0.5
        start = c & (brk | ~np.r_[False, c[:-1]])
        pos = idx - np.maximum.accumulate(np.where(start, idx, -1))
        for k in (1, 3):
            act, ons = early_warning_flags(w, s, {"enabled": True, "kind": kind, "threshold": 0.4, "sustain": k})
            assert (act == (c & (pos >= k - 1))).all() and (ons == (c & (pos == k - 1))).all()
            assert ons.sum() >= 3, "the random condition must start several episodes"


def test_13_ew_metrics_zones():
    """C-8 event-level metrics (B-18 definitions) on a hand-built 7-day gun: T = the first own terminal code of the last
    15 min; hit 10-60 min before T; chance at T - 1..6 days; false alarms = episode starts > 60 min before T."""
    from train import early_warning_metrics

    n = 7 * 1440
    t = pd.date_range("2021-09-01", periods=n, freq="60s")
    ttf = (n - 1 - np.arange(n)) * 60.0
    term = (ttf <= 240).astype(float)
    w = pd.DataFrame({"file": "E01_0", "class": "E01", "ttf_s": ttf, "error_active": term, "terminal_any": term,
                      "non_welding": 0.0, "warmup": 0.0, "term_1": term, "term_2": 0.0, "term_3": 0.0, "term_4": 0.0},
                     index=t)
    lead = ttf - 240
    s = np.zeros(n)
    s[(lead >= 1200) & (lead <= 1800)] = 1  # in the real zone (10-60 min before T)
    s[(lead >= 86400 + 1200) & (lead <= 86400 + 1500)] = 1  # in day 1's chance zone -> also a false alarm
    s[np.argmin(np.abs(lead - 3 * 86400))] = 1  # one more false alarm, outside every zone
    m = early_warning_metrics(w, s, {"enabled": True, "kind": "model_ng", "threshold": 0.9, "sustain": 1,
                                     "cooldown_s": 3600, "rule": "model_ng_k1"})
    days = (lead > 3600).sum() / 1440
    assert m["events"] == 1 and m["recall"] == 1.0 and m["chance"] == pytest.approx(1 / 6)
    assert m["excess"] == pytest.approx(5 / 6)
    assert m["false_alarms_per_gun_day"] == pytest.approx(2 / days)
    assert m["handoff_onsets_per_gun_day"] == pytest.approx(2 / days)
    assert m["per_file"]["E01_0"]["false_alarms"] == 2


def test_14_train_bundle_early_warning(workspace):
    """C-8 in train.py: the early-warning threshold comes from gun-held-out scores (CV out-of-fold with --cv, else the
    val files), the bundle carries the block, every split gets event-level metrics; --no-early-warning turns it off."""
    import train

    m3 = json.load(open(workspace["models"] / "baseline_iforest_metrics.json", encoding="utf-8"))  # test_3: no --cv
    ew = m3["early_warning"]
    assert ew["enabled"] and ew["rule"] == train.EW_DEFAULT["rule"] and ew["quantile"] == train.EW_DEFAULT["q"]
    assert ew["calibration"]["source"] == "val_files"
    assert ew["calibration"]["fallback"] is True, "2 h files have no window a day before the failure"
    kind, k = train.EW_RULES[ew["rule"]]
    assert ew["kind"] == kind and ew["sustain"] == k and ew["cooldown_s"] == 3600 and ew["horizon_min"] == [10, 60]
    assert ew["evidence"] is None, "B-20 measured the ensemble, not the IsolationForest"
    assert 0 <= ew["val_exceed_rate"] <= 1
    for split in ("val", "test"):
        e = m3["metrics"][split]["early_warning"]
        assert {"recall", "chance", "excess", "false_alarms_per_gun_day", "handoff_onsets_per_gun_day",
                "per_file"} <= set(e) and e["threshold"] == ew["threshold"]
    scores = pd.read_csv(workspace["models"] / "scores" / "test_0_iforest.csv", index_col=0)
    assert "early_warning" in scores.columns

    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--val-frac", "0.5", "--cv", "2",
              "--no-test", "--warmup-hours", "0.5", "--model-dir", workspace["ws"] / "models_cv", "--model", "iforest")
    assert f"early warning: {train.EW_DEFAULT['rule']}" in out
    m = json.load(open(workspace["ws"] / "models_cv" / "baseline_iforest_metrics.json", encoding="utf-8"))
    cal = m["early_warning"]["calibration"]
    assert cal["source"] == "cv2_out_of_fold" and cal["n_windows"] > ew["calibration"]["n_windows"], "8 guns vs 4"
    assert m["metrics"]["cv"]["early_warning"]["events"] >= 1 and "early_warning" in m["metrics"]["val"]

    run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--val-frac", "0.5", "--no-test",
        "--warmup-hours", "0.5", "--no-early-warning", "--model-dir", workspace["ws"] / "models_noew", "--model", "iforest")
    m = json.load(open(workspace["ws"] / "models_noew" / "baseline_iforest_metrics.json", encoding="utf-8"))
    assert m["early_warning"] is None and "early_warning" not in m["metrics"]["val"]


def test_15_api_ew_matches_offline(workspace, monkeypatch):
    """C-8 online = offline: main.py's early_warning per window equals train.early_warning_flags on the same file
    (score_frame), with the threshold at the median score so that the warning switches often."""
    from train import load_bundle, score_frame

    b = load_bundle(str(workspace["models"] / "baseline_iforest.joblib"))
    pre = pd.read_parquet(workspace["pre"] / "test" / "test_0.parquet")
    thr = float(np.median(score_frame(b, pre)["score"]))
    b["early_warning"] = {**b["early_warning"], "threshold": thr}
    off = score_frame(b, pre)
    monkeypatch.setenv("RSW_MODEL_PATH", str(workspace["models"] / "baseline_iforest.joblib"))
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    with TestClient(main.app) as client:
        main.app.state.detector.early_warning["threshold"] = thr
        stream_chunks(client, "EWO", raw)
        pts = client.get("/guns/EWO/trace", params={"minutes": 180}).json()["points"]
    on = pd.DataFrame({"t": [pd.Timestamp(p["window_start"]).floor("60s") for p in pts],
                       "ew": [p["early_warning"] for p in pts], "score": [p["score"] for p in pts]}).set_index("t")
    j = on.join(off[["early_warning", "score"]], rsuffix="_off", how="inner")
    clear = ((j["score"] - thr).abs() > 1e-3) & ((j["score_off"] - thr).abs() > 1e-3)
    assert len(j) >= 80 and clear.sum() >= 60
    assert j["ew"].any() and not j["ew"].all()
    assert (j.loc[clear, "ew"] == j.loc[clear, "early_warning"]).all()


def test_16_ew_handoff_rules(workspace, monkeypatch):
    """C-8 handoffs: off by default (the response fields still say it); with RSW_EARLY_WARNING_HANDOFFS=1 one handoff
    per episode start (event_type early_warning, no code, no class, the profile as candidates), a cooldown per gun,
    none while a critical event or a terminal code is there; /handoffs?event_type= and /stats count them."""
    monkeypatch.setenv("RSW_MODEL_PATH", str(workspace["models"] / "baseline_iforest.joblib"))
    monkeypatch.delenv("RSW_EARLY_WARNING_HANDOFFS", raising=False)
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    quiet = raw.iloc[:-600]  # no terminal code
    with TestClient(main.app) as client:
        d = main.app.state.detector
        assert client.get("/health").json()["early_warning"] == {"model": True, "handoffs": False}
        d.early_warning["threshold"] = -1e9  # every window after the warm-up warns
        res = stream_chunks(client, "OFF", quiet)
        ons = [r for r in res if r["early_warning_onset"]]
        assert ons and all(r["early_warning_handoff_suppressed"] == "handoffs_off" for r in ons)
        assert client.get("/handoffs").json() == []
        assert res[-1]["early_warning"] and res[-1]["early_warning_threshold"] == -1e9
        assert not res[0]["early_warning"], "warm-up windows never warn"
        st = client.get("/guns/OFF/stats", params={"days": 31}).json()["total"]
        assert st["early_warnings"] == len(ons) and st["early_warning_handoffs"] == 0 and st["early_warning_windows"] > 0
        gun = next(x for x in client.get("/guns").json() if x["gun_id"] == "OFF")
        assert gun["early_warning"] and gun["early_warning_since"] is not None
        kinds = [e["kind"] for e in client.get("/guns/OFF/trace", params={"minutes": 180}).json()["events"]]
        assert kinds.count("early_warning_start") == len(ons)

    monkeypatch.setenv("RSW_EARLY_WARNING_HANDOFFS", "1")
    main = importlib.reload(importlib.import_module("main"))
    with TestClient(main.app) as client:
        d = main.app.state.detector
        assert client.get("/health").json()["early_warning"]["handoffs"] is True
        d.early_warning["threshold"] = -1e9
        assert client.put("/guns/ON/profile", json={"expected_fault_classes": ["E04"]}).status_code == 200
        ons = [r for r in stream_chunks(client, "ON", quiet) if r["early_warning_onset"]]
        assert len(ons) >= 2, "the warm-up end and the 90 s gap start an episode each"
        assert [r["early_warning_handoff_suppressed"] for r in ons] == [None] + ["cooldown"] * (len(ons) - 1)
        hs = client.get("/handoffs", params={"event_type": "early_warning"}).json()
        assert len(hs) == 1 and client.get("/handoffs", params={"event_type": "critical"}).json() == []
        h = hs[0]["handoff"]
        assert h["event_id"] == ons[0]["handoff"]["event_id"] and h["schema_version"] == "1.4"
        assert h["event_type"] == "early_warning" and h["trigger"]["source"] == "early_warning"
        assert h["trigger"]["rule_code"] is None and h["fault_class"] is None
        e = h["early_warning"]
        assert e["candidate_fault_classes"] == ["E04"] and e["candidate_terminal_codes"] == ["E029"]
        assert e["rule"] == d.early_warning["rule"] and e["horizon_min"] == [10, 60] and e["active_min"] >= 1
        assert h["caveats"] == main.rag_mapping.CAVEATS_EARLY_WARNING
        st = client.get("/guns/ON/stats", params={"days": 31}).json()["total"]
        assert st["early_warning_handoffs"] == 1 and st["critical_events"] == 0

        # an episode that starts in the request of a terminal code: never an early warning
        d.early_warning["cooldown_s"] = 0
        assert client.put("/guns/TC/profile", json={"expected_fault_classes": ["E01"]}).status_code == 200
        for gun in ("CR", "TC"):  # CR: the E029 rule fires (critical); TC: E029 is outside the profile (no handoff)
            d.early_warning["threshold"] = 1e9
            stream_chunks(client, gun, quiet, step=600)
            d.early_warning["threshold"] = -1e9
            last = stream_chunks(client, gun, raw.iloc[-600:], step=600)[-1]
            assert last["early_warning_onset"] and last["rule_triggered"]
            assert last["early_warning_handoff_suppressed"] == ("critical" if gun == "CR" else "terminal_code")
        assert len(client.get("/handoffs", params={"event_type": "early_warning"}).json()) == 1
    os.environ.pop("RSW_EARLY_WARNING_HANDOFFS", None)


def test_17_ew_restart_and_old_bundle(workspace, tmp_path, monkeypatch):
    """C-8 state survives a restart (the run and the last handoff; under another model only the cooldown), and a bundle
    without an early_warning block serves as before: fields False / None, no early-warning handoff even with the flag."""
    import joblib

    monkeypatch.setenv("RSW_MODEL_PATH", str(workspace["models"] / "baseline_iforest.joblib"))
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    g = main.GunState(gun_norm=True)
    g.ew = {"run": 4, "last_t": pd.Timestamp("2021-09-01 01:00"), "since": pd.Timestamp("2021-09-01 00:57")}
    g.ew_active, g.ew_last_handoff = True, pd.Timestamp("2021-09-01 00:57:59")
    snap = json.loads(json.dumps(g.snapshot("X", "m1")))
    g2 = main.GunState.restore(snap, gun_norm=True, model_created="m1")
    assert g2.ew == g.ew and g2.ew_active and g2.ew_last_handoff == g.ew_last_handoff
    g3 = main.GunState.restore(snap, gun_norm=True, model_created="m2")
    assert g3.ew["run"] == 0 and not g3.ew_active and g3.ew_last_handoff == g.ew_last_handoff

    b = joblib.load(workspace["models"] / "baseline_iforest.joblib")
    b.pop("early_warning")
    joblib.dump(b, tmp_path / "old.joblib")
    monkeypatch.setenv("RSW_MODEL_PATH", str(tmp_path / "old.joblib"))
    monkeypatch.setenv("RSW_EARLY_WARNING_HANDOFFS", "1")
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    with TestClient(main.app) as client:
        assert client.get("/health").json()["early_warning"] == {"model": False, "handoffs": True}
        assert client.get("/model").json()["model"]["early_warning_rule"] is None
        out = stream_chunks(client, "OLD", raw.iloc[:3600], step=600)
        assert out and not any(r["early_warning"] or r["early_warning_onset"] for r in out)
        assert all(r["early_warning_threshold"] is None for r in out)
        assert client.get("/handoffs", params={"event_type": "early_warning"}).json() == []
    os.environ.pop("RSW_EARLY_WARNING_HANDOFFS", None)
    os.environ.pop("RSW_MODEL_PATH", None)


# ------------------------------------------------------------------ v1.4 (spec section 10)
def v14_load(workspace, monkeypatch, ew=False):
    monkeypatch.setenv("RSW_MODEL_PATH", str(workspace["models"] / "baseline_iforest.joblib"))
    if ew:
        monkeypatch.setenv("RSW_EARLY_WARNING_HANDOFFS", "1")
    else:
        monkeypatch.delenv("RSW_EARLY_WARNING_HANDOFFS", raising=False)
    sys.path.insert(0, ROOT)
    return importlib.reload(importlib.import_module("main"))


def v14_raw(workspace):
    return pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})


def v14_t(raw):
    return pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None)


def v14_post(client, gun, part):
    r = client.post("/predict", json={"gun_id": gun, "readings": part.to_dict("records")})
    assert r.status_code == 200, r.text
    return r.json()


def v14_status(client, gun, part):
    return client.post("/predict", json={"gun_id": gun, "readings": part.to_dict("records")}).status_code


def v14_unpack(entries, block):
    """{second: value} of one channel / error, over the seconds of the window that hold a real row."""
    lo, hi = pd.Timestamp(block["window_start"]).second, pd.Timestamp(block["window_end"]).second
    changes, out, cur = dict(map(tuple, entries)), {}, None
    for s in range(lo, hi + 1):
        cur = changes.get(s, cur)
        if s not in block["missing_s"]:
            out[s] = cur
    return out


def v14_check_block(block, rows):
    """The block holds exactly the rows sent (rows: the raw CSV rows of that window)."""
    t = v14_t(rows)
    assert [s for s in range(pd.Timestamp(block["window_start"]).second, pd.Timestamp(block["window_end"]).second + 1)
            if s not in set(t.dt.second)] == block["missing_s"]
    for i in range(1, 20):
        c = f"c{i}"
        want = rows[c].map({"on": 1.0, "off": 0.0}) if c == "c10" else rows[c].astype(float)
        assert v14_unpack(block["channels"][c], block) == dict(zip(t.dt.second, want)), c
    assert v14_unpack(block["error"], block) == dict(zip(t.dt.second, rows["error"]))


def v14_filter(contribs):
    return [c for c in contribs if c["contribution"] > 0 and c["sensor"] not in ("hour_sin", "hour_cos")][:3]


def v14_rows(raw, end):
    """The raw rows of the clock minute a window end falls in."""
    t, e = v14_t(raw), pd.Timestamp(end).tz_localize(None)
    return raw[t.dt.floor("min") == e.floor("min")]


def test_v14_ew_basic(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch, ew=True)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        main.app.state.detector.early_warning["threshold"] = -1e9
        res = stream_chunks(client, "V1", raw.iloc[:-600])
    r = next(x for x in res if (x["handoff"] or {}).get("event_type") == "early_warning")
    h, b, cf = r["handoff"], r["handoff"]["sensor_features"], r["handoff"]["contributing_features"]
    assert h["schema_version"] == "1.4" and b["window_end"] == h["early_warning"]["onset_at"]
    assert set(b["channels"]) == {f"c{i}" for i in range(1, 20)}
    assert 0 < len(cf) <= 3 and all(c["contribution"] > 0 for c in cf) and not any(c["sensor"][:5] == "hour_" for c in cf)
    assert r["windows_scored"] == 1 and cf == v14_filter(r["contributing_features"])
    v14_check_block(b, v14_rows(raw, b["window_end"]))


def test_v14_block_is_what_was_received(workspace, monkeypatch):
    """A warning minute with c16 <= 0 seconds and missing seconds: the block holds the rows, not what the model saw."""
    main = v14_load(workspace, monkeypatch, ew=True)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace).copy()
    t = v14_t(raw)
    m0 = pd.Timestamp("2021-10-01 00:35:00")
    minute = t.dt.floor("min") == m0
    raw.loc[minute & t.dt.second.between(20, 29), "c16"] = 0.0
    raw.loc[minute & t.dt.second.between(30, 31), "error"] = "E003"
    keep = ~(minute & t.dt.second.between(10, 14))
    raw, t = raw[keep], t[keep]
    with TestClient(main.app) as client:
        d = main.app.state.detector
        d.early_warning["threshold"] = 1e9
        stream_chunks(client, "V2", raw[t < m0], step=600)
        d.early_warning["threshold"] = -1e9
        r = v14_post(client, "V2", raw[t.dt.floor("min") == m0])
        g, b = main.app.state.guns["V2"], r["handoff"]["sensor_features"]
        assert r["handoff"]["event_type"] == "early_warning" and b["window_end"] == "2021-10-01T00:35:59Z"
        assert b["missing_s"] == [10, 11, 12, 13, 14]
        v14_check_block(b, raw[t.dt.floor("min") == m0])
        # the model saw other numbers in the c16 = 0 seconds: the last welding values, not what was sent
        rows = pd.DataFrame(list(g.rows)).set_index("time").sort_index()
        f = d.features(rows.reset_index(), g.carry).f
        seen = f.loc["2021-10-01 00:35:20":"2021-10-01 00:35:29", "c5"] * d.scaler["c5"]["std"] + d.scaler["c5"]["mean"]
        sent = rows.loc["2021-10-01 00:35:20":"2021-10-01 00:35:29", "c5"]
        assert (seen.round(1) != sent.round(1)).any() and seen.nunique() == 1, "carried forward"


def test_v14_ew_chunk_onset_is_not_the_latest(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch, ew=True)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        d = main.app.state.detector
        d.early_warning["threshold"] = -1e9
        v14_post(client, "V3", raw.iloc[:1200])
        part = raw.iloc[1200:2380]
        r = v14_post(client, "V3", part)
        g = main.app.state.guns["V3"]
        p = next(x for x in g.trace if x["window_end"] == pd.Timestamp("2021-10-01 00:30:59"))
        feats = [c.model_dump() for c in d.score(p["vec"])[1]]
    h = r["handoff"]
    assert h["event_type"] == "early_warning" and r["windows_scored"] > 1
    assert h["sensor_features"]["window_end"] == h["early_warning"]["onset_at"] == "2021-10-01T00:30:59Z"
    v14_check_block(h["sensor_features"], v14_rows(part, "2021-10-01T00:30:59Z"))
    assert h["contributing_features"] == v14_filter(feats)
    assert h["contributing_features"] != v14_filter(r["contributing_features"])
    assert h["detected_at"] == r["window_end"] + "Z" and h["detected_at"] > h["early_warning"]["onset_at"]
    assert h["trigger"]["anomaly_score"] == h["early_warning"]["score"] == r["anomaly_score"]
    assert h["sensor_findings"] == main.rag_mapping.sensor_findings(r["contributing_features"])
    assert h["context"]["non_welding_share"] == r["context"]["non_welding_share"]
    assert all(s["points"][-1]["t"] == h["detected_at"] for s in h["sensor_trend"])


def test_v14_ew_two_onsets_take_the_first(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch, ew=True)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        main.app.state.detector.early_warning["threshold"] = -1e9
        v14_post(client, "V4", raw.iloc[:1200])
        v14_post(client, "V4", raw.iloc[1200:1450])
        r = v14_post(client, "V4", raw.iloc[1450:2450])  # the warm-up end and the 90 s gap: an episode each
        ons = [e["time"] for e in main.app.state.guns["V4"].trace_events if e["kind"] == "early_warning_start"]
    h = r["handoff"]
    assert len(ons) == 2
    assert h["sensor_features"]["window_end"] == h["early_warning"]["onset_at"] == "2021-10-01T00:30:59Z"
    assert h["early_warning"]["since"] > h["early_warning"]["onset_at"], "since: the later episode"


def test_v14_stored_v13_record_loads(workspace, monkeypatch, tmp_path):
    main = v14_load(workspace, monkeypatch)
    from datetime import datetime, timezone

    from fastapi.testclient import TestClient

    doc = json.load(open(os.path.join(ROOT, "docs", "rag_handoff.example.json"), encoding="utf-8"))
    doc = {k: v for k, v in doc.items() if k not in ("contributing_features", "sensor_features")}
    doc["schema_version"] = "1.3"  # make_examples.py rewrites this example to v1.4 (spec section 11)
    box = tmp_path / "state" / "outbox"
    box.mkdir(parents=True)
    rec = {"handoff": doc, "created_at": datetime.now(timezone.utc).isoformat(), "delivery": "pull_only"}
    (box / f"{doc['event_id']}.json").write_text(json.dumps(rec), encoding="utf-8")
    with TestClient(main.app) as client:
        got = client.get("/handoffs").json()
    assert len(got) == 1 and got[0]["handoff"]["schema_version"] == "1.3"
    assert got[0]["handoff"]["contributing_features"] == [] and got[0]["handoff"]["sensor_features"] is None


def test_v14_top_features_unit():
    from types import SimpleNamespace

    sys.path.insert(0, ROOT)
    main = importlib.import_module("main")

    def detector(pairs):
        feats = [main.FeatureContribution(feature=f"{s}_mean", sensor=s, sensor_name=s, statistic="mean", value=1.0,
                                          reference=0.0, contribution=c, share=0.1) for s, c in pairs]
        return SimpleNamespace(score=lambda vec: (0.9, feats))

    def sensors(pairs):
        return [c["sensor"] for c in main.handoff_top_features(detector(pairs), None)]

    mixed = [("hour_sin", 0.5), ("c5", 0.4), ("hour_cos", 0.3), ("c2", 0.2), ("c3", 0.0), ("c6", -0.1)]
    assert sensors(mixed) == ["c5", "c2"], "no hour_*, no contribution <= 0, fewer than 3 is fine"
    assert sensors([("c1", 0.4), ("c2", 0.3), ("c3", 0.2), ("c5", 0.1)]) == ["c1", "c2", "c3"]


def test_v14_minute_without_real_rows(workspace, monkeypatch):
    """Review focus: a clock minute that is all gap fill. The event stays, the block is empty, the pull list works."""
    main = v14_load(workspace, monkeypatch, ew=True)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    t = v14_t(raw)
    m0 = pd.Timestamp("2021-10-01 00:35:00")
    with TestClient(main.app) as client:
        d = main.app.state.detector
        d.early_warning["threshold"] = 1e9
        stream_chunks(client, "EM", raw[t < m0], step=600)
        d.early_warning["threshold"] = -1e9
        r = v14_post(client, "EM", raw[t.dt.floor("min") == m0 + pd.Timedelta(minutes=1)])
        assert client.get("/handoffs", params={"gun_id": "EM"}).status_code == 200
    b = r["handoff"]["sensor_features"]
    assert r["windows_scored"] == 2 and b["window_end"] == "2021-10-01T00:35:59Z"
    assert b["missing_s"] == list(range(60)) and b["error"] == [] and len(b["channels"]) == 19
    assert all(v == [] for v in b["channels"].values())


def test_v14_push_sends_the_block(workspace, monkeypatch):
    """Review focus: the push path writes the block as JSON arrays, integers stay integers."""
    import asyncio
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import httpx

    main = v14_load(workspace, monkeypatch)
    doc = json.load(open(os.path.join(ROOT, "docs", "rag_handoff.example.json"), encoding="utf-8"))
    doc["sensor_features"] = {"window_start": "2021-09-03T09:25:00Z", "window_end": "2021-09-03T09:25:59Z",
                              "missing_s": [3], "channels": {"c1": [[0, 1], [5, 1.5]]}, "error": [[0, "0"]]}
    ho = main.RagHandoff(**doc)
    now = datetime.now(timezone.utc)
    rec = main.HandoffRecord(handoff=ho, created_at=now, delivery="pending", next_attempt_at=now)
    seen = []

    def receive(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "ok"})

    app = SimpleNamespace(state=SimpleNamespace(
        rag_url="http://api.test/diagnose", rag_transport=httpx.MockTransport(receive), outbox=main.Outbox(None),
        rag_backoff_s=(0.05,), rag_max_attempts=3))
    asyncio.run(main.push_handoff(app, rec))
    assert rec.delivery == "delivered" and seen[0]["sensor_features"] == doc["sensor_features"]
    assert isinstance(seen[0]["sensor_features"]["channels"]["c1"][0][1], int)


def test_v14_openapi_has_the_block(workspace, monkeypatch):
    """Review focus: /docs reads this schema; the tuple types must not break it."""
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        r = client.get("/openapi.json")
    assert r.status_code == 200
    schemas = r.json()["components"]["schemas"]
    assert "HandoffSensorFeatures" in schemas
    assert {"contributing_features", "sensor_features"} <= set(schemas["RagHandoff"]["properties"])


def test_v14_critical_code_in_a_scored_minute(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    part = raw.iloc[:118].copy()
    part.iloc[70:90, part.columns.get_loc("error")] = "E029"
    with TestClient(main.app) as client:
        r = v14_post(client, "C1", part)
    h, b = r["handoff"], r["handoff"]["sensor_features"]
    assert h["event_type"] == "critical" and h["trigger"]["rule_code"] == "E029"
    assert b["window_end"] == "2021-10-01T00:01:59Z" and [10, "E029"] in b["error"]
    assert b["missing_s"] == [40, 41] and h["contributing_features"] == v14_filter(r["contributing_features"])
    v14_check_block(b, v14_rows(part, b["window_end"]))


def test_v14_critical_minute_not_finished(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    part = raw.iloc[:900].copy()
    t = pd.to_datetime(part["time"], utc=True)
    part.loc[(t >= t.iloc[-1].floor("min")).to_numpy(), "error"] = "E029"
    with TestClient(main.app) as client:
        r = v14_post(client, "C2", part)
    h, b = r["handoff"], r["handoff"]["sensor_features"]
    assert b["window_end"] == h["detected_at"] and b["window_start"] == h["window_start"]
    assert h["trigger"]["rule_trigger_time"] > b["window_end"]
    assert not any(code == "E029" for _, code in b["error"])
    assert h["contributing_features"] == v14_filter(r["contributing_features"])
    assert any(x["code"] == "E029" for x in h["error_timeline"])


def test_v14_critical_model_only_first_critical_window(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        d = main.app.state.detector
        d.threshold, d.warmup_alarms = -1e9, True
        r = v14_post(client, "C3", raw.iloc[:1170])
        res = stream_chunks(client, "C3b", raw.iloc[:1170])  # test_9's 60-row client: one window per request
    h, b = r["handoff"], r["handoff"]["sensor_features"]
    assert h["trigger"]["source"] == "model" and r["windows_scored"] == 19
    assert b["window_end"] == "2021-10-01T00:09:59Z" and h["detected_at"] > b["window_end"]
    one = next(x for x in res if x["handoff"])
    assert one["handoff"]["trigger"]["source"] == "model" and one["windows_scored"] == 1
    assert one["handoff"]["sensor_features"]["window_end"] == one["handoff"]["detected_at"] == "2021-10-01T00:09:59Z"


def test_v14_critical_chunk_code_in_the_middle(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    part = raw.iloc[:600].copy()
    part.iloc[300:340, part.columns.get_loc("error")] = "E029"
    with TestClient(main.app) as client:
        r = v14_post(client, "C4", part)
    h, b = r["handoff"], r["handoff"]["sensor_features"]
    assert b["window_end"] == "2021-10-01T00:05:59Z" and any(code == "E029" for _, code in b["error"])
    assert h["detected_at"] == r["window_end"] + "Z" and h["detected_at"] > b["window_end"]
    assert h["sensor_findings"] == main.rag_mapping.sensor_findings(r["contributing_features"])
    v14_check_block(b, v14_rows(part, b["window_end"]))


def test_v14_critical_two_codes_each_their_minute(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    part = raw.iloc[:900].copy()
    part.loc[300:339, "error"] = "E016"
    part.loc[400:439, "error"] = "E029"  # test_10 has E029 at row 340: the same clock minute as E016
    with TestClient(main.app) as client:
        r = v14_post(client, "C5", part)
        hs = [x["handoff"] for x in client.get("/handoffs", params={"gun_id": "C5"}).json()]
    assert [h["trigger"]["rule_code"] for h in hs] == ["E016", "E029"] and r["windows_scored"] > 2
    assert hs[0]["event_id"] != hs[1]["event_id"] and hs[0]["event_id"] == r["handoff"]["event_id"]
    for h in hs:
        b = h["sensor_features"]
        assert b["window_start"] <= h["trigger"]["rule_trigger_time"] <= b["window_end"]
        assert any(code == h["trigger"]["rule_code"] for _, code in b["error"])
        assert h["detected_at"] == r["window_end"] + "Z"
    assert hs[0]["sensor_features"]["window_end"] != hs[1]["sensor_features"]["window_end"]
    assert hs[0]["contributing_features"] != hs[1]["contributing_features"]


def test_v14_critical_held_code_before_the_block(workspace, monkeypatch):
    """A code seen in a 202 whose minute is never scored: the block is the first window after it."""
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        assert v14_status(client, "HC", raw.iloc[:300]) == 200
        assert v14_status(client, "HC", raw.iloc[400:430].assign(error="E029")) == 202
        r = v14_post(client, "HC", raw.iloc[430:560].assign(error="E029"))
    h, b = r["handoff"], r["handoff"]["sensor_features"]
    assert r["windows_scored"] == 2 and h["trigger"]["rule_trigger_time"] < b["window_start"]
    assert b["window_end"] == "2021-10-01T00:07:59Z" and h["detected_at"] == "2021-10-01T00:08:59Z"


def test_v14_critical_beats_an_onset_in_the_same_request(workspace, monkeypatch):
    """Review focus: an early warning that starts in the request of a code. Only the critical handoff goes out, and its
    block is the code's minute, not the onset's."""
    main = v14_load(workspace, monkeypatch, ew=True)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        d = main.app.state.detector
        d.early_warning["threshold"] = 1e9
        stream_chunks(client, "CB", raw.iloc[:-600], step=600)
        d.early_warning["threshold"] = -1e9
        last = stream_chunks(client, "CB", raw.iloc[-600:], step=600)[-1]
        hs = client.get("/handoffs", params={"gun_id": "CB"}).json()
    h, b = last["handoff"], last["handoff"]["sensor_features"]
    assert last["early_warning_onset"] and last["early_warning_handoff_suppressed"] == "critical"
    assert h["event_type"] == "critical" and h["early_warning"] is None and len(hs) == 1
    assert b["window_start"] <= h["trigger"]["rule_trigger_time"] <= b["window_end"]


def test_v14_restart_keeps_a_partial_window(workspace, monkeypatch):
    """Review focus: rows lost in a restart. The block holds what is left, in clock seconds; the lost head of the minute
    is before window_start, not in missing_s."""
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    t = v14_t(raw)
    a = int((t < pd.Timestamp("2021-10-01 00:06:20")).sum())
    b = int((t < pd.Timestamp("2021-10-01 00:07:10")).sum())
    with TestClient(main.app) as client:
        assert v14_status(client, "RS", raw.iloc[:a]) == 200
    part = raw.iloc[a + 5:b].copy()
    part.iloc[-15:, part.columns.get_loc("error")] = "E016"
    with TestClient(main.app) as client:
        r = v14_post(client, "RS", part)
    bk = r["handoff"]["sensor_features"]
    assert pd.Timestamp(bk["window_start"]).second != 0 and bk["missing_s"] == []
    assert bk["window_end"] == r["handoff"]["detected_at"]
    v14_check_block(bk, v14_rows(part, bk["window_end"]))


def test_v14_critical_window_unit():
    from types import SimpleNamespace

    sys.path.insert(0, ROOT)
    main = importlib.import_module("main")

    def verdict(minute, critical=False):
        t = pd.date_range(f"2021-10-01 00:{minute:02d}:00", periods=60, freq="s")
        return SimpleNamespace(w=pd.DataFrame(index=t), critical=critical)

    vs = [verdict(1), verdict(2), verdict(3)]
    vecs = np.array([[1.0], [2.0], [3.0]])

    def picked(verdicts, rule):
        w, vec = main.critical_window(verdicts, vecs, rule)
        return w.index[0].minute, float(vec[0])

    hit = main.RuleHit(pd.Timestamp("2021-10-01 00:02:30"), "E029", False)
    assert picked(vs, hit) == (2, 2.0), "the code in the middle window"
    assert picked(vs, main.RuleHit(pd.Timestamp("2021-10-01 00:04:30"), "E029", False)) == (3, 3.0), "after all windows"
    model = [verdict(1), verdict(2, critical=True), verdict(3, critical=True)]
    assert picked(model, None) == (2, 2.0), "no code: the first critical window"
    rep = main.RuleHit(pd.Timestamp("2021-10-01 00:01:30"), "E029", True)
    assert picked(model, rep) == (2, 2.0), "a repeat code is ignored"
    out = main.RuleHit(pd.Timestamp("2021-10-01 00:01:30"), "E029", False, True)
    assert picked(model, out) == (2, 2.0), "an out-of-profile code is ignored"
    assert picked(vs, None) == (3, 3.0) and picked(vs, rep) == (3, 3.0), "nothing applies: the latest window"
    before = main.RuleHit(pd.Timestamp("2021-10-01 00:00:30"), "E029", False)
    assert picked(vs, before) == (1, 1.0), "before all windows: the first window"
    between = [verdict(1), verdict(3)]
    gap = main.RuleHit(pd.Timestamp("2021-10-01 00:02:30"), "E029", False)
    assert main.critical_window(between, vecs[:2], gap)[0].index[0].minute == 3, "between two windows: the later one"


def test_v14_window_choice_failure_keeps_the_event(workspace, monkeypatch):
    """A rule time that cannot be compared with the window index: the latest window, and the event stays."""
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        v14_post(client, "WF", raw.iloc[:300])
        hit = main.RuleHit(pd.Timestamp("2021-10-01 00:04:00", tz="UTC"), "E029", False)
        main.app.state.guns["WF"].pending_rules.append(hit)
        r = v14_post(client, "WF", raw.iloc[300:360])
        recs = client.get("/handoffs", params={"gun_id": "WF"}).json()
    h = r["handoff"]
    assert len(recs) == 1 and h["sensor_features"]["window_end"] == h["detected_at"] and h["contributing_features"]


def v14_two_hits(client, gun, raw):
    part = raw.iloc[:900].copy()
    part.loc[300:339, "error"] = "E016"
    part.loc[400:439, "error"] = "E029"
    r = v14_post(client, gun, part)
    resp = client.get("/handoffs", params={"gun_id": gun})
    assert resp.status_code == 200, resp.text
    return r, [x["handoff"] for x in resp.json()]


def test_v14_get_handoffs_carries_the_block(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        r, hs = v14_two_hits(client, "Q", v14_raw(workspace))
        assert hs[0] == r["handoff"] and hs[0]["sensor_features"] is not None
        one = client.get(f"/handoffs/{hs[1]['event_id']}")
        assert one.status_code == 200 and one.json()["handoff"] == hs[1]
    with TestClient(main.app) as client:  # a restart reads the stored v1.4 records back unchanged
        assert [x["handoff"] for x in client.get("/handoffs", params={"gun_id": "Q"}).json()] == hs


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_v14_nan_costs_only_the_top_features(workspace, monkeypatch, bad):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    class BadValue:
        """Only what handoff_top_features sees: d.score itself must stay healthy for /predict."""

        def __init__(self, d):
            self.d = d

        def score(self, vec):
            s, feats = self.d.score(vec)
            i = next(i for i, c in enumerate(feats) if c.contribution > 0 and c.sensor[:5] != "hour_")
            feats[i] = feats[i].model_copy(update={"value": bad})
            return s, feats

    real = main.handoff_top_features

    def poisoned(d, vec):
        return real(BadValue(d), vec)

    monkeypatch.setattr(main, "handoff_top_features", poisoned)
    with TestClient(main.app) as client:
        r, hs = v14_two_hits(client, "N", v14_raw(workspace))
    assert len(hs) == 2 and all(h["contributing_features"] == [] and h["sensor_features"] for h in hs)
    assert r["contributing_features"], "the /predict list is untouched"


def test_v14_nan_without_the_check_breaks_the_pull_list(workspace, monkeypatch):
    """Section 6's reason for the check: NaN in the outbox costs every GET /handoffs."""
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    real = main.handoff_top_features

    def unchecked(d, vec):
        top = real(d, vec)
        return [{**top[0], "value": float("nan")}] + top[1:]

    monkeypatch.setattr(main, "handoff_top_features", unchecked)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        part = v14_raw(workspace).iloc[:600].copy()
        part.iloc[300:340, part.columns.get_loc("error")] = "E029"
        r = client.post("/predict", json={"gun_id": "NN", "readings": part.to_dict("records")})
        assert client.get("/handoffs").status_code == 500
        assert r.status_code == 500, "the /predict response carries it too"


@pytest.mark.parametrize("wrong", [{"window_start": 1}, {"window_start": "a", "window_end": "b", "missing_s": [],
                                                         "channels": {"c1": [[0, "text"]]}, "error": []}, None])
def test_v14_wrong_block_costs_only_the_block(workspace, monkeypatch, wrong):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    def builder(*args):
        return wrong

    monkeypatch.setattr(main.rag_mapping, "sensor_block", builder)
    with TestClient(main.app) as client:
        r, hs = v14_two_hits(client, "W", v14_raw(workspace))
    assert len(hs) == 2 and r["handoff"]["sensor_features"] is None
    assert all(h["sensor_features"] is None and len(h["contributing_features"]) > 0 for h in hs)


def test_v14_block_exception_costs_only_the_block(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    def builder(*args):
        raise RuntimeError("block")

    monkeypatch.setattr(main.rag_mapping, "sensor_block", builder)
    with TestClient(main.app) as client:
        _, hs = v14_two_hits(client, "B", v14_raw(workspace))
    assert len(hs) == 2 and all(h["sensor_features"] is None and h["contributing_features"] for h in hs)


def test_v14_context_failure_leaves_the_two_fields(workspace, monkeypatch):
    main = v14_load(workspace, monkeypatch)
    from fastapi.testclient import TestClient

    def boom(*args):
        raise RuntimeError("timeline")

    monkeypatch.setattr(main, "handoff_error_timeline", boom)
    with TestClient(main.app) as client:
        _, hs = v14_two_hits(client, "T", v14_raw(workspace))
    assert len(hs) == 2 and all(h["error_timeline"] == [] and h["sensor_trend"] == [] for h in hs)
    assert all(h["sensor_features"] and h["contributing_features"] for h in hs)


# ------------------------------------------------------------------ D-11 model loop
# (docs/superpowers/specs/2026-11-11-model-loop-design.md; the logic without data is in test_model_loop.py)
def loop_list(path, names):
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


def test_18_loop_files_from(workspace):
    """train.py --files-from: only the listed preprocessed files are used (the loop trains on the files that ended)."""
    d = workspace["ws"] / "models_loop"
    names = ["E01_0", "E01_1", "E02_0", "E02_1", "E03_0", "E03_1"]
    lst = loop_list(workspace["ws"] / "loop_train.txt", names[:3] + ["", "  "] + names[3:])  # blank lines are skipped
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--model-dir", d, "--model", "iforest",
              "--val-frac", "0.5", "--warmup-hours", "0.5", "--no-test", "--files-from", lst)
    assert "train files 3, val files 3" in out
    m = json.load(open(d / "baseline_iforest_metrics.json", encoding="utf-8"))
    assert sorted(m["train_files"] + m["val_files"]) == [f"{n}.parquet" for n in names], "no E04 file"
    assert "test" not in m["metrics"]


def test_18b_loop_files_from_missing_file(workspace):
    lst = loop_list(workspace["ws"] / "loop_missing.txt", ["E01_0", "E09_9"])
    d = workspace["ws"] / "models_loop_missing"
    r = subprocess.run([sys.executable, os.path.join(PY, "train.py"), "--data-dir", str(workspace["pre"]),
                        "--model-dir", str(d), "--model", "iforest", "--no-test", "--files-from", str(lst)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "E09_9" in r.stdout + r.stderr
    assert "E01_0" not in (r.stdout + r.stderr).split("E09_9")[0].splitlines()[-1], "only the missing name is listed"
    assert not d.exists(), "nothing was trained"


def test_18c_loop_eval_files_from(workspace, tmp_path):
    """--evaluate --eval-files-from: the saved bundle on chosen files of --data-dir (the loop's shadow files)."""
    d = workspace["ws"] / "models_loop"
    lst = loop_list(tmp_path / "shadow.txt", ["E04_0", "E04_1"])
    out = tmp_path / "eval" / "e.json"
    (tmp_path / "no_test").mkdir()  # no test split is needed for this
    run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--test-dir", tmp_path / "no_test",
        "--model-dir", d, "--model", "iforest", "--evaluate", "--eval-files-from", lst, "--eval-out", out)
    e = json.load(open(out, encoding="utf-8"))
    assert e["test"]["files"] == ["E04_0.parquet", "E04_1.parquet"]
    assert set(e["test"]["per_file"]) == {"E04_0", "E04_1"}
    assert 0 <= e["test"]["auroc_gun_mean"] <= 1 and all(0 <= v["auroc"] <= 1 for v in e["test"]["per_file"].values())
    assert e["test"]["early_warning"]["per_file"].keys() == {"E04_0", "E04_1"}
    assert not (d / "scores").exists(), "no per-file score CSVs for a loop evaluation"
    assert not (d / "baseline_iforest_test_metrics.json").exists(), "--eval-out replaces the default path"


def test_18d_loop_options_absent(workspace):
    """Without the new options --evaluate is what it was: the test split, the default JSON, the score CSVs."""
    d = workspace["ws"] / "models_loop"
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--model-dir", d, "--model", "iforest",
              "--evaluate")
    e = json.load(open(d / "baseline_iforest_test_metrics.json", encoding="utf-8"))
    assert e["test"]["files"] == ["test_0.parquet", "test_1.parquet"] and "test: AUROC" in out
    assert (d / "scores" / "test_0_iforest.csv").exists()


@pytest.fixture(scope="module")
def append_ws(workspace, tmp_path_factory):
    """A copy of the preprocessed train folder that the --split append tests may change (they run in order on it),
    and a folder for the new raw CSVs."""
    import shutil

    root = tmp_path_factory.mktemp("append")
    shutil.copytree(workspace["pre"], root / "pre")
    (root / "incoming").mkdir()
    return {"pre": root / "pre", "incoming": root / "incoming"}


def loop_append(ws):
    return run(os.path.join(PY, "preprocess.py"), "--split", "append", "--input-dir", ws["incoming"],
               "--train-out-dir", ws["pre"])


def test_19_loop_append_named_file(append_ws):
    """preprocess.py --split append: a new raw file joins the preprocessed folder with the train run's config and
    scaler - refitting the scaler would put the older bundles on another scale."""
    pre = append_ws["pre"]
    frozen = {n: (pre / n).read_bytes() for n in ("scaler.json", "preprocess_config.json")}
    old = (pre / "manifest.csv").read_text(encoding="utf-8").splitlines()
    synth_file(append_ws["incoming"] / "E01_7.csv", "E01", 31, start="2021-11-01T00:00:00Z")
    out = loop_append(append_ws)
    assert "appended 1, skipped 0, dropped 0" in out
    assert all((pre / n).read_bytes() == b for n, b in frozen.items()), "the scaler and the config stay as they were"
    new = (pre / "manifest.csv").read_text(encoding="utf-8").splitlines()
    assert new[0] == old[0] + ",source" and new[1:len(old)] == [ln + "," for ln in old[1:]], "old rows are untouched"
    man = pd.read_csv(pre / "manifest.csv")
    row = man[man["file"] == "E01_7"].iloc[0]
    assert (row["source"], row["class"], row["class_source"], row["gun"]) == ("E01_7", "E01", "filename", 7)
    assert pd.Timestamp(row["end"]) == pd.Timestamp("2021-11-01 01:59:59")
    df = pd.read_parquet(pre / "E01_7.parquet")
    assert list(df.columns) == list(pd.read_parquet(pre / "E01_0.parquet").columns)
    assert df["label"].sum() == PRE_FAILURE_S + 1, "the train run's --pre-failure-window is inherited"
    normal = df[(df["label"] == 0) & (df["error_active"] == 0)]
    assert abs(normal["c5"].mean()) < 0.5, "scaled with the train scaler"


def test_19b_loop_append_infers_class(append_ws):
    """A file without a class in its name gets one from its terminal code and the next free E0x_N name: train.py
    splits by that prefix."""
    pre = append_ws["pre"]
    synth_file(append_ws["incoming"] / "line_3.csv", "E03", 32, start="2021-11-02T00:00:00Z")
    out = loop_append(append_ws)
    assert "appended 1, skipped 1, dropped 0" in out, "E01_7 is already in the manifest"
    assert not (pre / "line_3.parquet").exists()
    df = pd.read_parquet(pre / "E03_2.parquet")
    assert set(df["file"]) == {"E03_2"} and set(df["class"]) == {"E03"} and set(df["gun"]) == {2}
    man = pd.read_csv(pre / "manifest.csv")
    row = man[man["file"] == "E03_2"].iloc[0]
    assert (row["source"], row["class_source"], row["gun"]) == ("line_3", "terminal code E028", 2)


def test_19c_loop_append_skips_no_fault(append_ws):
    """A recording that does not end at a failure has no label: it is left out (and tried again next time)."""
    pre = append_ws["pre"]
    p = append_ws["incoming"] / "line_4.csv"
    synth_file(p, "E02", 33, start="2021-11-03T00:00:00Z")
    raw = pd.read_csv(p, dtype=str)
    raw["error"] = "0"
    raw.to_csv(p, index=False)
    before = (pre / "manifest.csv").read_bytes()
    out = loop_append(append_ws)
    assert "no terminal code in the last 10 min" in out and "appended 0, skipped 3, dropped 0" in out
    assert (pre / "manifest.csv").read_bytes() == before and not list(pre.glob("*_4.parquet"))


def test_19d_loop_append_is_idempotent(append_ws):
    pre = append_ws["pre"]
    before = (pre / "manifest.csv").read_bytes()
    stamp = (pre / "E03_2.parquet").stat().st_mtime_ns
    assert "appended 0, skipped 3, dropped 0" in loop_append(append_ws)
    assert (pre / "manifest.csv").read_bytes() == before and (pre / "E03_2.parquet").stat().st_mtime_ns == stamp


def test_19e_loop_append_then_train(append_ws, tmp_path):
    """The appended files train like the others, and the loop's calendar sees them."""
    import model_loop

    pre = append_ws["pre"]
    names = ["E01_0", "E01_7", "E02_0", "E02_1", "E03_0", "E03_2", "E04_0", "E04_1"]
    run(os.path.join(PY, "train.py"), "--data-dir", pre, "--model-dir", tmp_path / "m", "--model", "iforest",
        "--val-frac", "0.5", "--warmup-hours", "0.5", "--no-test", "--files-from", loop_list(tmp_path / "t.txt", names))
    m = json.load(open(tmp_path / "m" / "baseline_iforest_metrics.json", encoding="utf-8"))
    assert sorted(m["train_files"] + m["val_files"]) == [f"{n}.parquet" for n in names]
    ends = dict(model_loop.load_manifest(str(pre)))
    assert len(ends) == 10 and ends["E03_2"] == pd.Timestamp("2021-11-02 01:59:59")


def test_19f_loop_append_refusals(append_ws):
    """What append does not take: no --input-dir, an --out-dir, a name it cannot read, a file with too many gaps."""
    pre, incoming = append_ws["pre"], append_ws["incoming"]
    exe = [sys.executable, os.path.join(PY, "preprocess.py"), "--split", "append", "--train-out-dir", str(pre)]
    r = subprocess.run(exe, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "--input-dir" in r.stderr
    r = subprocess.run(exe + ["--input-dir", str(incoming), "--out-dir", str(pre)], capture_output=True, text=True,
                       cwd=ROOT)
    assert r.returncode != 0 and "--out-dir" in r.stderr
    (incoming / "notes.csv").write_text("not a recording\n", encoding="utf-8")
    synth_file(incoming / "E02_9.csv", "E02", 34, start="2021-11-04T00:00:00Z")
    sparse = pd.read_csv(incoming / "E02_9.csv", dtype=str)
    pd.concat([sparse.iloc[::2].iloc[:-1], sparse.iloc[[-1]]]).to_csv(incoming / "E02_9.csv", index=False)  # 50 % missing
    out = loop_append(append_ws)
    assert "notes" in out and "<prefix>_<number>" in out and "appended 0, skipped 4, dropped 1" in out
    man = pd.read_csv(pre / "manifest.csv", dtype=str, keep_default_na=False)
    row = man[man["file"] == "E02_9"].iloc[0]
    assert row["dropped"] == "missing rate 51%", "every other row + the file's own gaps"
    assert row["source"] == "E02_9" and not (pre / "E02_9.parquet").exists()
    import model_loop

    assert "E02_9" not in dict(model_loop.load_manifest(str(pre)))
    assert "appended 0, skipped 5, dropped 0" in loop_append(append_ws), "a dropped file is not tried again"


@pytest.fixture(scope="module")
def loop_ws(workspace, tmp_path_factory):
    """A registry for the end-to-end loop tests (they run in order on it): small settings and a light recipe, so that
    the 8 synthetic files (one ends each day from 2021-09-01) give a first model, a challenger and one decision."""
    root = tmp_path_factory.mktemp("loop")
    recipe = root / "recipe.json"
    recipe.write_text(json.dumps(["--model", "ensemble", "--val-frac", "0.5", "--warmup-hours", "0.5",
                                  "--exclude-non-welding"]), encoding="utf-8")
    common = ["--registry", root / "registry", "--data-dir", workspace["pre"]]
    return {"reg": root / "registry", "out": root / "report", "common": common,
            "settings": ["--initial-min-files", "6", "--train-min-new", "1", "--shadow-min-files", "1",
                         "--seeds", "42,43", "--recipe-file", recipe]}


def loop_cli(*args):
    return run(os.path.join(PY, "model_loop.py"), *args)


def test_20_loop_replay_end_to_end(workspace, loop_ws):
    """model_loop.py replay with real training: first model, challenger + its noise model, shadow decision, next
    challenger - all through train.py, into a registry the server can read."""
    pytest.importorskip("lightgbm")
    import joblib

    out = loop_cli("replay", "--start", "2021-09-01", "--interval-days", "1", *loop_ws["common"], *loop_ws["settings"])
    root = loop_ws["reg"]
    reg = json.load(open(root / "registry.json", encoding="utf-8"))
    ticks = {t["at"]: t["actions"] for t in reg["ticks"]}
    assert list(ticks)[0] == "2021-09-02T00:00:00" and list(ticks)[-1] == "2021-09-09T00:00:00"
    assert ticks["2021-09-06T00:00:00"] == ["wait_first:5"] and ticks["2021-09-07T00:00:00"] == ["first:v001"]
    assert ticks["2021-09-08T00:00:00"] == ["train:v002"]
    judged, trained = ticks["2021-09-09T00:00:00"]
    assert judged in ("swap:v001>v002", "keep:v001|v002") and trained == "train:v003"
    assert "2021-09-07T00:00:00  first:v001" in out
    assert [len(v["train_files"]) for v in reg["versions"].values()] == [6, 7, 8]
    assert reg["versions"]["v002"]["train_files"][-1] == "E04_0" and reg["challenger"] == "v003"
    d, = reg["decisions"]
    assert d["shadow_files"] == ["E04_1"] and (d["champion"], d["challenger"]) == ("v001", "v002")
    assert set(d["values"]) == {"auroc_a", "auroc_b", "noise", "files_b_ge_a", "files_compared", "hits_a", "hits_b",
                                "fa_a", "fa_b"}
    assert all(isinstance(d["values"][k], float) for k in ("auroc_a", "auroc_b", "noise")), d
    assert d["values"]["files_compared"] == 1 and len(d["checks"]) == 5 and d["swap"] == all(d["checks"].values())
    assert d["swap"] == (judged == "swap:v001>v002") and reg["champion"] == ("v002" if d["swap"] else "v001")
    for v in ("v001", "v002", "v003"):
        assert reg["versions"][v]["bundle"] == f"{v}/baseline_ensemble.joblib"
        b = joblib.load(root / v / "baseline_ensemble.joblib")
        assert b["created"] == reg["versions"][v]["created"] and b["early_warning"]["enabled"]
        assert sorted(b["train_files"] + b["val_files"]) == sorted(f"{n}.parquet" for n in reg["versions"][v]["train_files"])
    assert not (root / "v001" / "seeds").exists(), "the first model is not judged: no noise model"
    seed = joblib.load(root / "v002" / "seeds" / "s43" / "baseline_ensemble.joblib")
    assert seed["args"]["seed"] == 43 and not (seed["early_warning"] or {}).get("enabled")
    assert len(list((root / "v002" / "eval").glob("*.json"))) == 1 and not (root / "registry.lock").exists()


def test_20b_loop_replay_resume_skips_training(workspace, loop_ws):
    """The same command again: nothing is trained, nothing changes. Other settings are refused."""
    pytest.importorskip("lightgbm")
    root = loop_ws["reg"]
    before = (root / "registry.json").read_bytes()
    stamps = {p: p.stat().st_mtime_ns for p in root.rglob("*.joblib")}
    out = loop_cli("replay", "--start", "2021-09-01", "--interval-days", "1", *loop_ws["common"], *loop_ws["settings"])
    assert "first:" not in out and "train:" not in out
    assert (root / "registry.json").read_bytes() == before
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*.joblib")} == stamps and len(stamps) == 5
    r = subprocess.run([sys.executable, os.path.join(PY, "model_loop.py"), "tick", "--now", "2021-09-10",
                        *map(str, loop_ws["common"])], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "initial_min_files" in r.stderr, "the default settings are not this registry's"
    assert (root / "registry.json").read_bytes() == before


def test_20c_loop_report(workspace, loop_ws):
    """report: every version on the test split (never used for a decision), the ticks and the decisions."""
    pytest.importorskip("lightgbm")
    root, out_dir = loop_ws["reg"], loop_ws["out"]
    loop_cli("report", "--out", out_dir, *loop_ws["common"])
    reg = json.load(open(root / "registry.json", encoding="utf-8"))
    rep = json.load(open(out_dir / "report.json", encoding="utf-8"))
    assert rep["config"] == reg["config"] and rep["ticks"] == reg["ticks"] and rep["decisions"] == reg["decisions"]
    assert list(rep["versions"]) == ["v001", "v002", "v003"]
    for name, v in rep["versions"].items():
        assert 0 <= v["test"]["auroc_gun_mean"] <= 1 and 0 <= v["test"]["auroc"] <= 1
        assert v["status"] == reg["versions"][name]["status"] and v["cutoff"] == reg["versions"][name]["cutoff"]
        assert sum(v["by_class"].values()) == v["n_train_files"] == len(reg["versions"][name]["train_files"])
        assert (root / name / "test_metrics.json").exists()
    assert rep["versions"]["v001"]["by_class"] == {"E01": 2, "E02": 2, "E03": 2, "E04": 0}
    fv = rep["first_vs_final"]
    assert fv["first"] == "v001" and fv["final"] == reg["champion"]
    assert fv["diff"] == pytest.approx(fv["final_auroc_gun_mean"] - fv["first_auroc_gun_mean"])
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert all(h in md for h in ("## 점검", "## 버전", "## 판정", "## 테스트")) and "v003" in md
    stamp = (root / "v001" / "test_metrics.json").stat().st_mtime_ns
    loop_cli("report", "--out", out_dir, *loop_ws["common"])
    assert (root / "v001" / "test_metrics.json").stat().st_mtime_ns == stamp, "the test scores are computed once"
    assert "v003" in loop_cli("status", "--registry", root)


LOOP_NEW = "2099-01-01T00:00:00"  # `created` of the second bundle in loop_registry


def loop_registry(workspace, tmp_path):
    """A registry as model_loop.py leaves it: the workspace IsolationForest as v001 (champion) and the same model
    under another `created` as v002 (challenger)."""
    import shutil

    import joblib
    import model_registry as mr

    reg = mr.Registry(str(tmp_path / "reg"))
    data = reg.init({"made_by": "test"})
    src = workspace["models"] / "baseline_iforest.joblib"
    bundle = joblib.load(src)
    for v in ("v001", "v002"):
        os.makedirs(reg.version_dir(v))
    shutil.copy(src, os.path.join(reg.version_dir("v001"), "b.joblib"))
    joblib.dump({**bundle, "created": LOOP_NEW}, os.path.join(reg.version_dir("v002"), "b.joblib"), compress=3)
    mr.add_version(data, "v001", cutoff="2021-09-06T00:00:00", train_files=["E01_0"], seed=42,
                   created=bundle["created"], bundle="v001/b.joblib", status="champion")
    mr.set_champion(data, "v001", at="2021-09-06T00:00:00", by="first")
    mr.add_version(data, "v002", cutoff="2021-09-13T00:00:00", train_files=["E01_0", "E01_1"], seed=42,
                   created=LOOP_NEW, bundle="v002/b.joblib", status="challenger")
    mr.set_challenger(data, "v002")
    reg.save(data)
    return reg


def loop_main(reg, monkeypatch):
    monkeypatch.setenv("RSW_MODEL_REGISTRY", reg.root)
    monkeypatch.delenv("RSW_MODEL_PATH", raising=False)
    monkeypatch.delenv("RSW_EARLY_WARNING_HANDOFFS", raising=False)
    sys.path.insert(0, ROOT)
    return importlib.reload(importlib.import_module("main"))


def loop_promote(reg, version="v002"):
    """What a swap decision of model_loop.py does to the registry."""
    import model_registry as mr

    data = reg.load()
    mr.set_champion(data, version, at="2021-09-20T00:00:00", by="decision")
    reg.save(data)


def loop_post(client, gun, part):
    return client.post("/predict", json={"gun_id": gun, "readings": part.to_dict("records")})


def test_21_loop_gate_swap_waits_for_requests(workspace):
    """ModelGate: a swap waits for the requests in flight, and goes before the requests that come after it."""
    import threading

    gate = load_main(workspace).ModelGate()
    inside, release, order = threading.Event(), threading.Event(), []

    def first_request():
        with gate.shared():
            inside.set()
            release.wait(5)
        order.append("request 1 done")

    def swap():
        with gate.exclusive():
            order.append("swap")

    def later_request():
        with gate.shared():
            order.append("request 2")

    threads = [threading.Thread(target=f) for f in (first_request, swap, later_request)]
    threads[0].start()
    assert inside.wait(5)
    threads[1].start()
    assert wait_for(lambda: gate._swaps_waiting == 1), "the swap is waiting"
    threads[2].start()
    time.sleep(0.2)
    assert order == [], "the swap waits for request 1; request 2 waits behind the swap"
    release.set()
    for t in threads:
        t.join(5)
    assert order.index("swap") < order.index("request 2") and len(order) == 3


def test_21_loop_gate_requests_run_together(workspace):
    import threading

    gate = load_main(workspace).ModelGate()
    both = threading.Barrier(2)

    def request():
        with gate.shared():
            both.wait(5)  # passes only if two requests are inside at once

    threads = [threading.Thread(target=request) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(6)
    assert not both.broken
    held, entered, release = threading.Event(), threading.Event(), threading.Event()

    def swap():
        with gate.exclusive():
            held.set()
            release.wait(5)

    def request_during_swap():
        with gate.shared():
            entered.set()

    threads = [threading.Thread(target=swap), threading.Thread(target=request_during_swap)]
    threads[0].start()
    assert held.wait(5)
    threads[1].start()
    assert not entered.wait(0.2), "no request runs while the model is being swapped"
    release.set()
    assert entered.wait(5)
    for t in threads:
        t.join(5)


def test_21a_loop_server_reads_registry(workspace, tmp_path, monkeypatch, caplog):
    """RSW_MODEL_REGISTRY: the server serves the registry's champion and shows the loop at GET /model/loop."""
    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        h = client.get("/health").json()
        assert h["model_version"] == "v001" and h["model_path"] == reg.bundle_path(reg.load(), "v001")
        loop = client.get("/model/loop").json()
        assert loop == {"registry": reg.root, "loaded": {"version": "v001", "created": h["model"]["created"]},
                        "in_sync": True,
                        "champion": {"version": "v001", "cutoff": "2021-09-06T00:00:00", "train_files": 1},
                        "challenger": {"version": "v002", "cutoff": "2021-09-13T00:00:00", "train_files": 2},
                        "last_tick": None, "decisions": []}
        assert client.get("/model/loop?limit=0").status_code == 422
        assert {"/model/reload", "/model/loop"} <= set(client.get("/openapi.json").json()["paths"])
    monkeypatch.setenv("RSW_MODEL_PATH", str(workspace["models"] / "baseline_iforest.joblib"))
    with caplog.at_level(logging.WARNING), TestClient(main.app) as client:
        assert client.get("/health").json()["model_version"] == "v001", "the registry wins"
    assert "RSW_MODEL_PATH is ignored" in caplog.text


def test_21b_loop_server_without_registry(workspace, monkeypatch):
    """Without RSW_MODEL_REGISTRY the server is what it was: the bundle of RSW_MODEL_PATH, no loop."""
    monkeypatch.delenv("RSW_MODEL_REGISTRY", raising=False)
    main = load_main(workspace)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        h = client.get("/health").json()
        assert h["model_version"] is None and h["model_path"] == str(workspace["models"] / "baseline_iforest.joblib")
        assert client.get("/model/loop").json() == {"registry": None}
        r = client.post("/model/reload")
        assert r.status_code == 409 and "RSW_MODEL_REGISTRY" in r.json()["detail"]


def test_21c_loop_reload_noop(workspace, tmp_path, monkeypatch):
    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        assert client.post("/model/reload").json() == {"changed": False, "version": "v001"}
        assert client.get("/health").json()["model_version"] == "v001"


def test_21d_loop_reload_swaps(workspace, tmp_path, monkeypatch):
    """POST /model/reload after the registry's champion changed: the new bundle serves without a restart and every
    gun is where a restart under the new bundle would have put it (warming up again; the rest carried over)."""
    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        assert stream_chunks(client, "G", raw.iloc[:2700], step=300)
        assert client.get("/guns").json()[0]["gun_norm"] == "gun", "45 min in: past the 30-min warm-up"
        old = client.get("/health").json()["model"]["created"]
        before = main.app.state.guns["G"].snapshot("G", old)
        loop_promote(reg)
        assert client.get("/model/loop").json()["in_sync"] is False, "the registry moved, the server has not yet"
        assert client.post("/model/reload").json() == {"changed": True, "version": "v002", "previous": "v001",
                                                       "created": LOOP_NEW}
        h = client.get("/health").json()
        assert (h["model_version"], h["model"]["created"]) == ("v002", LOOP_NEW)
        assert h["model_path"] == reg.bundle_path(reg.load(), "v002") and h["guns_tracked"] == 1
        assert client.get("/model").json()["model"]["created"] == LOOP_NEW
        loop = client.get("/model/loop").json()
        assert loop["in_sync"] is True and loop["loaded"] == {"version": "v002", "created": LOOP_NEW}
        g = client.get("/guns").json()[0]
        assert g["gun_norm"] == "warming_up" and g["gun_threshold"] is None
        want = main.GunState.restore(before, True, LOOP_NEW).snapshot("G", LOOP_NEW)
        got = main.app.state.guns["G"].snapshot("G", LOOP_NEW)
        assert {**got, "saved_at": None} == {**want, "saved_at": None}, "the same as a restart under the new bundle"
        saved = json.load(open(os.path.join(os.environ["RSW_STATE_DIR"], "guns", "G.json"), encoding="utf-8"))
        assert saved["model_created"] == LOOP_NEW, "the state file is the new model's"
        assert loop_post(client, "G", raw.iloc[2700:3000]).status_code in (200, 202), "the stream goes on"
        assert client.post("/model/reload").json() == {"changed": False, "version": "v002"}


def test_21e_loop_reload_keeps_stats_and_rules(workspace, tmp_path, monkeypatch):
    """What does not belong to a model survives the swap: counters, profile, the rule's cooldown."""
    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    col = raw.columns.get_loc("error")
    raw.iloc[2000:2050, col] = "E029"
    raw.iloc[2750:2800, col] = "E029"  # again 12 min later: inside the 30-min cooldown
    with TestClient(main.app) as client:
        assert client.put("/guns/G/profile", json={"expected_fault_classes": ["E04"]}).status_code == 200
        fired = [r["rule_code"] for r in stream_chunks(client, "G", raw.iloc[:2700], step=300) if r["rule_triggered"]]
        assert fired == ["E029"]
        stats = client.get("/guns/G/stats").json()
        last = dict(main.app.state.guns["G"].rule_last_by_code)
        loop_promote(reg)
        assert client.post("/model/reload").json()["changed"] is True
        assert client.get("/guns/G/stats").json() == stats
        assert client.get("/guns/G/profile").json() == {"expected_fault_classes": ["E04"]}
        assert main.app.state.guns["G"].rule_last_by_code == last
        after = stream_chunks(client, "G", raw.iloc[2700:3300], step=300)
        assert after and not any(r["rule_triggered"] for r in after), "still in the cooldown of the old trigger"


def test_21f_loop_reload_bad_bundle_keeps_old(workspace, tmp_path, monkeypatch):
    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        open(reg.bundle_path(reg.load(), "v002"), "wb").close()  # an unreadable bundle becomes the champion
        loop_promote(reg)
        r = client.post("/model/reload")
        assert r.status_code == 500 and "v002" in r.json()["detail"] and "v001" in r.json()["detail"]
        assert client.get("/health").json()["model_version"] == "v001"
        assert client.get("/model/loop").json()["in_sync"] is False
        assert stream_chunks(client, "G", raw.iloc[:600], step=300), "the old model keeps serving"


def test_21g_loop_reload_broken_registry_keeps_old_model(workspace, tmp_path, monkeypatch):
    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    with TestClient(main.app) as client:
        with open(reg.path, "w", encoding="utf-8") as fh:
            fh.write('{"schema": 1,')
        r = client.post("/model/reload")
        assert r.status_code == 500 and "registry.json" in r.json()["detail"]
        assert client.get("/model/loop").status_code == 500
        assert client.get("/health").json()["model_version"] == "v001"
        assert stream_chunks(client, "G", raw.iloc[:600], step=300)


def test_21h_loop_registry_missing_bundle(workspace, tmp_path, monkeypatch):
    """A registry the server cannot serve from stops it at start-up, with the reason."""
    import model_registry as mr
    from fastapi.testclient import TestClient

    reg = loop_registry(workspace, tmp_path)
    bundle = reg.bundle_path(reg.load(), "v001")
    os.remove(bundle)
    main = loop_main(reg, monkeypatch)
    with pytest.raises(RuntimeError) as e:
        with TestClient(main.app):
            pass
    assert bundle in str(e.value) and "v001" in str(e.value)
    empty = mr.Registry(str(tmp_path / "empty"))
    empty.init({"made_by": "test"})
    for root, word in ((empty.root, "no champion"), (str(tmp_path / "nothing"), "no registry")):
        monkeypatch.setenv("RSW_MODEL_REGISTRY", root)
        with pytest.raises(RuntimeError) as e:
            with TestClient(main.app):
                pass
        assert word in str(e.value) and root in str(e.value)
    broken = loop_registry(workspace, tmp_path / "broken")
    path = broken.bundle_path(broken.load(), "v001")
    open(path, "wb").close()  # there, but not a bundle
    monkeypatch.setenv("RSW_MODEL_REGISTRY", broken.root)
    with pytest.raises(RuntimeError) as e:
        with TestClient(main.app):
            pass
    assert path in str(e.value) and "v001" in str(e.value)


def test_21i_loop_reload_under_load(workspace, tmp_path, monkeypatch):
    """A swap in the middle of concurrent /predict streams: no request fails, none is lost, every gun ends under
    the new model."""
    import threading

    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    codes, errors, after, swapped = [], [], [], threading.Event()

    def stream(client, gun):
        """Minute after minute until 5 requests have gone out after the swap (however long the swap takes)."""
        n = 0
        for a in range(0, 6600, 60):
            was_swapped = swapped.is_set()
            try:
                codes.append(loop_post(client, gun, raw.iloc[a:a + 60]).status_code)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
            n += was_swapped
            if n >= 5:
                break
        after.append(n)

    with TestClient(main.app) as client:
        threads = [threading.Thread(target=stream, args=(client, f"L{k}")) for k in range(4)]
        for t in threads:
            t.start()
        assert wait_for(lambda: len(codes) >= 30), "the streams are running"
        loop_promote(reg)
        r = client.post("/model/reload")
        swapped.set()
        for t in threads:
            t.join(120)
        assert r.status_code == 200 and r.json()["changed"] is True
        assert after == [5, 5, 5, 5], "every stream was running at the swap and went on after it"
        assert not errors and len(codes) >= 50 and set(codes) <= {200, 202}, (errors[:2], sorted(set(codes)))
        assert client.get("/health").json()["model_version"] == "v002"
        assert sorted(main.app.state.guns) == ["L0", "L1", "L2", "L3"]
        for k in range(4):
            path = os.path.join(os.environ["RSW_STATE_DIR"], "guns", f"L{k}.json")
            assert json.load(open(path, encoding="utf-8"))["model_created"] == LOOP_NEW


def test_21j_loop_reload_keeps_pending_handoffs(workspace, tmp_path, monkeypatch):
    """Handoffs made before the swap stay in the outbox as they were (their detector is the old model)."""
    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    part = v14_raw(workspace).iloc[:600].copy()
    part.iloc[150:200, part.columns.get_loc("error")] = "E029"
    with TestClient(main.app) as client:
        old = client.get("/health").json()["model"]["created"]
        event = loop_post(client, "H", part).json()["handoff"]
        before = client.get("/handoffs").json()
        assert len(before) == 1 and before[0]["handoff"]["detector"]["created"] == old
        loop_promote(reg)
        assert client.post("/model/reload").json()["changed"] is True
        assert client.get("/handoffs").json() == before
        assert client.get(f"/handoffs/{event['event_id']}").json()["handoff"]["detector"]["created"] == old


def test_19g_loop_append_survives_a_bad_file(append_ws):
    """A file that cannot be read must not cost the others (found by the final review): it is reported and left for
    the next run, the rest is appended - before, the first bad file ended the run and every later run."""
    pre, incoming = append_ws["pre"], append_ws["incoming"]
    (incoming / "bad_1.csv").write_text("not,a,recording\n1,2,3\n", encoding="utf-8")
    synth_file(incoming / "line_20261112.csv", "E04", 35, start="2021-11-05T00:00:00Z")  # a date as the number
    synth_file(incoming / "line_7.csv", "E04", 36, start="2021-11-06T00:00:00Z")
    out = loop_append(append_ws)
    assert "skipped: KeyError" in out and "32767" in out
    assert "appended 1, skipped 7, dropped 0" in out
    man = pd.read_csv(pre / "manifest.csv", dtype=str, keep_default_na=False)
    row = man[man["source"] == "line_7"].iloc[0]
    assert row["file"] == "E04_2" and (pre / "E04_2.parquet").exists()
    assert not set(man["source"]) & {"bad_1", "line_20261112"}, "not recorded: they are tried again next time"
    assert not list(pre.glob("manifest.csv.*")), "no temp file left"


def test_21k_loop_reload_waits_for_a_request_in_flight(workspace, tmp_path, monkeypatch):
    """/predict holds the gate (found by the final review: nothing pinned it): a swap waits for the request in
    flight, that request is answered by the old model, and what it did to the gun is in the state the swap carries."""
    import threading

    reg = loop_registry(workspace, tmp_path)
    main = loop_main(reg, monkeypatch)
    from fastapi.testclient import TestClient

    raw = v14_raw(workspace)
    inside, release, answers = threading.Event(), threading.Event(), {}
    score = main.score_request

    def held(req, request, g):
        out = score(req, request, g)
        inside.set()
        release.wait(30)
        return out

    with TestClient(main.app) as client:
        old = client.get("/health").json()["model"]["created"]
        monkeypatch.setattr(main, "score_request", held)
        request = threading.Thread(target=lambda: answers.update(predict=loop_post(client, "G", raw.iloc[:300])))
        request.start()
        assert inside.wait(30), "the request is inside /predict"
        loop_promote(reg)
        reload = threading.Thread(target=lambda: answers.update(reload=client.post("/model/reload")))
        reload.start()
        reload.join(2)
        assert reload.is_alive(), "the swap waits for the request in flight"
        assert client.get("/health").json()["model_version"] == "v001"
        release.set()
        request.join(30)
        reload.join(30)
        assert answers["reload"].json() == {"changed": True, "version": "v002", "previous": "v001", "created": LOOP_NEW}
        assert answers["predict"].status_code == 200
        assert answers["predict"].json()["model"]["created"] == old, "answered by the model it started with"
        saved = json.load(open(os.path.join(os.environ["RSW_STATE_DIR"], "guns", "G.json"), encoding="utf-8"))
        assert saved["model_created"] == LOOP_NEW
        assert saved["latest"].startswith(raw["time"].iloc[299].rstrip("Z")), "the request's rows are in the state"
        assert client.get("/health").json()["model_version"] == "v002"
