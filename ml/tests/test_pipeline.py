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
    monkeypatch.delenv("RSW_RAG_URL", raising=False)


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
    assert metrics["rule"] == {"codes": ["E012", "E016", "E028", "E029"], "cooldown_s": 1800, "repeat_s": 86400}
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
        assert ho is not None and ho["schema_version"] == "1.0" and ho["gun_id"] == "G2"
        assert ho["fault_class"]["code"] == "E04" and ho["fault_class"]["basis"] == "rule_trigger"
        assert ho["trigger"]["source"] in ("rule", "model+rule") and ho["trigger"]["rule_code"] == "E029"
        assert ho["situation_ids"][0] == "S04" and "E04" in ho["summary_ko"]
        assert "hypotheses" not in ho and "search" not in ho, "causes / manuals are the ontology's job"
        recs = client.get("/handoffs", params={"gun_id": "G2"}).json()
        assert recs[-1]["handoff"]["event_id"] == ho["event_id"] and recs[-1]["delivery"] == "pull_only"
        assert client.get(f"/handoffs/{ho['event_id']}").status_code == 200
        assert client.get("/handoffs/nope").status_code == 404
        prev = client.post("/handoffs/preview", json={k: v for k, v in r.json().items() if k != "handoff"})
        assert prev.status_code == 200 and prev.json()["fault_class"]["code"] == "E04"
        assert prev.json()["situation_ids"] == ho["situation_ids"]
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
        assert rep["severity"] == "critical" if rep["sustained_in_request"] else rep["severity"] == "warning"

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

        # a switch from one terminal code to another is the same episode (train.rule_triggers on terminal_any)
        r = post(client, "SW", raw.iloc[:100].assign(error="E016"))
        assert r.json()["rule_code"] == "E016"
        r = post(client, "SW", raw.iloc[100:200].assign(error="E029"))
        assert r.json()["rule_triggered"] is False, "E016 -> E029 without a 0 in between is not a new episode"

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

