"""
RSW welding-gun anomaly detection — real-time inference API.

Loads models/baseline_<model>.joblib (written by .py/train.py), keeps a per-gun rolling
buffer of raw 1 Hz sensor readings, applies the same preprocessing as .py/preprocess.py
online, scores the latest complete window and returns a JSON document for the ontology / RAG stage.

    uvicorn main:app --reload
    RSW_MODEL_PATH=models/baseline_iforest.joblib uvicorn main:app     # the IsolationForest alone

Endpoints
    GET  /health                  liveness + model status
    GET  /model                   model card (window, threshold, features, preprocessing, val/test metrics)
    POST /predict                 append readings for one gun, score the latest window
    GET  /guns                    buffer / alarm state of every gun seen so far
    GET  /guns/{gun_id}/trace     last N minutes of scored windows (score, threshold, codes, sensor deviations) + the
                                  contributing features of the latest warning / critical window, for a cause chart
    GET  /guns/{gun_id}/trace/view  that chart as a live HTML page (static/trace.html, refreshes every 60 s)
    DELETE /guns/{gun_id}         forget a gun's buffer (e.g. after maintenance)
    GET  /handoffs                RAG handoff documents of critical events (pull; newest last)
    GET  /guns/{gun_id}/stats     daily counters of one gun (windows, alarms, rule triggers, ...) + drift warnings
    GET  /stats                   the same summed per gun, every gun
    GET  /handoffs/{event_id}     one handoff document + its delivery status
    POST /handoffs/preview        AnomalyResult JSON -> handoff document (no state; for the RAG side's tests)

ML -> RAG handoff (.py/rag_mapping.py, schema v1.0): the first request of every critical episode (and every
terminal-code rule trigger) carries `handoff` - sensor findings in words, fault class, symptoms and the situation
ids (S01..S10) the ontology looks up. Causes, checks and manual sections are the ontology's job. Every handoff
is kept in the outbox (GET /handoffs); with RSW_RAG_URL set a background worker POSTs it there, retrying with
backoff (header Idempotency-Key = event_id - the receiver drops duplicates).

State survives a restart (RSW_STATE_DIR, default ./state; "" = memory only): every gun's judgement state
(warm-up result, gun threshold, alarm run, rule cooldown / repeat history, carry) is written after each request to
<dir>/guns/<gun_id>.json, every handoff to <dir>/outbox/<event_id>.json. Not kept: the reading buffer (the first
windows after a restart have less rolling history), the trace ring, and a warm-up in progress (it restarts). A gun
whose state was written under another model (bundle `created`) goes back to warming up.

Windows are the clock-aligned `window_s` buckets train.py builds offline (00:01:00-00:01:59), scored once
complete - the latest result is up to window_s - 1 seconds behind the newest reading. A gun gets HTTP 202
"warming_up" until its first complete window with at least half the samples; the 10-minute rolling
features are exact once 600 s of history exist.

Per-gun normalisation (bundle `gun_norm`, see train.py): during the first `warmup_s` of a gun's
stream - or, with `warmup_rows`, until that many normal welding rows have come (at most `warmup_s`) -
the rows are scored with the global scaling and collected; at the end of the warm-up the
gun's mean (/std) over its normal welding rows and, optionally, its own threshold are fixed and
every later window is re-normalised with them. `AnomalyResult.gun_norm` says which regime a
result comes from. DELETE /guns/{id} restarts the warm-up (do it after maintenance).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, ".py"))  # train.py / preprocess.py live there
import train as trainlib  # noqa: E402  (also makes PCADetector importable for pca bundles)
from preprocess import BINARY_COL, ROLL, SENSOR_COLS, TERMINAL_TO_CLASS, VALUE_COLS, fill_gaps  # noqa: E402
import rag_mapping  # noqa: E402  (ML -> RAG handoff; pure Python, shared with the RAG side)

TRACE_HTML = os.path.join(PROJECT_ROOT, "static", "trace.html")
# default: the IsolationForest + LightGBM ensemble (B-9, history.md §18); models/baseline_iforest.joblib is the fallback
MODEL_PATH = os.environ.get("RSW_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "baseline_ensemble.joblib"))
ROLL_S = int(pd.Timedelta(ROLL).total_seconds())  # 10-minute rolling features, as in preprocess.py
# defaults for bundles written before preprocess_config was stored; a current bundle overrides them
DEFAULT_PREPROCESS = {"gap_fill_limit": 60, "outlier_hi": 5.0, "outlier_lo": 0.0, "resample": None}
# a request may not carry more rows than fit in the buffer next to 10 min of history: the rolling
# features of its first row need that history, and rows beyond the buffer would be dropped silently.
# Plus one window: the unfinished minute a request ends in is scored with the next one, and its first
# rows need their 10 min too (otherwise the served window differs from train.py's at every chunk seam).
MAX_CHUNK = 1200
MAX_WINDOW_S = 60  # longest bundle window the buffer is sized for
BUFFER_S = MAX_CHUNK + ROLL_S + MAX_WINDOW_S  # per-gun history kept in memory
TOP_K_FEATURES = 8
TRACE_WINDOWS = 360  # per-gun history of scored windows for GET /guns/{id}/trace (6 h at 60 s windows)
TRACE_TOP_FEATURES = 4  # sensors a trace charts by default: the focus window's top contributors
# not charted by default: time of day is not a sensor, and the error share is the code strip of the chart itself
TRACE_SKIP = {"hour_sin", "hour_cos", "error_share_10min"}
# push target for handoffs (e.g. http://127.0.0.1:8001/diagnose); unset = pull only (GET /handoffs)
RAG_URL = os.environ.get("RSW_RAG_URL")
RAG_TIMEOUT_S = float(os.environ.get("RSW_RAG_TIMEOUT_S", "10"))
OUTBOX_MAX = 1000  # handoffs kept in memory when nothing is persisted (RSW_STATE_DIR="")
STATE_DIR = os.path.join(PROJECT_ROOT, "state")  # default RSW_STATE_DIR
RAG_BACKOFF_S = (10.0, 30.0, 120.0, 600.0)  # wait before retry 1, 2, 3, 4+ (RSW_RAG_BACKOFF_S="10,30,120,600")
RAG_MAX_ATTEMPTS = 5  # then delivery="failed" (retried once more after a restart)
OUTBOX_POLL_S = 1.0  # how often the push worker looks for due handoffs
OUTBOX_RETENTION_DAYS = 30.0  # delivered / pull-only / rejected handoffs are deleted after this
# C-4 operations monitoring: daily counters per gun (data time, UTC day of the window end), kept STATS_DAYS days
STATS_DAYS = 31
STAT_KEYS = ("windows", "alarms", "held", "normal", "normal_alarms", "normal_after_warmup", "normal_alarms_after_warmup",
             "sustained_episodes", "rule_triggers", "rule_repeats", "critical_events", "rule_out_of_profile")
DRIFT_ALARM_RATE = 0.03  # normal alarm rate after the warm-up (the design point is ~1 %) ...
DRIFT_DAYS = 3  # ... on this many consecutive days ...
DRIFT_MIN_WINDOWS = 360  # ... each with at least 6 h of normal windows
DRIFT_RULE_CRITICAL = 2  # or this many critical (non-repeat) rule triggers on one day
CODE_TO_CLASS = TERMINAL_TO_CLASS  # E012 -> E01 ...
FaultClass = Literal["E01", "E02", "E03", "E04"]
SENSOR_NAME = {
    "c1": "Electrode cap offset", "c2": "Electrode force", "c3": "Electrode position",
    "c4": "Force build-up", "c5": "Balance pressure", "c6": "Friction", "c7": "Maximum aperture",
    "c8": "Maximum electrode force", "c9": "Start friction", "c10": "US2",
    "c11": "Welding point count", "c12": "Position count", "c13": "Setpoint counterbalance pressure",
    "c14": "Setpoint electrode force", "c15": "Setpoint electrode position",
    "c16": "Setpoint sheet thickness", "c17": "Setpoint velocity", "c18": "Setpoint force build-up",
    "c19": "Offset value in robot",
    "welds_delta": "Welds per second", "pos_delta": "Position moves per second",
    "welds_10min": "Welds in last 10 min", "weld_duty_10min": "Welding duty cycle (10 min)",
    "error_share_10min": "Share of time in error state (10 min)",
    "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)",
}
log = logging.getLogger("rsw-api")


# ============================================================ input schema
class SensorReading(BaseModel):
    """One 1 Hz sample from the welding controller (same columns as the dataset CSVs)."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, json_schema_extra={"example": {
        "time": "2021-09-05T02:23:20Z", "c1": 0.0, "c2": 0, "c3": 104.1, "c4": 0, "c5": 0, "c6": -93,
        "c7": 160.58, "c8": 6026, "c9": 106, "c10": "off", "c11": 4374310, "c12": 18192620,
        "c13": 14.99, "c14": 3000, "c15": 130.0, "c16": 2.6, "c17": 1, "c18": 0, "c19": 940521253,
        "error": "0"}})

    time: datetime = Field(description="sample timestamp (ISO 8601, UTC if no offset)")
    c1: float = Field(description="Electrode cap offset")
    c2: float = Field(description="Electrode force")
    c3: float = Field(description="Electrode position")
    c4: float = Field(description="Force build-up")
    c5: float = Field(description="Balance pressure")
    c6: float = Field(description="Friction")
    c7: float = Field(description="Maximum aperture (gun constant)")
    c8: float = Field(description="Maximum electrode force (gun constant)")
    c9: float = Field(description="Start friction (gun constant)")
    c10: bool = Field(description="US2 status; accepts true/false, 'on'/'off', 1/0")
    c11: float = Field(ge=0, description="Welding point count (cumulative)")
    c12: float = Field(ge=0, description="Position count (cumulative)")
    c13: float = Field(description="Setpoint counterbalance pressure")
    c14: float = Field(description="Setpoint electrode force")
    c15: float = Field(description="Setpoint electrode position")
    c16: float = Field(description="Setpoint sheet thickness; <= 0 = non-welding operation (cap dressing); the bundle may add an upper bound")
    c17: float = Field(description="Setpoint velocity")
    c18: float = Field(description="Setpoint force build-up")
    c19: float = Field(description="Offset value in robot (a time counter - not a model input)")
    error: str = Field("0", pattern=r"^(0|E\d{3})$", description="controller state code, '0' = none")

    @field_validator("c10", mode="before")
    @classmethod
    def _parse_c10(cls, v: Any) -> bool:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("on", "true", "1"):
                return True
            if s in ("off", "false", "0"):
                return False
            raise ValueError("c10 must be on/off")
        return bool(v)

    @field_validator("time", mode="after")
    @classmethod
    def _naive_utc(cls, v: datetime) -> datetime:
        # preprocess.py works in naive UTC (pd.to_datetime(utc=True).tz_localize(None)); match it. Sub-second
        # timestamps are floored to the 1 Hz grid (off-grid rows would all count as missing)
        v = v.astimezone(timezone.utc).replace(tzinfo=None) if v.tzinfo else v
        return v.replace(microsecond=0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gun_id: str = Field(min_length=1, max_length=64, description="welding gun identifier")
    readings: list[SensorReading] = Field(
        min_length=1, max_length=MAX_CHUNK,
        description=f"new samples, any order; appended to the gun's buffer. At most {MAX_CHUNK} (the {BUFFER_S}-s buffer minus "
                    "10 min of rolling history and one window); every window completed by the chunk is scored")


# ============================================================ output schema
class FeatureContribution(BaseModel):
    feature: str = Field(description="window feature name, e.g. 'c5_mean'")
    sensor: str = Field(description="underlying signal, e.g. 'c5'")
    sensor_name: str
    statistic: Literal["mean", "std"] = Field(description="window statistic of the signal")
    value: float = Field(description="feature value in the model's (z-scored) input space")
    reference: float = Field(description="typical value on normal windows")
    contribution: float = Field(description="score decrease when this feature is set to its reference; > 0 pushes toward anomaly")
    share: float = Field(ge=0, le=1, description="contribution / sum of positive contributions")


class WindowContext(BaseModel):
    latest_error_code: str
    error_active_share: float = Field(ge=0, le=1, description="share of window samples with a non-zero code")
    non_welding_share: float = Field(ge=0, le=1, description="share of window samples flagged as cap dressing / changing")
    welds_in_window: float
    welds_10min: float
    weld_duty_10min: float = Field(ge=0, le=1)
    known_code_class_hint: FaultClass | None = Field(
        description="fault class whose terminal code equals the latest error code (E012→E01, E016→E02, E028→E03, E029→E04)")
    known_code_in_profile: bool | None = Field(
        None, description="known_code_class_hint is in the gun's profile (C-3); None = no profile or no hint. False: the "
                          "handoff does not use the hint as the fault class")
    mean_dev: dict[str, float] | None = Field(
        None, description="window mean - normal reference per c-sensor (gun-centred z-score) - lets the handoff check "
                          "that a setpoint stayed at its usual level (rag_mapping steady / normal, guide §8)")


class ModelInfo(BaseModel):
    name: str
    type: str
    created: str
    window_s: int
    threshold: float = Field(description="global threshold (quantile of the training-normal scores)")
    threshold_q: float
    sustain: int
    n_features: int
    alarm_max_non_welding: float | None = Field(
        None, description="windows whose non_welding_share exceeds this never alarm (None = no gate)")
    gun_norm: Literal["none", "center", "scale"] = Field(
        "none", description="per-gun re-normalisation after the warm-up: center = subtract the gun mean, scale = also divide by its std")
    gun_warmup_s: int | None = Field(None, description="warm-up length per gun stream (with gun_warmup_rows: its cap)")
    gun_warmup_rows: int | None = Field(None, description="the warm-up ends after this many normal welding rows "
                                        "(None: fixed gun_warmup_s)")
    gun_threshold_q: float | None = Field(None, description="quantile of the warm-up scores used as the gun's threshold (None = global)")
    rule_cooldown_s: int = Field(description="after a terminal-code rule trigger the rule stays quiet this long per gun")
    rule_per_code: bool = Field(False, description="the rule cooldown is kept per terminal code and a switch from one "
                                                   "code to another is a new onset (C-6); false = one shared cooldown")
    rule_repeat_s: int | None = Field(None, description="a trigger whose code already fired this long before in the same "
                                                        "gun is a repeat: reported, not critical (None = no repeat check)")
    warmup_alarms: bool = Field(True, description="false: the model raises no alarm in a gun's warm-up (alarm_held)")
    critical_sustain: int | None = Field(None, description="windows of continuous model alarm before the model alone is "
                                                           "critical (RAG handoff); older bundles: = sustain")
    restart_hold_s: int | None = Field(None, description="no model alarm for this long after an idle block of "
                                                         "restart_idle_windows windows (alarm_held; None = off)")
    restart_idle_windows: int | None = None


# ------------------------------------------------------------ ML -> RAG handoff (rag_mapping.build_handoff)
# scope: (1) what is unusual + (2) symptom translation with situation ids S01..S10. Causes, check procedures and
# manual sections belong to the ontology / RAG stage and are NOT in this document.
class SensorFinding(BaseModel):
    sensor: str
    sensor_name: str
    sensor_name_ko: str
    group: str = Field(description="compensation | force | position | friction | electrode | io | setpoint | activity")
    statistic: Literal["mean", "std"]
    direction: Literal["high", "low", "unstable"] = Field(description="vs this gun's normal; unstable = window std up")
    deviation_z: float = Field(description="value - reference in the model's z-scored, gun-centred space")
    share: float = Field(ge=0, le=1, description="share of the anomaly score this feature explains")
    text_ko: str = Field(description="e.g. '보정(밸런스) 압력(c5) 평소보다 낮음'")


class Symptom(BaseModel):
    id: str = Field(description="symptom rule id P1..P8; P9 = rule fired with no sensor signal")
    name_ko: str
    match: Literal["full", "partial", "no_sensor_signal"]
    evidence: list[str]
    related_classes: list[FaultClass]
    situation_ids: list[str] = Field(description="keys of the MVP situation definition doc (S01..S10)")
    agrees_with_fault_class: bool
    confidence: Literal["low", "medium"] = Field(
        description="medium only when the terminal-code rule fired and agrees; never high (v1.0)")


class FaultClassInfo(BaseModel):
    code: FaultClass
    name_en: str
    name_ko: str
    terminal_code: str
    situation_id: str
    definition: str
    basis: Literal["rule_trigger", "latest_error_code"]


class HandoffTrigger(BaseModel):
    source: Literal["model", "rule", "model+rule", "none"]
    rule_code: str | None
    rule_trigger_time: str | None
    anomaly_score: float | None
    threshold: float | None
    score_z: float | None
    alarm_duration_s: int
    sustained: bool


class HandoffContext(BaseModel):
    latest_error_code: str | None
    error_active_share: float | None
    non_welding_share: float | None
    welds_in_window: float | None
    welds_10min: float | None
    alarm_held: bool
    gun_norm: str | None


class HandoffDetector(BaseModel):
    name: str | None
    created: str | None
    window_s: int | None


class RagHandoff(BaseModel):
    """What the ontology / RAG stage receives for one critical event (schema_version 1.0)."""
    schema_version: str
    event_id: str
    gun_id: str
    detected_at: str
    window_start: str
    trigger: HandoffTrigger
    summary_ko: str = Field(description="one line, e.g. '보정 압력 평소보다 낮음; 고장 유형 E01 ...로 보임'")
    sensor_findings: list[SensorFinding]
    fault_class: FaultClassInfo | None
    symptoms: list[Symptom] = Field(description="rule-agreeing first, then full sensor match; may be empty")
    situation_ids: list[str] = Field(description="situation ids to look up in the ontology, most likely first")
    context: HandoffContext
    detector: HandoffDetector
    caveats: list[str]


class HandoffRecord(BaseModel):
    handoff: RagHandoff
    created_at: datetime
    delivery: Literal["pull_only", "pending", "delivered", "failed"] = Field(
        description="pull_only: no RSW_RAG_URL; pending: queued / retrying; delivered: 2xx; failed: rejected (4xx) or "
                    "out of attempts")
    delivery_detail: str | None = None
    rag_response: dict[str, Any] | None = Field(None, description="body the RAG endpoint returned on push")
    attempts: int = Field(0, description="push attempts so far")
    last_attempt_at: datetime | None = None
    next_attempt_at: datetime | None = Field(None, description="when the worker pushes it next (pending only)")


class AnomalyResult(BaseModel):
    """Document handed to the ontology / root-cause stage."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "gun_id": "G17", "status": "ok", "window_start": "2021-09-05T02:22:21", "window_end": "2021-09-05T02:23:20",
        "n_samples": 60, "history_s": 1800, "is_anomaly": True, "anomaly_score": 0.61, "threshold": 0.54,
        "score_z": 4.6, "severity": "critical", "rule_triggered": True, "severity_source": "model+rule",
        "sustained_alarm": True, "consecutive_alarms": 3,
        "contributing_features": [{"feature": "c5_mean", "sensor": "c5", "sensor_name": "Balance pressure",
                                   "statistic": "mean", "value": -3.1, "reference": 0.02, "contribution": 0.08, "share": 0.41}],
        "context": {"latest_error_code": "E029", "error_active_share": 1.0, "non_welding_share": 0.0,
                    "welds_in_window": 0, "welds_10min": 12, "weld_duty_10min": 0.02, "known_code_class_hint": "E04"},
        "model": {"name": "baseline_iforest", "type": "iforest", "created": "2026-09-22T16:33:02", "window_s": 60,
                  "threshold": 0.54, "threshold_q": 0.99, "sustain": 3, "n_features": 46}}})

    gun_id: str
    status: Literal["ok"] = "ok"
    window_start: datetime
    window_end: datetime
    n_samples: int = Field(description="1 Hz samples in the scored window")
    history_s: int = Field(description="seconds of contiguous history behind the window; < 600 means rolling features are partial")
    is_anomaly: bool = Field(description="anomaly_score > threshold (raw model verdict, before the non-welding gate)")
    model_critical: bool = Field(False, description="the latest window's model alarm has lasted >= model.critical_sustain "
                                                    "windows (the model's share of severity critical). C-5 2026-10-06 "
                                                    "changed what makes the model critical from sustain (3) to "
                                                    "critical_sustain (10) windows; sustained_alarm keeps the 3")
    alarm_held: bool = Field(False, description="no alarm is raised or counted for this window: it is mostly non-welding "
                                                "(share > model.alarm_max_non_welding - its sensors are carried-forward "
                                                "constants), with model.warmup_alarms false in the gun's warm-up, or "
                                                "within model.restart_hold_s after an idle block (restart)")
    hold_reason: str | None = Field(None, description="why alarm_held is true")
    anomaly_score: float = Field(description="higher = more anomalous")
    threshold: float = Field(description="threshold this window was judged against: the gun's own once it has one, else the global")
    gun_norm: Literal["warming_up", "gun", "global"] = Field(
        "global", description="warming_up: global scaling while the gun statistics are being collected; "
                              "gun: window re-normalised with this gun's warm-up statistics; global: no per-gun normalisation")
    gun_threshold: float | None = Field(None, description="this gun's threshold once its warm-up is over (None otherwise)")
    score_z: float = Field(description="(score - normal mean) / normal std on training windows")
    severity: Literal["normal", "warning", "critical"] = Field(
        description="normal: below threshold or held; warning: above (also a sustained alarm shorter than "
                    "model.critical_sustain); critical: the model alarmed continuously for >= critical_sustain x window "
                    "seconds (C-5: 10 min; older bundles sustain x window), OR the terminal-code rule fired in this "
                    "request (rule_triggered)")
    rule_triggered: bool = Field(False, description="a terminal code (E012/E016/E028/E029) STARTED in this request's rows "
                                                    "(or in rows of an earlier 202 response) and that code's cooldown had "
                                                    "expired: severity is forced to critical regardless of the model, "
                                                    "unless rule_repeat or rule_out_of_profile. With model.rule_per_code "
                                                    "(C-6, 2026-10-06) a switch to another terminal code is a new trigger "
                                                    "and the cooldown is per code; before, one trigger per episode. Fires "
                                                    "on the start, not while the code persists. Several triggers in one "
                                                    "request: this response carries the first critical one, every critical "
                                                    "one gets its own handoff (GET /handoffs)")
    rule_repeat: bool = Field(False, description="the rule fired, but the same code already fired in this gun within "
                                                 "model.rule_repeat_s: severity warning, no handoff (repeats were false in "
                                                 "17 of 18 cases, history.md §11)")
    rule_out_of_profile: bool = Field(False, description="the rule fired, but the code's fault class is not in the gun's "
                                                         "profile (PUT /guns/{id}/profile): severity warning, no handoff "
                                                         "(C-3; every critical false trigger of the 80 guns was another "
                                                         "class's code)")
    rule_trigger_time: datetime | None = Field(None, description="timestamp of the row that fired the rule")
    rule_code: str | None = Field(None, description="the terminal code that fired the rule (it may have cleared by window_end)")
    rule_class_hint: FaultClass | None = Field(None, description="fault class of rule_code - use this, not "
                                                                 "context.known_code_class_hint, when rule_triggered")
    rule_code_active: bool = Field(False, description="the latest error code is a terminal code (a state, not a trigger)")
    severity_source: Literal["model", "rule", "model+rule", "none"] = Field(
        "none", description="what made severity critical: the model's critical-length alarm, the terminal-code rule, or both")
    sustained_in_request: bool = Field(False, description="some window of this request had a sustained model alarm")
    sustained_alarm: bool = Field(description="the model has alarmed continuously for >= sustain x window seconds "
                                              "(time-based: independent of how often the client calls)")
    consecutive_alarms: int = Field(description="consecutive alarming windows scored (they overlap when calls are < window apart)")
    alarm_duration_s: int = Field(0, description="seconds of continuous alarm up to window_end (0 = no alarm)")
    windows_scored: int = Field(1, description="windows scored for this request: every completed window since the previous "
                                               "request (spaced window_s apart, ending at the latest row)")
    critical_source: Literal["model", "rule", "model+rule", "none"] = Field(
        "none", description="what made this REQUEST critical (severity_source describes the latest window only: a "
                            "sustained run can start and end inside one chunk); the handoff trigger uses this")
    critical_in_request: bool = Field(False, description="some window of this request was critical (a critical-length alarm "
                                                         "or a rule trigger) - even if the latest window is not")
    contributing_features: list[FeatureContribution] = Field(description="top features by contribution, descending")
    context: WindowContext
    model: ModelInfo
    handoff: RagHandoff | None = Field(
        None, description="set on the first request of a critical episode and on every rule trigger: "
                          "the RAG handoff document (also in GET /handoffs)")
    drift_warning: list[str] = Field(default_factory=list, description="operations monitoring: why this gun no longer "
                                     "behaves as designed (e.g. normal alarm rate > 3 % for 3 days) - see GET /stats")


class WarmingUp(BaseModel):
    gun_id: str
    status: Literal["warming_up"] = "warming_up"
    n_samples: int
    samples_needed: int
    message: str


class TraceFeature(BaseModel):
    feature: str = Field(description="window feature, e.g. 'c5_mean'")
    sensor: str
    sensor_name: str
    statistic: Literal["mean", "std"]
    share: float | None = Field(None, description="its share of the focus window's anomaly contribution (None if asked for)")


class TracePoint(BaseModel):
    """One scored 60 s window of a gun's stream, as the alarm logic saw it."""
    window_start: datetime
    window_end: datetime
    score: float
    threshold: float = Field(description="the threshold this window was judged against (gun's own after its warm-up)")
    alarm: bool = Field(description="score > threshold and not held by the non-welding gate")
    held: bool
    sustained: bool = Field(description="part of an alarm run >= sustain x window (a warning-level sustained alarm)")
    critical: bool = Field(False, description="part of an alarm run >= model.critical_sustain x window (critical by the model)")
    non_welding_share: float
    error_codes: list[str] = Field(description="non-zero controller codes present in the window, in order of appearance")
    gun_norm: Literal["warming_up", "gun", "global"]
    deviation: dict[str, float] = Field(
        description="feature -> value minus the normal reference, in the model's input units (global sigma, gun-centred "
                    "after the warm-up); only the trace's `features`")


class TraceEvent(BaseModel):
    time: datetime
    kind: Literal["critical_start", "rule", "rule_repeat", "rule_out_of_profile"]
    code: str | None = None
    class_hint: FaultClass | None = None


class TraceFocus(BaseModel):
    """The window the explanation is about: the latest critical / warning window in the trace, else the latest one."""
    window_end: datetime
    anomaly_score: float
    threshold: float
    severity: Literal["normal", "warning", "critical"]
    severity_source: Literal["model", "rule", "model+rule", "none"]
    contributing_features: list[FeatureContribution]


class TraceAxes(BaseModel):
    """Fixed axis ranges for charts (auto-scaling makes the normal score noise look like an event)."""
    score: tuple[float, float]
    deviation: tuple[float, float] = (-4.0, 4.0)
    deviation_bands: tuple[float, float] = (1.0, 2.0)
    contribution_share: tuple[float, float] = (0.0, 1.0)


class Trace(BaseModel):
    """GET /guns/{id}/trace: the last `minutes` of scored windows of one gun, for a cause chart."""
    gun_id: str
    window_s: int
    minutes: int
    n_points: int
    model: ModelInfo
    focus: TraceFocus | None
    features: list[TraceFeature]
    axes: TraceAxes
    events: list[TraceEvent]
    points: list[TracePoint]


class GunProfile(BaseModel):
    """What the site knows about a gun (C-3): the fault classes it is exposed to. A terminal code of another class
    still fires the rule but only warns (rule_out_of_profile); None = no profile, every class is critical."""
    expected_fault_classes: list[FaultClass] | None = Field(
        None, description="e.g. ['E04']; None or omitted = no profile (every terminal code critical)")


class GunStatus(BaseModel):
    gun_id: str
    n_samples: int
    first_time: datetime | None
    last_time: datetime | None
    consecutive_alarms: int
    last_score: float | None
    gun_norm: Literal["warming_up", "gun", "global"] = "global"
    gun_threshold: float | None = None
    warmup_rows: int = Field(0, description="rows collected for the gun statistics so far")
    expected_fault_classes: list[FaultClass] | None = Field(None, description="the gun's profile (None = any class)")
    drift_warning: list[str] = Field(default_factory=list)


class DayStats(BaseModel):
    day: str = Field(description="UTC date of the window ends (data time)")
    windows: int = Field(0, description="scored windows (each clock minute once)")
    alarms: int = Field(0, description="windows above their threshold and not held")
    held: int = Field(0, description="windows above their threshold but held by the non-welding gate")
    normal: int = Field(0, description="windows without any controller error code")
    normal_alarms: int = 0
    normal_after_warmup: int = Field(0, description="normal windows judged after the gun's warm-up")
    normal_alarms_after_warmup: int = 0
    sustained_episodes: int = Field(0, description="model alarm runs that became sustained (>= sustain windows; critical from model.critical_sustain)")
    rule_triggers: int = Field(0, description="terminal-code rule triggers, repeats included")
    rule_repeats: int = Field(0, description="of which repeats (warning, no handoff)")
    rule_out_of_profile: int = Field(0, description="of which outside the gun's profile (warning, no handoff; C-3)")
    critical_events: int = Field(0, description="events handed to the RAG (critical episode start or rule trigger)")
    normal_alarm_rate: float | None = Field(None, description="normal_alarms / normal")
    normal_alarm_rate_after_warmup: float | None = None


class GunStats(BaseModel):
    gun_id: str
    days: list[DayStats] = Field(description="oldest first")
    total: DayStats = Field(description="the days summed (day = 'total')")
    drift_warning: list[str] = Field(default_factory=list)


# ============================================================ model + state
class Detector:
    """Wraps the joblib bundle: online preprocessing -> window vector -> score -> attributions."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.bundle: dict[str, Any] = joblib.load(path)
        b = self.bundle
        self.model = b["model"]
        self.window: int = int(b["window"])
        if self.window > MAX_WINDOW_S:
            raise RuntimeError(f"bundle window {self.window}s > {MAX_WINDOW_S}s: the per-gun buffer is sized for "
                               "10 min of history + one window - raise MAX_WINDOW_S")
        self.threshold: float = float(b["threshold"])
        self.sustain: int = int(b.get("sustain", 3))
        # C-5: the model alone is critical after critical_sustain windows; restart hold after idle blocks
        self.critical_sustain: int = int(b.get("critical_sustain") or self.sustain)
        self.restart_hold: dict[str, Any] | None = b.get("restart_hold")
        self.duty_idx: int | None = (list(b["model_cols"]).index("weld_duty_10min_mean")
                                     if "weld_duty_10min_mean" in b["model_cols"] else None)
        if self.restart_hold and self.duty_idx is None:
            raise RuntimeError("bundle restart_hold needs weld_duty_10min in the model features")
        self.feat_cols: list[str] = list(b["feature_cols"])
        self.model_cols: list[str] = list(b["model_cols"])
        scaler = b.get("scaler") or {}
        if scaler.get("scale", "global") != "global":
            raise RuntimeError(f"bundle was trained on '{scaler['scale']}'-scaled data; the API can only reproduce "
                               "the global scaler online - retrain from preprocess.py --scale global")
        self.scaler: dict[str, dict[str, float]] = scaler.get("params") or {}
        cfg = {**DEFAULT_PREPROCESS, **{k: v for k, v in (b.get("preprocess_config") or {}).items()
                                        if k in DEFAULT_PREPROCESS}}
        if cfg["resample"]:
            raise RuntimeError(f"bundle was trained on {cfg['resample']}-resampled data; the API scores 1 Hz windows "
                               "- retrain without --resample")
        if not b.get("preprocess_config"):
            log.warning("bundle has no preprocess_config (retrain with the current train.py); using defaults %s", cfg)
        self.gap_fill_limit: int = int(cfg["gap_fill_limit"])
        self.outlier_hi: float | None = None if cfg["outlier_hi"] is None else float(cfg["outlier_hi"])
        self.outlier_lo: float = float(cfg["outlier_lo"])
        g = b.get("alarm_max_non_welding")
        self.warmup_alarms: bool = bool(b.get("warmup_alarms", True))  # False: no model alarm in a gun's warm-up
        self.alarm_max_non_welding: float | None = None if g is None else float(g)
        gn = b.get("gun_norm")
        self.gun_norm: dict[str, Any] | None = dict(gn) if gn else None
        self.norm_cols: list[str] = [c for c in (gn or {}).get("columns", []) if c in self.feat_cols]
        if self.gun_norm and not self.norm_cols:
            raise RuntimeError("bundle gun_norm has no usable columns")
        tr = b.get("metrics", {}).get("train", {})
        self.score_mean: float = float(tr.get("score_mean", 0.0))
        self.score_std: float = float(tr.get("score_std", 1.0)) or 1.0
        ref = b.get("feature_reference")
        self.reference = np.asarray(ref, dtype=np.float32) if ref else np.zeros(len(self.model_cols), np.float32)
        if not ref:
            log.warning("bundle has no feature_reference (retrain with the current train.py); using zeros")
        rule = b.get("rule") or {"codes": list(trainlib.TERMINAL_CODES), "cooldown_s": trainlib.RULE_COOLDOWN_S}
        self.rule_codes: list[str] = list(rule["codes"])
        self.rule_cooldown_s: int = int(rule["cooldown_s"])
        self.rule_repeat_s: int | None = None if rule.get("repeat_s") is None else int(rule["repeat_s"])
        self.rule_per_code: bool = bool(rule.get("per_code", False))  # C-6; older bundles: one shared cooldown
        # score axis for charts: normal mass to a few sd past the threshold (an ensemble score is a CDF in [0, 1])
        lo, hi = self.score_mean - 4 * self.score_std, self.threshold + 4 * self.score_std
        self.score_axis = (0.0, 1.0) if b.get("model_type") == "ensemble" else \
            (float(np.floor(lo * 20) / 20), float(np.ceil(hi * 20) / 20))
        self.info = ModelInfo(name=os.path.splitext(os.path.basename(path))[0], type=str(b.get("model_type", "?")),
                              created=str(b.get("created", "?")), window_s=self.window, threshold=self.threshold,
                              threshold_q=float(b.get("threshold_q", float("nan"))), sustain=self.sustain,
                              n_features=len(self.model_cols), alarm_max_non_welding=self.alarm_max_non_welding,
                              gun_norm=(gn or {}).get("mode", "none"), gun_warmup_s=(gn or {}).get("warmup_s"),
                              gun_warmup_rows=(gn or {}).get("warmup_rows"),
                              gun_threshold_q=(gn or {}).get("threshold_q"), rule_cooldown_s=self.rule_cooldown_s,
                              rule_repeat_s=self.rule_repeat_s, rule_per_code=self.rule_per_code,
                              warmup_alarms=self.warmup_alarms,
                              critical_sustain=self.critical_sustain,
                              restart_hold_s=None if not self.restart_hold else int(self.restart_hold["hold_s"]),
                              restart_idle_windows=None if not self.restart_hold else int(self.restart_hold["idle_windows"]))

    # ---- per-gun normalisation, the online counterpart of train.window_file()
    def gun_normalise(self, g: "GunState", f: pd.DataFrame) -> pd.DataFrame:
        """Collect warm-up rows / fix the gun statistics when the warm-up ends / re-normalise the rows after it.
        (The threshold of a window is window_threshold's job.)"""
        gn = self.gun_norm
        if gn is None:
            return f
        if g.t0 is None:
            g.t0 = f.index[0]
        t_end = g.t0 + pd.Timedelta(seconds=int(gn["warmup_s"]))  # the cap when warmup_rows is set
        if g.norm_status == "warming_up":
            new = f[(f.index < t_end) & ((f.index > g.last_warm_time) if g.last_warm_time is not None else True)]
            target = gn.get("warmup_rows")
            if len(new) and target:
                # train.gun_norm_stats: the warm-up ends right after the target-th normal welding row
                ok = ((new["non_welding"] == 0) & (new["error_active"] == 0)).to_numpy()
                reached = np.flatnonzero(np.cumsum(ok) >= int(target) - g.warmup_ok)
                if len(reached):
                    new = new.iloc[: reached[0] + 1]
                    t_end = new.index[-1] + pd.Timedelta(seconds=1)
                g.warmup_ok += int(ok[: len(new)].sum())
            if len(new):
                g.warm_parts.append(new[self.feat_cols + ["non_welding", "error_active"]])
                g.last_warm_time, g.warmup_rows = new.index[-1], g.warmup_rows + len(new)
            if f.index[-1] >= t_end:
                self._finish_warmup(g, t_end)
        if g.norm_status != "gun":
            return f
        f = f.copy()
        sel = f.index >= g.norm["t_end"]
        if sel.any():
            f.loc[sel, self.norm_cols] = ((f.loc[sel, self.norm_cols].to_numpy(dtype="float64") - g.norm["mean"])
                                          / g.norm["std"]).astype("float32")
        return f

    def _finish_warmup(self, g: "GunState", t_end: pd.Timestamp) -> None:
        gn = self.gun_norm
        warm = pd.concat(g.warm_parts) if g.warm_parts else pd.DataFrame(columns=self.feat_cols + ["non_welding", "error_active"])
        g.warm_parts.clear()
        ok = (warm["non_welding"].to_numpy() == 0) & (warm["error_active"].to_numpy() == 0)
        if int(ok.sum()) < int(gn["min_rows"]):
            g.norm_status = "global"
            log.warning("gun warm-up ended with %d normal welding rows (< %d): keeping the global scaling", int(ok.sum()), gn["min_rows"])
            return
        x = warm.loc[ok, self.norm_cols].astype("float64")
        mean = x.mean().to_numpy()
        std = np.maximum(x.std(ddof=0).fillna(0.0).to_numpy(), float(gn["std_floor"])) if gn["mode"] == "scale" \
            else np.ones(len(self.norm_cols))
        g.norm = {"mean": mean, "std": std, "t_end": t_end}
        q = gn.get("threshold_q")
        if q is not None:
            # the warm-up windows, re-normalised with the gun statistics, calibrate the gun's threshold
            # (minute-aligned windows per contiguous segment, as train.py does offline)
            wf = warm.copy()
            wf[self.norm_cols] = ((wf[self.norm_cols].to_numpy(dtype="float64") - mean) / std).astype("float32")
            step = wf.index.to_series().diff().dt.total_seconds().fillna(1)
            wf["segment_id"] = (step > 1).cumsum()
            w = trainlib.calibration_windows(trainlib.window_features(wf, self.window, self.feat_cols),
                                             self.alarm_max_non_welding)
            if len(w):
                s = trainlib.anomaly_score(self.model, w[self.model_cols].to_numpy(dtype=np.float32))
                g.gun_threshold = trainlib.gun_threshold(s, self.threshold, float(q))
        g.norm_status = "gun"

    def in_warmup(self, g: "GunState", window_start: pd.Timestamp) -> bool:
        """The window has rows of the gun's warm-up (train.window_file's `warmup` flag). A warm-up that ends with too
        few normal welding rows leaves the gun 'global' - offline that file has no warm-up at all, so its first hours
        are held here but not in train.py (rare: 2 of 64 train guns)."""
        if self.gun_norm is None or g.norm_status == "global":
            return False
        return g.norm_status == "warming_up" or window_start < g.norm["t_end"]

    def window_threshold(self, g: "GunState", window_start: pd.Timestamp) -> float:
        """The gun's own threshold for windows entirely after its warm-up, the global one otherwise
        (train.window_thresholds: warm-up windows are judged with the global threshold)."""
        if g.norm_status == "gun" and g.gun_threshold is not None and window_start >= g.norm["t_end"]:
            return g.gun_threshold
        return self.threshold

    # ---- terminal-code rule: shared mode (older bundles) fires when a terminal-code episode starts (any code on after
    # none - a switch is the same episode, train.rule_triggers on terminal_any) and keeps quiet for rule_cooldown_s;
    # per-code mode (C-6) fires on every code start incl. a switch, with the cooldown per code (train.rule_events).
    # A trigger whose code already fired within rule_repeat_s is a repeat.
    def rule_check(self, g: "GunState", code: pd.Series) -> list[RuleHit]:
        """Scan the rows not seen yet; return every trigger, oldest first (reported one per response). Per-code mode (bundle rule.per_code, C-6; train.rule_triggers(code=)): a row whose terminal code
        differs from the previous row's is an onset, and the cooldown is kept per code."""
        new = code[code.index > g.rule_seen] if g.rule_seen is not None else code
        if new.empty:
            return []
        term = new.isin(self.rule_codes)
        if self.rule_per_code:
            onset = term & (new.astype(str) != new.astype(str).shift(fill_value=g.rule_prev_code))
        else:
            onset = term & ~term.shift(fill_value=g.rule_prev_code in self.rule_codes)
        hits = []
        for t in new.index[onset.to_numpy()]:
            c = str(new.loc[t])
            prev_t = g.rule_last_by_code.get(c)
            last = prev_t if self.rule_per_code else g.rule_last_trigger
            if last is None or (t - last).total_seconds() >= self.rule_cooldown_s:
                repeat = (self.rule_repeat_s is not None and prev_t is not None
                          and (t - prev_t).total_seconds() < self.rule_repeat_s)
                out = g.profile is not None and CODE_TO_CLASS.get(c) not in g.profile
                g.rule_last_trigger, g.rule_last_by_code[c] = t, t
                hits.append(RuleHit(t, c, repeat, out))
        g.rule_seen, g.rule_prev_code = new.index[-1], str(new.iloc[-1])
        return hits

    def windows(self, g: "GunState", f: pd.DataFrame) -> list[slice]:
        """Row slices of the windows to score, chronological - the ones train.window_features() builds offline:
        clock-aligned `window`-second buckets of each contiguous segment with at least half the samples, scored once
        the bucket is complete (the stream has reached its last second - for a segment cut by a long gap that is
        as soon as the stream resumes), newer than the previous request's last window. EVERY segment of the buffer
        counts: a request that spans a gap used to score its last segment only and lose the windows before the gap
        (33 of test_5's). Nothing new (a resend, or a request inside the same minute): the latest complete window of
        the current segment again. [] when the current segment has no complete window yet (HTTP 202)."""
        b = f.index.floor(f"{self.window}s")
        seg = f["segment"].to_numpy()
        starts = np.flatnonzero(np.r_[True, (b[1:] != b[:-1]) | (seg[1:] != seg[:-1])])
        bounds = list(zip(starts, np.r_[starts[1:], len(f)]))
        last = f.index[-1]
        done = [slice(int(a), int(e)) for a, e in bounds
                if b[a] + pd.Timedelta(seconds=self.window - 1) <= last and e - a >= max(2, self.window // 2)]
        new = [s for s in done if g.last_end is None or f.index[s.stop - 1] > g.last_end]
        # the re-score fallback stays in the current segment: a window from before a gap is not "the latest"
        return new or [s for s in done if seg[s.start] == seg[-1]][-1:]

    # ---- preprocessing identical in spirit to preprocess.py, on a rolling buffer
    def non_welding(self, c16: pd.Series) -> pd.Series:
        nw = c16 <= self.outlier_lo
        if self.outlier_hi is not None:
            nw |= c16 > self.outlier_hi
        return nw

    def is_welding_row(self, row: dict[str, Any]) -> bool:
        return not (row["c16"] <= self.outlier_lo or (self.outlier_hi is not None and row["c16"] > self.outlier_hi))

    def features(self, raw: pd.DataFrame, carry: dict[str, float] | None = None) -> Features | None:
        """Buffer rows -> Features: the feature frame of the whole buffer (global scale; column `segment` numbers the
        contiguous segments - a window never straddles a long gap) and the controller code of every buffered row.
        `carry` = sensor values of the last welding row that has left the buffer: non-welding rows at the buffer
        start take it, as preprocess.py carries the last welding value however long ago it was.
        None when nothing can be filled yet (the stream has not shown a single welding row)."""
        df = raw.set_index("time").sort_index(kind="stable")  # stable: a resent row keeps its arrival order
        df = df[~df.index.duplicated(keep="last")]
        raw_index = df.index
        df = df.resample("s").asfreq()
        present = df.index.isin(raw_index)
        code = df["error"].where(present).ffill().fillna("0").astype(str)
        x = fill_gaps(df[SENSOR_COLS].astype("float32"), present, self.gap_fill_limit)
        if len(x) == 0:
            return None
        code = code.loc[x.index]
        non_welding = self.non_welding(x["c16"])
        pressing = (x["c2"] > 0).astype("float32")  # measured, before the carry (preprocess.py)
        held_cols = VALUE_COLS + [BINARY_COL]
        x.loc[non_welding, held_cols] = np.nan
        if carry is not None and non_welding.iloc[0]:
            x.iloc[0, x.columns.get_indexer(held_cols)] = [carry[c] for c in held_cols]
        x = x.ffill().bfill()  # the carry runs across segment breaks, as in preprocess.py
        if x[held_cols].isna().any().any():
            return None
        # features over the whole buffer, exactly as preprocess.py builds them over the whole file (same float32
        # rounding): counter deltas per segment, the 10-min rolling features ACROSS long gaps; then only the last
        # contiguous segment is scored (windows never straddle a long gap)
        step = x.index.to_series().diff().dt.total_seconds().fillna(1)
        seg = (step > 1).cumsum()
        f = pd.DataFrame(index=x.index)
        f["welds_delta"] = x["c11"].groupby(seg).diff().fillna(0).clip(lower=0).astype("float32")
        f["pos_delta"] = x["c12"].groupby(seg).diff().fillna(0).clip(lower=0).astype("float32")
        f["welds_10min"] = f["welds_delta"].rolling(f"{ROLL_S}s", min_periods=1).sum().astype("float32")
        f["weld_duty_10min"] = pressing.rolling(f"{ROLL_S}s", min_periods=1).mean().astype("float32")
        error_active = (code != "0").astype("float32")
        f["error_share_10min"] = error_active.rolling(f"{ROLL_S}s", min_periods=1).mean().astype("float32")
        for c in VALUE_COLS + [BINARY_COL]:
            f[c] = x[c].astype("float32")
        hour = f.index.hour + f.index.minute / 60
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
        for c, p in self.scaler.items():
            if c in f.columns:
                f[c] = ((f[c] - p["mean"]) / p["std"]).astype("float32")
        f["error_active"], f["non_welding"], f["error_code"] = error_active, non_welding.astype("float32"), code
        f["segment"] = seg.to_numpy()
        return Features(f, code)

    def unscale(self, col: str, z: float, n: int = 1) -> float:
        """Global z-score -> raw units for a sum of n z-scored values (context fields of the result)."""
        p = self.scaler.get(col, {"mean": 0.0, "std": 1.0})
        return float(z * p["std"] + n * p["mean"])

    def window_vector(self, w: pd.DataFrame) -> np.ndarray:
        """The rows of one window -> model input, through train.window_features itself: the same groupby numerics
        to the last bit (a Series mean / std differs by ~1e-7, enough to flip a LightGBM split - B-9 ensemble)."""
        row = trainlib.window_features(w[self.feat_cols + ["non_welding"]].assign(segment_id=0), self.window,
                                       self.feat_cols, min_rows=2)
        return row.iloc[-1].reindex(self.model_cols).to_numpy(dtype=np.float32)

    def score(self, vec: np.ndarray) -> tuple[float, list[FeatureContribution]]:
        n = len(vec)
        X = np.tile(vec, (n + 1, 1))
        X[np.arange(1, n + 1), np.arange(n)] = self.reference  # row j+1: feature j set to its reference
        s = trainlib.anomaly_score(self.model, X)
        contrib = s[0] - s[1:]
        pos = float(np.clip(contrib, 0, None).sum()) or 1.0
        order = np.argsort(-contrib)[:TOP_K_FEATURES]
        feats = []
        for j in order:
            sensor, stat = split_feature(self.model_cols[j])
            feats.append(FeatureContribution(feature=self.model_cols[j], sensor=sensor, sensor_name=SENSOR_NAME.get(sensor, sensor),
                                             statistic=stat, value=float(vec[j]), reference=float(self.reference[j]),
                                             contribution=float(contrib[j]), share=float(max(contrib[j], 0) / pos)))
        return float(s[0]), feats

    def mean_dev(self, vec: np.ndarray) -> dict[str, float]:
        """Window mean - normal reference for every c-sensor (rag_mapping: setpoint / sensor at its usual level)."""
        out: dict[str, float] = {}
        for j, col in enumerate(self.model_cols):
            sensor, stat = split_feature(col)
            if stat == "mean" and sensor[:1] == "c" and sensor[1:].isdigit():
                out[sensor] = round(float(vec[j] - self.reference[j]), 3)
        return out

    # ---- one /predict request, stage by stage
    def ingest(self, g: "GunState", readings: list[SensorReading]) -> None:
        """Append readings to the gun's buffer. A welding row pushed out of the buffer becomes the carry."""
        held_cols = VALUE_COLS + [BINARY_COL]
        for r in readings:
            row = r.model_dump()
            row[BINARY_COL] = 1.0 if row[BINARY_COL] else 0.0
            if len(g.rows) == g.rows.maxlen and self.is_welding_row(g.rows[0]):
                g.carry = {c: g.rows[0][c] for c in held_cols}
            g.rows.append(row)
            g.latest = row["time"] if g.latest is None else max(g.latest, row["time"])

    def judge(self, g: "GunState", f: pd.DataFrame, wins: list[slice], scores: np.ndarray, vecs: np.ndarray,
              fresh: bool) -> list[Verdict]:
        """Chronological: advance the gun's alarm state window by window and record each new window in the trace."""
        out = []
        for s, sc, vec in zip(wins, scores, vecs):
            w = f.iloc[s]
            threshold = self.window_threshold(g, w.index[0])
            nw_share = float(w["non_welding"].mean())
            is_anomaly = bool(sc > threshold)
            hold = None
            restart = False
            if self.restart_hold and fresh:
                # train.restart_step on the window's clock label, as train.restart_flags does offline
                restart = trainlib.restart_step(g.restart, pd.Timestamp(w.index[0]).floor(f"{self.window}s"),
                                                float(vec[self.duty_idx]), nw_share, self.restart_hold, self.window)
                g.restart_held = restart
            elif self.restart_hold:
                restart = g.restart_held  # a re-scored window keeps its verdict
            if self.alarm_max_non_welding is not None and nw_share > self.alarm_max_non_welding:
                hold = f"non_welding_share {nw_share:.2f} > gate {self.alarm_max_non_welding}"
            elif not self.warmup_alarms and self.in_warmup(g, w.index[0]):
                hold = "gun warm-up: the model raises no alarm until the gun statistics are fixed (the rule still does)"
            elif restart:
                hold = (f"restart: within {int(self.restart_hold['hold_s']) // 60} min after an idle block of "
                        f">= {int(self.restart_hold['idle_windows'])} windows (the rule still fires)")
            held = hold is not None
            alarm = is_anomaly and not held
            if fresh:
                # a long gap since the last scored window starts a new alarm run (inside a segment the next window
                # starts 1 s after it). Right after a restart the buffer is empty: the first window may start up to
                # a window later without a real gap (the rows of the unfinished minute were lost)
                if g.last_end is not None and (w.index[0] - g.last_end).total_seconds() >                         self.gap_fill_limit + 1 + (self.window if g.restored else 0):
                    g.alarm_since, g.consecutive_alarms = None, 0
                g.restored = False
                g.consecutive_alarms = g.consecutive_alarms + 1 if alarm else 0
                g.alarm_since = (g.alarm_since or w.index[0]) if alarm else None
                g.last_end = w.index[-1]
            duration = int((w.index[-1] - g.alarm_since).total_seconds()) + 1 if alarm and g.alarm_since is not None else 0
            v = Verdict(w, float(sc), float(threshold), is_anomaly, held, alarm, duration,
                        duration >= self.sustain * self.window, nw_share, hold,
                        critical=duration >= self.critical_sustain * self.window)
            # trace: non-overlapping windows only, so the ring covers TRACE_WINDOWS minutes whatever the request cadence
            if fresh and (not g.trace or g.trace[-1]["window_end"] < w.index[0]):
                if v.critical and not (g.trace and g.trace[-1]["critical"]):
                    g.trace_events.append({"time": w.index[-1], "kind": "critical_start"})
                normal = not bool((w["error_active"] > 0).any())
                after = g.norm_status != "warming_up" and (g.norm is None or w.index[0] >= g.norm["t_end"])
                count(g, w.index[-1], windows=1, alarms=alarm, held=v.held and is_anomaly, normal=normal,
                      normal_alarms=normal and alarm, normal_after_warmup=normal and after,
                      normal_alarms_after_warmup=normal and after and alarm,
                      sustained_episodes=v.sustained and not g.was_sustained)
                g.was_sustained = v.sustained
                codes = w["error_code"]
                g.trace.append({"window_start": w.index[0], "window_end": w.index[-1], "score": v.score,
                                "threshold": v.threshold, "alarm": alarm, "held": v.held, "sustained": v.sustained,
                                "critical": v.critical,
                                "non_welding_share": nw_share, "gun_norm": g.norm_status,
                                "error_codes": list(dict.fromkeys(codes[codes != "0"].tolist())), "vec": vec})
            out.append(v)
        return out


def split_feature(name: str) -> tuple[str, str]:
    """'c5_mean' -> ('c5', 'mean')."""
    sensor, stat = name.rsplit("_", 1)
    return sensor, stat


@dataclass
class Features:
    f: pd.DataFrame  # the whole buffer, global scale, column `segment`
    code: pd.Series  # controller code of every buffered row


@dataclass
class RuleHit:
    time: pd.Timestamp
    code: str
    repeat: bool
    out_of_profile: bool = False  # C-3: the code's class is not in the gun's profile

    @property
    def critical(self) -> bool:
        return not self.repeat and not self.out_of_profile


@dataclass
class Verdict:
    w: pd.DataFrame
    score: float
    threshold: float
    is_anomaly: bool
    held: bool
    alarm: bool
    duration: int
    sustained: bool
    nw_share: float
    hold_reason: str | None = None
    critical: bool = False  # alarm run >= critical_sustain x window: the model alone is critical (C-5)


class GunState:
    def __init__(self, gun_norm: bool = False) -> None:
        self.rows: deque[dict[str, Any]] = deque(maxlen=BUFFER_S)
        self.latest: datetime | None = None  # newest reading time seen (time-travel guard)
        self.lock = threading.Lock()  # /predict runs in the threadpool: one request per gun at a time
        self.consecutive_alarms = 0
        self.last_score: float | None = None
        # time-based sustained alarm: start of the current alarm run, end of the last scored window
        self.alarm_since: pd.Timestamp | None = None
        self.last_end: pd.Timestamp | None = None
        # terminal-code rule: last row seen, its code, time of the last trigger (cooldown), a trigger seen while the
        # response was 202 (reported with the next scored response)
        self.rule_seen: pd.Timestamp | None = None
        self.rule_prev_code = "0"
        self.rule_last_trigger: pd.Timestamp | None = None
        self.rule_last_by_code: dict[str, pd.Timestamp] = {}  # repeat check
        # rule triggers not reported yet (a 202 response, or more than one trigger in a request): one per response,
        # oldest first
        self.pending_rules: list[RuleHit] = []
        # C-3 gun profile: fault classes this gun is exposed to (None = any); kept across restarts and models
        self.profile: list[str] | None = None
        # sensor values of the last welding row that has left the buffer: what non-welding rows at its start carry
        self.carry: dict[str, float] | None = None
        # C-5 restart hold: train.restart_step state (consecutive idle windows, start of the last restart) and the
        # verdict of the latest scored window
        self.restart: dict[str, Any] = {"idle_run": 0, "restart_at": None, "last_t": None}
        self.restart_held = False
        # per-gun normalisation: rows collected during the warm-up, then the fixed statistics
        self.norm_status: str = "warming_up" if gun_norm else "global"
        self.t0: pd.Timestamp | None = None
        self.last_warm_time: pd.Timestamp | None = None
        self.warm_parts: list[pd.DataFrame] = []
        self.warmup_rows = 0
        self.warmup_ok = 0  # normal welding rows among them (row-based warm-up)
        self.restored = False  # loaded from the state dir, no request scored since (the buffer is empty)
        # C-4: day -> counters (STAT_KEYS), the drift reasons last computed, and whether the last window was sustained
        self.stats: dict[str, dict[str, int]] = {}
        self.drift: list[str] = []
        self.was_sustained = False
        self.norm: dict[str, Any] | None = None
        self.gun_threshold: float | None = None
        # RAG handoff: one per critical episode (critical_in_request False -> True) + one per rule trigger
        self.was_critical = False
        # GET /guns/{id}/trace: every scored window (feature vector + verdict), rule / critical events, and the
        # explanation of the latest warning / critical window
        self.trace: deque[dict[str, Any]] = deque(maxlen=TRACE_WINDOWS)
        self.trace_events: deque[dict[str, Any]] = deque(maxlen=TRACE_WINDOWS)
        self.focus: dict[str, Any] | None = None

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows))

    # ---- persistence (C-1): the judgement state, not the buffer / trace / warm-up rows
    def snapshot(self, gun_id: str, model_created: str) -> dict[str, Any]:
        warm = self.norm_status == "warming_up"  # a warm-up in progress restarts after a restart
        return {"version": 1, "gun_id": gun_id, "model_created": model_created,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "latest": iso(self.latest), "consecutive_alarms": self.consecutive_alarms, "last_score": self.last_score,
                "alarm_since": iso(self.alarm_since), "last_end": iso(self.last_end), "was_critical": self.was_critical,
                "rule_seen": iso(self.rule_seen), "rule_prev_code": self.rule_prev_code,
                "rule_last_trigger": iso(self.rule_last_trigger),
                "rule_last_by_code": {c: iso(t) for c, t in self.rule_last_by_code.items()},
                "pending_rules": [{"time": iso(p.time), "code": p.code, "repeat": p.repeat,
                                   "out_of_profile": p.out_of_profile} for p in self.pending_rules],
                "profile": self.profile,
                "carry": None if self.carry is None else {c: float(v) for c, v in self.carry.items()},
                "norm_status": self.norm_status, "t0": None if warm else iso(self.t0),
                "warmup_rows": 0 if warm else self.warmup_rows, "warmup_ok": 0 if warm else self.warmup_ok,
                "norm": None if self.norm is None else {"mean": [float(x) for x in self.norm["mean"]],
                                                        "std": [float(x) for x in self.norm["std"]],
                                                        "t_end": iso(self.norm["t_end"])},
                "gun_threshold": self.gun_threshold,
                "stats": self.stats, "drift": self.drift, "was_sustained": self.was_sustained,
                "restart": {"idle_run": self.restart["idle_run"], "restart_at": iso(self.restart["restart_at"]),
                            "last_t": iso(self.restart.get("last_t"))},
                "restart_held": self.restart_held}

    @classmethod
    def restore(cls, s: dict[str, Any], gun_norm: bool, model_created: str) -> "GunState":
        """Rebuild a gun from its snapshot. Under another model the warm-up result and the alarm run are dropped
        (the gun threshold is a quantile of the old model's scores): the gun warms up again."""
        g = cls(gun_norm=gun_norm)
        g.restored = True
        latest = ts(s.get("latest"))
        g.latest = None if latest is None else latest.to_pydatetime()
        g.last_end = ts(s.get("last_end"))
        g.rule_seen, g.rule_prev_code = ts(s.get("rule_seen")), str(s.get("rule_prev_code", "0"))
        g.rule_last_trigger = ts(s.get("rule_last_trigger"))
        g.rule_last_by_code = {c: ts(t) for c, t in (s.get("rule_last_by_code") or {}).items()}
        pend = s.get("pending_rules") or ([s["pending_rule"]] if s.get("pending_rule") else [])  # older state files
        g.pending_rules = [RuleHit(ts(p["time"]), str(p["code"]), bool(p["repeat"]), bool(p.get("out_of_profile", False)))
                           for p in pend]
        g.profile = s.get("profile")
        g.carry = s.get("carry")
        r = s.get("restart") or {}
        g.restart = {"idle_run": int(r.get("idle_run", 0)), "restart_at": ts(r.get("restart_at")),
                     "last_t": ts(r.get("last_t"))}
        g.restart_held = bool(s.get("restart_held"))
        # every STAT_KEYS counter, also on days saved before a counter existed (gun_stats sums them all)
        g.stats = {d: {**dict.fromkeys(STAT_KEYS, 0), **{k: int(v) for k, v in c.items()}}
                   for d, c in (s.get("stats") or {}).items()}
        g.drift = list(s.get("drift") or [])
        if s.get("model_created") != model_created:
            return g
        g.was_sustained = bool(s.get("was_sustained"))
        g.consecutive_alarms, g.last_score = int(s.get("consecutive_alarms", 0)), s.get("last_score")
        g.alarm_since, g.was_critical = ts(s.get("alarm_since")), bool(s.get("was_critical"))
        if gun_norm and s.get("norm_status") in ("gun", "global"):
            g.norm_status, g.t0 = s["norm_status"], ts(s.get("t0"))
            g.warmup_rows, g.warmup_ok = int(s.get("warmup_rows", 0)), int(s.get("warmup_ok", 0))
            n = s.get("norm")
            if n is not None:
                g.norm = {"mean": np.asarray(n["mean"], dtype="float64"), "std": np.asarray(n["std"], dtype="float64"),
                          "t_end": ts(n["t_end"])}
            g.gun_threshold = s.get("gun_threshold")
        return g


def count(g: GunState, t: pd.Timestamp, **inc: int) -> None:
    """Add to the gun's counters of the day of t; days beyond STATS_DAYS are dropped."""
    day = pd.Timestamp(t).strftime("%Y-%m-%d")
    c = g.stats.setdefault(day, dict.fromkeys(STAT_KEYS, 0))
    for k, v in inc.items():
        c[k] = c.get(k, 0) + int(v)  # .get: days saved before a counter existed
    if len(g.stats) > STATS_DAYS:
        for old in sorted(g.stats)[: len(g.stats) - STATS_DAYS]:
            del g.stats[old]


def drift_reasons(g: GunState) -> list[str]:
    """Normal alarm rate after the warm-up above DRIFT_ALARM_RATE on the last DRIFT_DAYS consecutive days (each with
    enough normal windows), or DRIFT_RULE_CRITICAL+ critical rule triggers on the latest day."""
    out = []
    days = sorted(g.stats)
    last = days[-DRIFT_DAYS:]
    consecutive = len(last) == DRIFT_DAYS and (pd.Timestamp(last[-1]) - pd.Timestamp(last[0])).days == DRIFT_DAYS - 1
    if consecutive:
        rates = [g.stats[d]["normal_alarms_after_warmup"] / g.stats[d]["normal_after_warmup"]
                 for d in last if g.stats[d]["normal_after_warmup"] >= DRIFT_MIN_WINDOWS]
        if len(rates) == DRIFT_DAYS and min(rates) > DRIFT_ALARM_RATE:
            out.append(f"normal alarm rate after warm-up {', '.join(f'{r:.1%}' for r in rates)} > {DRIFT_ALARM_RATE:.0%} "
                       f"on {DRIFT_DAYS} consecutive days ({last[0]}..{last[-1]}): re-warm the gun (DELETE) or check it")
    if days:
        c = g.stats[days[-1]]
        crit = c["rule_triggers"] - c["rule_repeats"] - c.get("rule_out_of_profile", 0)
        if crit >= DRIFT_RULE_CRITICAL:
            out.append(f"{crit} critical terminal-code rule triggers on {days[-1]}")
    return out


def day_stats(day: str, c: dict[str, int]) -> DayStats:
    c = {**dict.fromkeys(STAT_KEYS, 0), **c}
    return DayStats(day=day, **c, normal_alarm_rate=c["normal_alarms"] / c["normal"] if c["normal"] else None,
                    normal_alarm_rate_after_warmup=c["normal_alarms_after_warmup"] / c["normal_after_warmup"]
                    if c["normal_after_warmup"] else None)


def gun_stats(gun_id: str, g: GunState, days: int | None = None) -> GunStats:
    keep = sorted(g.stats)[-days:] if days else sorted(g.stats)
    total = {k: sum(g.stats[d].get(k, 0) for d in keep) for k in STAT_KEYS}
    return GunStats(gun_id=gun_id, days=[day_stats(d, g.stats[d]) for d in keep], total=day_stats("total", total),
                    drift_warning=g.drift)


def iso(t: Any) -> str | None:
    return None if t is None else pd.Timestamp(t).isoformat()


def ts(v: str | None) -> pd.Timestamp | None:
    return None if v is None else pd.Timestamp(v)


def write_json_atomic(path: str, obj: Any) -> None:
    """Temp file + rename: a crash mid-write leaves the previous version, never half a file."""
    tmp = f"{path}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, default=str)
    for attempt in range(5):  # Windows: a reader (or a virus scanner) holding the target makes the rename fail briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02 * (attempt + 1))


def read_json_dir(path: str) -> list[tuple[str, Any]]:
    """(file, content) of every *.json in path; unreadable files are logged and skipped."""
    out = []
    for name in sorted(os.listdir(path)) if os.path.isdir(path) else []:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(path, name), encoding="utf-8") as fh:
                out.append((name, json.load(fh)))
        except (OSError, ValueError):
            log.exception("unreadable state file %s", os.path.join(path, name))
    return out


class GunStore:
    """<state dir>/guns/<gun_id>.json per gun (C-1). root None = nothing persisted."""

    def __init__(self, root: str | None, model_created: str) -> None:
        self.dir = os.path.join(root, "guns") if root else None
        self.model_created = model_created
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    def path(self, gun_id: str) -> str:
        return os.path.join(self.dir, quote(gun_id, safe="-_.") + ".json")

    def save(self, gun_id: str, g: GunState) -> None:
        if self.dir:
            try:
                write_json_atomic(self.path(gun_id), g.snapshot(gun_id, self.model_created))
            except OSError:  # never fail a scored response over the state file; the next request writes it again
                log.exception("could not save the state of gun %s", gun_id)

    def delete(self, gun_id: str) -> None:
        if self.dir and os.path.exists(self.path(gun_id)):
            os.remove(self.path(gun_id))

    def load(self, gun_norm: bool) -> dict[str, GunState]:
        if not self.dir:
            return {}
        guns, stale = {}, 0
        for name, snap in read_json_dir(self.dir):
            try:
                guns[snap["gun_id"]] = GunState.restore(snap, gun_norm, self.model_created)
                stale += snap.get("model_created") != self.model_created
            except (KeyError, TypeError, ValueError):
                log.exception("bad gun state file %s", name)
        log.info("restored %d guns from %s (%d written under another model: warming up again)", len(guns), self.dir, stale)
        return guns


class Outbox:
    """Handoff records by event_id, each also written to <state dir>/outbox/<event_id>.json (D-1). Thread-safe:
    /predict adds from the threadpool, the push worker updates from the event loop."""

    def __init__(self, root: str | None) -> None:
        self.dir = os.path.join(root, "outbox") if root else None
        self.records: dict[str, HandoffRecord] = {}
        self.lock = threading.Lock()
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    def path(self, event_id: str) -> str:
        return os.path.join(self.dir, quote(event_id, safe="-_.") + ".json")

    def save(self, rec: HandoffRecord) -> None:
        if self.dir:
            try:
                write_json_atomic(self.path(rec.handoff.event_id), rec.model_dump(mode="json"))
            except OSError:
                log.exception("could not save handoff %s", rec.handoff.event_id)

    def add(self, rec: HandoffRecord) -> HandoffRecord:
        """Store a new record; an event_id already there (the same event produced twice) keeps the first one."""
        with self.lock:
            old = self.records.get(rec.handoff.event_id)
            if old is not None:
                return old
            self.records[rec.handoff.event_id] = rec
            if not self.dir and len(self.records) > OUTBOX_MAX:
                self.records.pop(next(iter(self.records)))
        self.save(rec)
        return rec

    def all(self) -> list[HandoffRecord]:
        with self.lock:
            return sorted(self.records.values(), key=lambda r: r.created_at)

    def get(self, event_id: str) -> HandoffRecord | None:
        with self.lock:
            return self.records.get(event_id)

    def due(self, now: datetime) -> list[HandoffRecord]:
        return [r for r in self.all() if r.delivery == "pending" and (r.next_attempt_at is None or r.next_attempt_at <= now)]

    def load(self, now: datetime) -> None:
        """Read the records back; queue the undelivered ones again (pending, and failed-after-retries - a rejected
        one (4xx) stays failed); prune what is past the retention."""
        if not self.dir:
            return
        for name, data in read_json_dir(self.dir):
            try:
                rec = HandoffRecord.model_validate(data)
            except ValueError:
                log.exception("bad handoff file %s", name)
                continue
            if rec.delivery == "failed" and not (rec.delivery_detail or "").startswith("rejected"):
                rec.delivery, rec.attempts = "pending", 0
            if rec.delivery == "pending":
                rec.next_attempt_at = now
            self.records[rec.handoff.event_id] = rec
        self.prune(now)
        n = sum(r.delivery == "pending" for r in self.records.values())
        log.info("outbox: %d handoffs restored from %s, %d to push", len(self.records), self.dir, n)

    def prune(self, now: datetime) -> None:
        keep_until = now - timedelta(days=OUTBOX_RETENTION_DAYS)
        with self.lock:
            old = [k for k, r in self.records.items() if r.delivery != "pending" and r.created_at < keep_until]
            for k in old:
                del self.records[k]
        for k in old:
            if self.dir and os.path.exists(self.path(k)):
                os.remove(self.path(k))


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("RSW_MODEL_PATH", MODEL_PATH)
    if not os.path.exists(model_path):
        raise RuntimeError(f"model not found: {model_path} — run .py/train.py or set RSW_MODEL_PATH")
    d = app.state.detector = Detector(model_path)
    log.info("loaded %s (window %ss, threshold %.4f)", model_path, d.window, d.threshold)
    state_dir = os.environ.get("RSW_STATE_DIR", STATE_DIR) or None
    app.state.state_dir = state_dir
    app.state.store = GunStore(state_dir, d.info.created)
    app.state.guns: dict[str, GunState] = app.state.store.load(d.gun_norm is not None)
    app.state.guns_lock = threading.Lock()
    app.state.outbox = Outbox(state_dir)
    app.state.outbox.load(datetime.now(timezone.utc))
    app.state.rag_url = os.environ.get("RSW_RAG_URL", RAG_URL or "") or None
    app.state.rag_backoff_s = tuple(float(x) for x in os.environ["RSW_RAG_BACKOFF_S"].split(",")) \
        if os.environ.get("RSW_RAG_BACKOFF_S") else RAG_BACKOFF_S
    app.state.rag_max_attempts = int(os.environ.get("RSW_RAG_MAX_ATTEMPTS", RAG_MAX_ATTEMPTS))
    app.state.outbox_poll_s = float(os.environ.get("RSW_OUTBOX_POLL_S", OUTBOX_POLL_S))
    if not hasattr(app.state, "rag_transport"):
        app.state.rag_transport = None  # httpx transport override (tests route the push to an in-process mock)
    worker = asyncio.create_task(outbox_worker(app)) if app.state.rag_url else None
    try:
        yield
    finally:
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass


app = FastAPI(title="RSW gun anomaly detection", version="0.1.0", lifespan=lifespan,
              description="Real-time anomaly scoring of resistance-spot-welding gun sensor streams.")


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 echoes the offending input, and a NaN / inf reading cannot be written as JSON (the
    handler itself raised -> 500). Echo non-finite numbers as strings."""
    def safe(v: Any) -> Any:
        return str(v) if isinstance(v, float) and not np.isfinite(v) else v
    errors = [{**e, "input": safe(e.get("input"))} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


# ============================================================ endpoints
@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    d: Detector = request.app.state.detector
    ob = request.app.state.outbox.all()
    return {"status": "ok", "model_path": d.path, "model": d.info.model_dump(), "guns_tracked": len(request.app.state.guns),
            "max_chunk": MAX_CHUNK, "state_dir": request.app.state.state_dir, "rag_url": request.app.state.rag_url,
            "outbox": {k: sum(r.delivery == k for r in ob) for k in ("pending", "delivered", "failed", "pull_only")}}


@app.get("/model")
async def model_card(request: Request) -> dict[str, Any]:
    d: Detector = request.app.state.detector
    b = d.bundle
    metrics = b.get("metrics", {})
    return {"model": d.info.model_dump(), "features": d.model_cols,
            "preprocess": {"gap_fill_limit_s": d.gap_fill_limit, "outlier_hi": d.outlier_hi, "outlier_lo": d.outlier_lo,
                           "alarm_max_non_welding": d.alarm_max_non_welding,
                           "rolling_s": ROLL_S, "scaled_columns": sorted(d.scaler)},
            "gun_norm": d.gun_norm, "dropped_features": b.get("dropped_features"),
            "validation": {k: v for k, v in metrics.get("val", {}).items() if k != "per_file"},
            "test": {k: v for k, v in metrics.get("test", {}).items() if k != "per_file"} or None,
            "train_files": b.get("train_files"), "val_files": b.get("val_files"),
            "test_files": metrics.get("test", {}).get("files")}


def gun_state(request: Request, gun_id: str, create: bool = False) -> GunState:
    with request.app.state.guns_lock:
        g = request.app.state.guns.get(gun_id)
        if g is None:
            if not create:
                raise HTTPException(status_code=404, detail=f"unknown gun {gun_id}")
            g = request.app.state.guns[gun_id] = GunState(gun_norm=request.app.state.detector.gun_norm is not None)
        return g


def check_times(g: GunState, readings: list[SensorReading]) -> None:
    """Refuse chunks the buffer cannot hold in time order: a stray timestamp would otherwise become the 'latest
    segment' and blind the gun (every later window 202, the rule skipping the rows before it) for 30 min."""
    times = [r.time for r in readings]
    span = (max(times) - min(times)).total_seconds()
    if span > BUFFER_S:
        raise HTTPException(status_code=422, detail=f"readings span {span:.0f} s > the {BUFFER_S} s buffer: send them in "
                                                    "time-ordered chunks (a wrong timestamp in the chunk?)")
    if g.latest is not None and (g.latest - max(times)).total_seconds() > BUFFER_S:
        raise HTTPException(status_code=409, detail=f"readings end at {max(times)}, more than {BUFFER_S} s before the "
                                                    f"newest reading of this gun ({g.latest}); if the clock was reset, "
                                                    "DELETE /guns/{gun_id} first")


# a sync endpoint: FastAPI runs it in the threadpool, so the ~40 ms of pandas work per request does not block the event loop
@app.post("/predict", response_model=AnomalyResult | WarmingUp,
          responses={202: {"model": WarmingUp, "description": "not enough history yet"},
                     409: {"description": "timestamps went back more than the buffer"},
                     422: {"description": "invalid readings, or a chunk longer than the buffer"}})
def predict(req: PredictRequest, request: Request) -> AnomalyResult | JSONResponse:
    g = gun_state(request, req.gun_id, create=True)
    with g.lock:
        check_times(g, req.readings)
        try:
            return score_request(req, request, g)
        finally:
            request.app.state.store.save(req.gun_id, g)  # C-1: the state as this request left it


def score_request(req: PredictRequest, request: Request, g: GunState) -> AnomalyResult | JSONResponse:
    """One /predict under the gun's lock: buffer -> features -> rule -> windows -> score / judge -> response."""
    d: Detector = request.app.state.detector
    d.ingest(g, req.readings)
    feats = d.features(g.frame(), g.carry)
    g.pending_rules.extend(d.rule_check(g, feats.code) if feats is not None else [])
    wins = d.windows(g, feats.f) if feats is not None else []
    if not wins:  # the triggers stay pending: reported with the next scored response, not lost
        n = 0 if feats is None else len(feats.f)
        body = WarmingUp(gun_id=req.gun_id, n_samples=n, samples_needed=d.window,
                         message="no welding row seen yet: non-welding rows (c16 <= 0) carry the last welding values"
                         if feats is None else f"no complete {d.window}-s clock window with >= {d.window // 2} "
                         f"contiguous 1 Hz samples yet (have {n} samples)")
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=body.model_dump())
    hits, g.pending_rules = g.pending_rules, []
    rule = next((h for h in hits if h.critical), hits[0] if hits else None)
    f = d.gun_normalise(g, feats.f)
    fresh = g.last_end is None or f.index[wins[-1].stop - 1] > g.last_end
    vecs = np.stack([d.window_vector(f.iloc[s]) for s in wins])
    verdicts = d.judge(g, f, wins, trainlib.anomaly_score(d.model, vecs), vecs, fresh)
    score, contribs = d.score(vecs[-1])  # the latest window, with feature attributions
    g.last_score = score
    result = build_result(d, g, req.gun_id, feats.f, f, verdicts, score, contribs, rule, vecs[-1])
    record_focus(g, result, rule)
    critical = result.critical_in_request
    for h in hits:
        count(g, h.time, rule_triggers=1, rule_repeats=h.repeat, rule_out_of_profile=h.out_of_profile and not h.repeat)
    if (critical and not g.was_critical) or (rule is not None and rule.critical):
        attach_handoff(request, result)
        count(g, result.window_end, critical_events=1)
    for h in hits:  # the other triggers of this request (e.g. E016 then E029): a handoff each now, not later
        if h is rule:
            continue
        record_focus(g, None, h)
        if h.critical:
            attach_handoff(request, build_result(d, g, req.gun_id, feats.f, f, verdicts, score, contribs, h, vecs[-1]))
            count(g, result.window_end, critical_events=1)
    g.was_critical = critical
    drift = drift_reasons(g)
    if drift != g.drift:
        for reason in drift:
            if reason not in g.drift:
                log.warning("gun %s drift: %s", req.gun_id, reason)
        g.drift = drift
    result.drift_warning = drift
    return result


def build_result(d: Detector, g: GunState, gun_id: str, raw_f: pd.DataFrame, f: pd.DataFrame, verdicts: list[Verdict],
                 score: float, contribs: list[FeatureContribution], rule: RuleHit | None,
                 vec: np.ndarray | None = None) -> AnomalyResult:
    """The response for the latest window. Severity: the model is critical after a sustained alarm, the
    terminal-code rule on an episode start (history.md B-3) - a repeat of the rule only warns."""
    v = verdicts[-1]
    w = v.w
    sustained_in_request = any(x.sustained for x in verdicts)
    model_critical_in_request = any(x.critical for x in verdicts)
    rule_critical = rule is not None and rule.critical
    severity = "critical" if (v.critical or rule_critical) else "warning" if (v.alarm or rule is not None) else "normal"

    def source(model: bool) -> str:
        return "model+rule" if model and rule_critical else "model" if model else "rule" if rule_critical else "none"

    latest_code = str(w["error_code"].iloc[-1])
    rw = raw_f.loc[w.index]  # context in raw units: the frame BEFORE the gun centring (it shifts the counters)
    ctx = WindowContext(latest_error_code=latest_code, error_active_share=float(w["error_active"].mean()),
                        non_welding_share=v.nw_share,
                        welds_in_window=d.unscale("welds_delta", float(rw["welds_delta"].sum()), len(rw)),
                        welds_10min=d.unscale("welds_10min", float(rw["welds_10min"].iloc[-1])),
                        weld_duty_10min=float(w["weld_duty_10min"].iloc[-1]),
                        known_code_class_hint=CODE_TO_CLASS.get(latest_code),
                        known_code_in_profile=None if g.profile is None or latest_code not in CODE_TO_CLASS
                        else CODE_TO_CLASS[latest_code] in g.profile,
                        mean_dev=d.mean_dev(vec) if vec is not None else None)
    return AnomalyResult(
        gun_id=gun_id, window_start=w.index[0].to_pydatetime(), window_end=w.index[-1].to_pydatetime(),
        n_samples=int(len(w)), history_s=int(((f["segment"] == w["segment"].iloc[-1]) & (f.index <= w.index[-1])).sum()), is_anomaly=v.is_anomaly, anomaly_score=score,
        alarm_held=v.held, hold_reason=v.hold_reason,
        threshold=v.threshold, gun_norm=g.norm_status, gun_threshold=g.gun_threshold,
        score_z=(score - d.score_mean) / d.score_std,
        model_critical=v.critical,
        severity=severity, rule_triggered=rule is not None, rule_repeat=rule is not None and rule.repeat,
        rule_out_of_profile=rule is not None and rule.out_of_profile,
        rule_trigger_time=rule.time.to_pydatetime() if rule else None, rule_code=rule.code if rule else None,
        rule_class_hint=CODE_TO_CLASS.get(rule.code) if rule else None,
        rule_code_active=latest_code in d.rule_codes, severity_source=source(v.critical),
        critical_source=source(model_critical_in_request),
        sustained_alarm=v.sustained, consecutive_alarms=g.consecutive_alarms, alarm_duration_s=v.duration,
        windows_scored=len(verdicts), sustained_in_request=sustained_in_request,
        critical_in_request=model_critical_in_request or rule_critical,
        contributing_features=contribs, context=ctx, model=d.info)


def record_focus(g: GunState, result: AnomalyResult | None, rule: RuleHit | None) -> None:
    """Trace bookkeeping: rule events, and the explanation of the latest warning / critical window."""
    if rule is not None:
        kind = "rule_repeat" if rule.repeat else "rule_out_of_profile" if rule.out_of_profile else "rule"
        g.trace_events.append({"time": rule.time, "kind": kind, "code": rule.code})
    if result is None:
        return
    if result.severity != "normal" or g.focus is None or g.focus["severity"] == "normal":
        g.focus = {"window_end": result.window_end, "anomaly_score": result.anomaly_score, "threshold": result.threshold,
                   "severity": result.severity, "severity_source": result.severity_source,
                   "contributing_features": result.contributing_features}


def handoff_event_id(result: AnomalyResult) -> str:
    """The same event always gets the same id (gun, window, trigger), so a resend after a restart cannot become a
    second event downstream - the receiver drops a known event_id."""
    key = f"{result.gun_id}|{result.window_end.isoformat()}|{result.critical_source}|{result.rule_trigger_time}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:32]


def attach_handoff(request: Request, result: AnomalyResult) -> None:
    """ML -> RAG handoff (same events replay.py prints) into the outbox; the push worker delivers it when
    RSW_RAG_URL is set. A mapping error is logged and leaves the response without a handoff: the scored state is
    already committed, so a 500 here would turn the client's retry into a resend that never produces the handoff."""
    try:
        result.handoff = RagHandoff(**rag_mapping.build_handoff(result.model_dump(mode="json"), handoff_event_id(result)))
    except Exception:
        log.exception("handoff mapping failed for %s at %s", result.gun_id, result.window_end)
        return
    now = datetime.now(timezone.utc)
    push = request.app.state.rag_url is not None
    request.app.state.outbox.add(HandoffRecord(handoff=result.handoff, created_at=now,
                                               delivery="pending" if push else "pull_only",
                                               next_attempt_at=now if push else None))


async def outbox_worker(app: FastAPI) -> None:
    """Push every due handoff, oldest first, then sleep; prune the retention once an hour. Never dies on an error."""
    last_prune = 0.0
    while True:
        try:
            for rec in app.state.outbox.due(datetime.now(timezone.utc)):
                await push_handoff(app, rec)
            if time.monotonic() - last_prune > 3600:
                app.state.outbox.prune(datetime.now(timezone.utc))
                last_prune = time.monotonic()
        except Exception:
            log.exception("outbox worker")
        await asyncio.sleep(app.state.outbox_poll_s)


async def push_handoff(app: FastAPI, rec: HandoffRecord) -> None:
    """POST one handoff (Idempotency-Key = event_id). 2xx -> delivered; 4xx other than 408 / 429 -> failed
    ("rejected", a contract problem - not retried); anything else -> retry after the backoff, failed after
    RSW_RAG_MAX_ATTEMPTS. The record is saved after every attempt."""
    import httpx  # only needed when pushing

    now = datetime.now(timezone.utc)
    rec.attempts, rec.last_attempt_at = rec.attempts + 1, now
    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT_S, transport=app.state.rag_transport) as client:
            r = await client.post(app.state.rag_url, json=rec.handoff.model_dump(mode="json"),
                                  headers={"Idempotency-Key": rec.handoff.event_id})
        code = r.status_code
        if 200 <= code < 300:
            rec.delivery, rec.delivery_detail, rec.next_attempt_at = "delivered", f"HTTP {code}", None
            if r.headers.get("content-type", "").startswith("application/json"):
                rec.rag_response = r.json()
        elif 400 <= code < 500 and code not in (408, 429):
            rec.delivery, rec.delivery_detail, rec.next_attempt_at = "failed", f"rejected: HTTP {code}", None
            log.warning("RAG rejected %s: HTTP %s %s", rec.handoff.event_id, code, r.text[:200])
        else:
            retry_later(app, rec, f"HTTP {code}", now)
    except Exception as e:  # network errors must not break anything; the record stays pullable
        retry_later(app, rec, f"{type(e).__name__}: {e}", now)
    app.state.outbox.save(rec)


def retry_later(app: FastAPI, rec: HandoffRecord, detail: str, now: datetime) -> None:
    backoff = app.state.rag_backoff_s
    if rec.attempts >= app.state.rag_max_attempts:
        rec.delivery, rec.delivery_detail, rec.next_attempt_at = "failed", f"{detail} (after {rec.attempts} attempts)", None
        log.warning("RAG push gave up on %s: %s", rec.handoff.event_id, rec.delivery_detail)
    else:
        rec.delivery_detail = detail
        rec.next_attempt_at = now + timedelta(seconds=backoff[min(rec.attempts, len(backoff)) - 1])


@app.get("/handoffs", response_model=list[HandoffRecord])
async def handoffs(request: Request, gun_id: str | None = None,
                   delivery: Literal["pull_only", "pending", "delivered", "failed"] | None = None,
                   limit: int = Query(50, ge=1, le=OUTBOX_MAX)) -> list[HandoffRecord]:
    """Handoff documents of critical events (oldest first; kept RSW_STATE_DIR / 30 days)."""
    recs = [r for r in request.app.state.outbox.all()
            if (gun_id is None or r.handoff.gun_id == gun_id) and (delivery is None or r.delivery == delivery)]
    return recs[-limit:]


@app.get("/handoffs/{event_id}", response_model=HandoffRecord)
async def handoff(event_id: str, request: Request) -> HandoffRecord:
    rec = request.app.state.outbox.get(event_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown event {event_id}")
    return rec


@app.post("/handoffs/preview", response_model=RagHandoff)
async def handoff_preview(result: AnomalyResult) -> RagHandoff:
    """Map any AnomalyResult (e.g. the /predict schema example) to its handoff without touching server state."""
    return RagHandoff(**rag_mapping.build_handoff(result.model_dump(mode="json")))


@app.get("/guns", response_model=list[GunStatus])
def guns(request: Request) -> list[GunStatus]:
    out = []
    with request.app.state.guns_lock:
        items = list(request.app.state.guns.items())
    for gid, g in items:
        times = [r["time"] for r in g.rows]
        out.append(GunStatus(gun_id=gid, n_samples=len(g.rows), first_time=min(times) if times else None,
                             last_time=max(times) if times else None, consecutive_alarms=g.consecutive_alarms,
                             last_score=g.last_score, gun_norm=g.norm_status, gun_threshold=g.gun_threshold,
                             warmup_rows=g.warmup_rows, drift_warning=g.drift, expected_fault_classes=g.profile))
    return out


@app.put("/guns/{gun_id}/profile", response_model=GunProfile)
def put_profile(gun_id: str, profile: GunProfile, request: Request) -> GunProfile:
    """Set the gun's fault-class profile (C-3). Creates the gun if it has not streamed yet. DELETE /guns/{id} clears it."""
    g = gun_state(request, gun_id, create=True)
    with g.lock:
        # [] = no class expected (every terminal code only warns); None = no profile (every code critical)
        cls = profile.expected_fault_classes
        g.profile = None if cls is None else sorted(set(cls))
        request.app.state.store.save(gun_id, g)
        return GunProfile(expected_fault_classes=g.profile)


@app.get("/guns/{gun_id}/profile", response_model=GunProfile)
def get_profile(gun_id: str, request: Request) -> GunProfile:
    g = gun_state(request, gun_id)
    return GunProfile(expected_fault_classes=g.profile)


@app.get("/guns/{gun_id}/stats", response_model=GunStats)
def one_gun_stats(gun_id: str, request: Request, days: int = Query(7, ge=1, le=STATS_DAYS)) -> GunStats:
    """Daily counters of one gun (last `days` days with data) - is the model behaving as designed (~1 % normal alarms)?"""
    g = gun_state(request, gun_id)
    with g.lock:
        return gun_stats(gun_id, g, days)


@app.get("/stats", response_model=list[GunStats])
def all_stats(request: Request, days: int = Query(7, ge=1, le=STATS_DAYS), drift_only: bool = False) -> list[GunStats]:
    """Every gun's counters over the last `days` days (per-day rows omitted: see /guns/{id}/stats)."""
    with request.app.state.guns_lock:
        items = sorted(request.app.state.guns.items())
    out = []
    for gid, g in items:
        with g.lock:
            st = gun_stats(gid, g, days)
        if not drift_only or st.drift_warning:
            out.append(st.model_copy(update={"days": []}))
    return out


def build_trace(d: Detector, gun_id: str, g: GunState, minutes: int, features: list[str] | None) -> Trace:
    """Last `minutes` of g.trace as chart series. Default features: the focus window's top positive contributors."""
    pts = list(g.trace)
    if pts:
        t_min = pts[-1]["window_end"] - pd.Timedelta(minutes=minutes)
        pts = [p for p in pts if p["window_end"] > t_min]
    focus = g.focus
    if focus is not None and pts and focus["window_end"] < pts[0]["window_start"].to_pydatetime():
        focus = None  # the explanation is older than the requested span
    col = {c: j for j, c in enumerate(d.model_cols)}
    if features:
        bad = [f for f in features if f not in col]
        if bad:
            raise HTTPException(status_code=422, detail=f"unknown features {bad}; see GET /model")
        chosen = [TraceFeature(feature=f, sensor=split_feature(f)[0], sensor_name=SENSOR_NAME.get(split_feature(f)[0], f),
                               statistic=split_feature(f)[1]) for f in features]
    else:
        contribs = focus["contributing_features"] if focus else []
        chosen = [TraceFeature(feature=c.feature, sensor=c.sensor, sensor_name=c.sensor_name, statistic=c.statistic,
                               share=c.share) for c in contribs
                  if c.contribution > 0 and c.sensor not in TRACE_SKIP][:TRACE_TOP_FEATURES]
    idx = [col[f.feature] for f in chosen]
    points = [TracePoint(window_start=p["window_start"].to_pydatetime(), window_end=p["window_end"].to_pydatetime(),
                         score=p["score"], threshold=p["threshold"], alarm=p["alarm"], held=p["held"],
                         sustained=p["sustained"], critical=p.get("critical", p["sustained"]),
                         non_welding_share=p["non_welding_share"], error_codes=p["error_codes"],
                         gun_norm=p["gun_norm"],
                         deviation={f.feature: round(float(p["vec"][j] - d.reference[j]), 4) for f, j in zip(chosen, idx)})
              for p in pts]
    t0 = points[0].window_start if points else None
    events = [TraceEvent(time=e["time"].to_pydatetime(), kind=e["kind"], code=e.get("code"),
                         class_hint=CODE_TO_CLASS.get(e.get("code") or ""))
              for e in g.trace_events if t0 is None or e["time"].to_pydatetime() >= t0]
    return Trace(gun_id=gun_id, window_s=d.window, minutes=minutes, n_points=len(points), model=d.info,
                 focus=TraceFocus(**focus) if focus else None, features=chosen, axes=TraceAxes(score=d.score_axis),
                 events=events, points=points)


@app.get("/guns/{gun_id}/trace", response_model=Trace)
def trace(gun_id: str, request: Request, minutes: int = Query(60, ge=5, le=TRACE_WINDOWS),
                features: list[str] | None = Query(None, description="window features to chart, e.g. c5_mean; "
                                                                      "default: the focus window's top contributors")) -> Trace:
    """Score, threshold, alarm state, controller codes and sensor deviations of the last `minutes` of scored windows,
    plus the explanation (contributing features) of the latest warning / critical window. Feeds the cause chart
    (GET /guns/{id}/trace/view)."""
    g = gun_state(request, gun_id)
    with g.lock:
        return build_trace(request.app.state.detector, gun_id, g, minutes, features)


@app.get("/guns/{gun_id}/trace/view", response_class=HTMLResponse)
async def trace_view(gun_id: str) -> HTMLResponse:
    """Live cause chart for one gun: reads GET /guns/{id}/trace every 60 s (static/trace.html)."""
    if not os.path.exists(TRACE_HTML):
        raise HTTPException(status_code=404, detail=f"chart page missing: {TRACE_HTML}")
    with open(TRACE_HTML, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.delete("/guns/{gun_id}", status_code=status.HTTP_204_NO_CONTENT)
def forget_gun(gun_id: str, request: Request, keep_profile: bool = True) -> None:
    """Forget the gun's stream state (buffer, warm-up, alarms, rule history, stats) - e.g. after maintenance. Its
    profile (PUT /guns/{id}/profile) is configuration and is kept unless keep_profile=false."""
    with request.app.state.guns_lock:
        old = request.app.state.guns.pop(gun_id, None)
        if old is None:
            raise HTTPException(status_code=404, detail=f"unknown gun {gun_id}")
        request.app.state.store.delete(gun_id)
        if keep_profile and old.profile is not None:
            g = request.app.state.guns[gun_id] = GunState(gun_norm=request.app.state.detector.gun_norm is not None)
            g.profile = old.profile
            request.app.state.store.save(gun_id, g)
