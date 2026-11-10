"""
D-9 ML feedback loop - inputs, report and command line (docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md,
4, 6, 8). Reads the ML server's state dir (window log, outbox, gun profiles), the closed cases (an api DB file, a
/api/history JSON file or the api's address) and, optionally, B-19's lab cache; the evaluator (.py/feedback_eval.py)
labels and judges; this writes report.md and report.json. The loop only proposes: it changes nothing.

    python .py/feedback_loop.py --state <state dir> [--cases <api DB> | --history <JSON> | --api <url>]
           [--lab <B-19 cache dir>] [--lab-meta <meta.csv>] [--tag <name>] [--out <dir>]

Exit 0 when the report is written, 2 when a given input is missing or cannot be opened. Inputs not given, and a
state dir without a window log yet, are reported as 해당 없음.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import feedback_eval as fe  # noqa: E402
from preprocess import TERMINAL_TO_CLASS  # noqa: E402

WINDOW_COLS = ["gun", "window_start", "score", "warmup", "non_welding_share", "error_active", "early_warning",
               "ew_threshold", "model"]


class InputError(Exception):
    """A given input is missing or cannot be opened (exit 2)."""


def utc_naive(s: Any) -> pd.Timestamp | None:
    """An ISO time (with Z or an offset, or naive UTC) -> naive UTC (MEMORY.md 2: naive UTC everywhere)."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    t = pd.Timestamp(s)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


# ---- the server's state dir
def read_windows(state: str) -> tuple[pd.DataFrame, dict, list[str]]:
    """windows/<gun>/<day>.jsonl -> the evaluator's window table (+ model), windows/models.json, warnings.
    Window starts are floored to the minute bucket; a (gun, minute) written twice keeps its last line (restart);
    with several models only the model of the latest window stays (scores of different models do not mix)."""
    root = os.path.join(state, "windows")
    notes: list[str] = []
    models: dict = {}
    mpath = os.path.join(root, "models.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as fh:
                models = json.load(fh)
        except (OSError, ValueError):
            notes.append("읽지 못한 줄 1개를 건너뜀 (windows/models.json)")
    rows, bad = [], 0
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        gun = unquote(os.path.basename(d))
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            with open(f, encoding="utf-8", errors="replace") as fh:  # a bad byte fails its line's JSON, not the report
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        x = json.loads(line)
                        rows.append((gun, x["window_start"], float(x["score"]), bool(x["warmup"]),
                                     float(x["non_welding_share"]), bool(x.get("error_codes")),
                                     bool(x["early_warning"]), x.get("ew_threshold"), str(x["model"])))
                    except (ValueError, KeyError, TypeError):
                        bad += 1
    if bad:
        notes.append(f"읽지 못한 줄 {bad}개를 건너뜀 (창 기록)")
    w = pd.DataFrame(rows, columns=WINDOW_COLS)
    w["window_start"] = pd.to_datetime(w["window_start"], utc=True).dt.tz_localize(None).dt.floor("60s")
    w["ew_threshold"] = pd.to_numeric(w["ew_threshold"], errors="coerce").astype(float)
    w = w.drop_duplicates(["gun", "window_start"], keep="last")
    if w["model"].nunique() > 1:
        last = w["window_start"].max()
        newest = max(w.loc[w["window_start"] == last, "model"])
        dropped = w.loc[w["model"] != newest, "model"].value_counts().sort_index()
        notes.append(f"창 기록에 모델이 여럿이라 가장 늦은 모델({newest})의 창만 평가함 (뺀 창: "
                     + ", ".join(f"{m} {n}개" for m, n in dropped.items()) + ")")
        w = w[w["model"] == newest]
    w = w.sort_values(["gun", "window_start"], kind="stable").reset_index(drop=True)
    return w, models, notes


def read_outbox(state: str) -> tuple[list[dict], list[str]]:
    """outbox/<event_id>.json -> the handoff documents (critical events and early warnings), warnings."""
    docs, bad = [], 0
    for f in sorted(glob.glob(os.path.join(state, "outbox", "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                docs.append(json.load(fh)["handoff"])
        except (OSError, ValueError, KeyError, TypeError):
            bad += 1
    return docs, ([f"읽지 못한 줄 {bad}개를 건너뜀 (outbox)"] if bad else [])


def read_profiles(state: str) -> dict[str, list[str] | None]:
    """guns/<gun>.json -> the gun's fault-class profile (None = every class; PUT /guns/{id}/profile)."""
    out: dict[str, list[str] | None] = {}
    for f in sorted(glob.glob(os.path.join(state, "guns", "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                s = json.load(fh)
            out[str(s["gun_id"])] = s.get("profile")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


# ---- the closed cases (spec 6.5)
def _case(case_id: str, e: dict, closure: dict | None, repeat_ids: list | None, simulated: bool) -> dict:
    tr = e.get("trigger") or {}
    return {"case_id": str(case_id),
            "event": {"event_id": e["event_id"], "gun_id": e["gun_id"], "detected_at": e["detected_at"],
                      "trigger": {k: tr.get(k) for k in ("source", "rule_code", "rule_trigger_time")},
                      "event_type": e.get("event_type") or "critical",  # v1.2 handoffs have none: critical
                      "onset_at": (e.get("early_warning") or {}).get("onset_at")},  # only the api DB keeps it
            "closure": closure, "repeat_event_ids": None if repeat_ids is None else [str(x) for x in repeat_ids],
            "simulated": bool(simulated)}


def _read_db(path: str) -> tuple[list[dict], dict[str, str]]:
    if not os.path.isfile(path):
        raise InputError(f"--cases {path}: no such file")
    try:
        con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)  # read-only
        try:
            rows = con.execute("SELECT data FROM cases").fetchall()
            events = {str(k): str(v) for k, v in con.execute("SELECT event_id, case_id FROM events").fetchall()}
        finally:
            con.close()
        cases = []
        for (data,) in rows:
            d = json.loads(data)
            cases.append(_case(d["case_id"], d["event"], (d.get("records") or {}).get("closure"),
                               [r["event_id"] for r in d.get("repeats") or []], d.get("simulated", False)))
    except (sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        raise InputError(f"--cases {path}: {exc}") from exc
    return cases, events


def _from_history(data: Any, where: str) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise InputError(f"{where}: not a /api/history response (no items list)")
    try:
        return [_case(i["case_id"], i["event"], i.get("closure"), i.get("repeat_event_ids"), i.get("simulated", False))
                for i in data["items"]]
    except (KeyError, TypeError) as exc:
        raise InputError(f"{where}: bad item ({exc})") from exc


def read_cases(cases: str | None = None, history: str | None = None,
               api: str | None = None) -> tuple[list[dict], dict[str, str] | None]:
    """The cases, normalised, and - for an api DB - its events table (event_id -> case_id)."""
    if cases:
        return _read_db(cases)
    if history:
        try:
            with open(history, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise InputError(f"--history {history}: {exc}") from exc
        return _from_history(data, f"--history {history}"), None
    if api:
        import httpx
        url = api.rstrip("/") + "/api/history"
        try:
            r = httpx.get(url, timeout=10)
            data = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError) as exc:
            raise InputError(f"--api {url}: {exc}") from exc
        if r.status_code != 200:
            raise InputError(f"--api {url}: HTTP {r.status_code}")
        return _from_history(data, f"--api {url}"), None
    return [], None


def case_index(cases: list[dict], events: dict | None, outbox: list[dict]) -> dict[str, dict]:
    """event_id -> its case (spec 6.5): the api DB's events table; else the representative and repeat_event_ids;
    for a /api/history from before 11.1 (no repeat_event_ids) the alerts of the same gun and rule code (none = none)
    detected between the case's detection and its closure that are no other case's representative."""
    by_id = {c["case_id"]: c for c in cases}
    if events is not None:
        return {eid: by_id[cid] for eid, cid in events.items() if cid in by_id}
    index: dict[str, dict] = {}
    for c in cases:
        index[c["event"]["event_id"]] = c
        for eid in c["repeat_event_ids"] or []:
            index[eid] = c
    old = [c for c in cases if c["repeat_event_ids"] is None and c["closure"]]
    reps = {c["event"]["event_id"] for c in cases}
    for h in outbox if old else []:
        eid = h.get("event_id")
        if eid in index or eid in reps:
            continue
        t, code = utc_naive(h.get("detected_at")), (h.get("trigger") or {}).get("rule_code")
        match = [c for c in old if c["event"]["gun_id"] == h.get("gun_id") and c["event"]["trigger"]["rule_code"] == code
                 and utc_naive(c["event"]["detected_at"]) <= t <= utc_naive(c["closure"]["at"])]
        if match:
            index[eid] = max(match, key=lambda c: utc_naive(c["event"]["detected_at"]))
    return index


def failures_from(outbox: list[dict], index: dict[str, dict], cases: list[dict] = ()) -> pd.DataFrame:
    """Spec 6.1: critical handoffs (event_type critical, or none before v1.3) with a rule code; T = rule_trigger_time
    (else detected_at). Closed as a false alarm -> rejected, resolved -> confirmed, anything else -> presumed.
    The outbox keeps 30 days; a handoff gone from it counts through its case's own event (a pruned repeat cannot)."""
    rows = []

    def add(e: dict, case: dict | None) -> None:
        tr = e.get("trigger") or {}
        if e.get("event_type") not in (None, "critical") or not tr.get("rule_code"):
            return
        outcome = ((case or {}).get("closure") or {}).get("outcome")
        status = "rejected" if outcome == "false_alarm" else "confirmed" if outcome == "resolved" else "presumed"
        rows.append((e["gun_id"], utc_naive(tr.get("rule_trigger_time") or e.get("detected_at")), status,
                     e["event_id"], tr["rule_code"]))

    for h in outbox:
        add(h, index.get(h.get("event_id")))
    in_outbox = {h.get("event_id") for h in outbox}
    for c in cases:
        if c["event"]["event_id"] not in in_outbox:
            add(c["event"], c)
    f = pd.DataFrame(rows, columns=["gun", "T", "status", "event_id", "rule_code"])
    f["T"] = pd.to_datetime(f["T"])
    return f.sort_values(["gun", "T"], kind="stable").reset_index(drop=True)


def critical_cases(cases: list[dict]) -> list[fe.CriticalCase]:
    """Spec 6.4: one per case (repeats are not counted); class from the terminal code (none = model only)."""
    out = []
    for c in cases:
        e = c["event"]
        if e["event_type"] == "early_warning":
            continue
        outcome = (c.get("closure") or {}).get("outcome")
        out.append(fe.CriticalCase(case_id=c["case_id"], gun=e["gun_id"],
                                   cls=TERMINAL_TO_CLASS.get(e["trigger"].get("rule_code") or ""),
                                   outcome="false_alarm" if outcome == "false_alarm" else
                                   "real" if outcome == "resolved" else "unknown",
                                   detected_at=utc_naive(e["detected_at"])))
    return out


def ew_links(cases: list[dict], index: dict[str, dict], outbox: list[dict]) -> list[fe.EwLink]:
    """Spec 6.3: every early-warning handoff of an early-warning case, tied to its onset window by early_warning.onset_at
    (the onset window's end, floored to the minute bucket = the window start). No outbox record -> the case's own copy
    (api DB, the representative alert), else no onset."""
    docs = {h.get("event_id"): h for h in outbox}
    links = []
    for c in cases:
        if c["event"]["event_type"] != "early_warning":
            continue
        rep = c["event"]["event_id"]
        eids = [rep] + [eid for eid, x in index.items() if x["case_id"] == c["case_id"] and eid != rep]
        closure = c.get("closure") or {}
        for eid in eids:
            doc = docs.get(eid)
            onset = utc_naive((doc.get("early_warning") or {}).get("onset_at") if doc
                              else c["event"].get("onset_at") if eid == rep else None)
            links.append(fe.EwLink(case_id=c["case_id"], outcome=closure.get("outcome") or "open",
                                   reason=closure.get("reason"), note=closure.get("note"), gun=c["event"]["gun_id"],
                                   onset_start=None if onset is None else onset.floor("60s")))
    return links


# ---- the lab (B-19's out-of-fold runs)
def iter_lab(lab_dir: str, meta_csv: str) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """One (run name, window table, failure table) per results/B-19/cache/scores_cv_*.parquet; T = meta fault_time."""
    meta = pd.read_csv(meta_csv, parse_dates=["fault_time"])
    for p in sorted(glob.glob(os.path.join(lab_dir, "scores_cv_*.parquet"))):
        d = pd.read_parquet(p, columns=["file", "time", "error_active", "non_welding", "warmup", "score_ens"])
        w = pd.DataFrame({"gun": d["file"].astype(str), "window_start": d["time"],
                          "score": d["score_ens"].astype(float), "warmup": d["warmup"].astype(bool),
                          "non_welding_share": d["non_welding"].astype(float), "error_active": d["error_active"] > 0})
        m = meta[meta["file"].astype(str).isin(set(w["gun"]))]
        f = pd.DataFrame({"gun": m["file"].astype(str).to_numpy(), "T": m["fault_time"].to_numpy()})
        yield os.path.splitext(os.path.basename(p))[0], w, f


# ---- the report (spec 8)
SCHEMA = "feedback_report/1"
NOTE_KO = "이 보고서는 제안만 하며, 루프는 아무것도 바꾸지 않음"
SIMULATED_KO = "시뮬레이션 라벨 포함 — 성능 아님"
CLOSED_ONLY_KO = ("닫힌 케이스만 읽음(--history · --api): 열린 케이스가 있어도 프로필 제안이 막히지 않고 critical 미확인 수에도 "
                  "들어가지 않음. 열린 케이스까지 보려면 --cases(api DB)로 돌릴 것")
CRITERIA = ["MARGIN", "CHEAPER", "Q_MINUS", "Q_PLUS", "MIN_FAILURES", "PROFILE_CASES", "PROFILE_DAYS", "FA_CAP",
            "REFERENCE", "GATE_NON_WELDING", "ZONE_S", "MIN_ZONE_WINDOWS", "PLACEBO_DAYS", "CALIBRATION_LEAD_S",
            "N_BOOT", "SEED_SINGLE", "SEED_PAIRED", "POST_FAILURE_S", "MIN_GUNS"]
LINKED = {"설비 정지 · 전원 꺼짐": "GATE", "정상 작업 중 변동": "Q+"}  # spec 7.7
VERDICT_ORDER = [("propose", "반영 제안"), ("review", "검토 필요"), ("hold", "보류"), ("reject", "반영 안 함"),
                 ("not_evaluated", "평가 안 함")]


def _z(t: Any) -> str | None:
    return None if t is None else pd.Timestamp(t).isoformat() + "Z"


def _clean(o: Any) -> Any:
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python, times -> ISO with Z."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, range)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (pd.Timestamp, datetime)):
        return _z(o)
    return o


def challenger_entry(ch: fe.Challenger, lab: dict | None, field: dict | None) -> dict:
    """One challenger in the report: both evidence sources, the combined verdict (7.5) and the thresholds behind them."""
    lab_r = lab["challengers"][ch.name] if lab else None
    field_r = field["challengers"][ch.name] if field else None
    lab_v, field_v = (lab_r or {}).get("verdict", "na"), (field_r or {}).get("verdict", "na")
    code, ko = fe.combine(lab_v, field_v, bool((field_r or {}).get("against_apply")))
    return {"name": ch.name, "reason": ch.reason, "apply_ko": ch.apply_ko, "lab": lab_r, "field": field_r,
            "lab_verdict": lab_v, "field_verdict": field_v, "verdict": code, "verdict_ko": ko,
            "thresholds": {"served": (field or {}).get("champion_threshold"), "field": (field_r or {}).get("threshold"),
                           "lab_mean": fe.lab_mean_thresholds(lab).get(ch.name)}}


def build_report(args: argparse.Namespace, now: datetime) -> dict:
    """Read every input, evaluate (lab, field, confirmed failures only, profile) and assemble feedback_report/1."""
    if not os.path.isdir(args.state):
        raise InputError(f"--state {args.state}: no such folder")
    if args.lab:
        if not glob.glob(os.path.join(args.lab, "scores_cv_*.parquet")):
            raise InputError(f"--lab {args.lab}: no scores_cv_*.parquet")
        if not os.path.isfile(args.lab_meta):
            raise InputError(f"--lab-meta {args.lab_meta}: no such file")
    cases, events = read_cases(args.cases, args.history, args.api)
    windows, models, notes = read_windows(args.state)
    outbox, more = read_outbox(args.state)
    notes += more
    index = case_index(cases, events, outbox)
    failures = failures_from(outbox, index, cases)
    links = ew_links(cases, index, outbox)
    lab = fe.lab_evaluation(iter_lab(args.lab, args.lab_meta)) if args.lab else None
    model = str(windows["model"].iloc[0]) if len(windows) else None
    entry = models.get(model) if model else None
    apply_thr = fe.lab_mean_thresholds(lab)
    field = fe.field_evaluation(windows, failures[failures["status"] != "rejected"], entry, links, apply_thr)
    confirmed = fe.field_evaluation(windows, failures[failures["status"] == "confirmed"], entry, links, apply_thr)
    if field is None:
        notes.append("창 기록 없음: 조기 경보 현장 평가는 해당 없음")
    else:
        notes += field["warnings"]

    guns = {}
    for g, grp in windows.groupby("gun"):
        first, last = grp["window_start"].min(), grp["window_start"].max() + pd.Timedelta(seconds=fe.WINDOW_S)
        guns[g] = {"first": _z(first), "last": _z(last), "windows": int(len(grp))}
    long_ = [g for g, v in guns.items()
             if pd.Timestamp(v["last"][:-1]) - pd.Timestamp(v["first"][:-1]) > pd.Timedelta(days=fe.OUTBOX_RETENTION_DAYS)]
    if long_:
        notes.append(f"창 기록이 30일보다 긴 설비({', '.join(long_)}): outbox는 30일만 보존하므로 케이스 기록이 없는 오래된 고장은 "
                     "빠졌을 수 있음")
    if args.history or args.api:  # GET /api/history leaves out the open cases
        notes.append(CLOSED_ONLY_KO)

    challengers = [challenger_entry(ch, lab, field) for ch in fe.CHALLENGERS]
    verdict_ko = {c["name"]: c["verdict_ko"] for c in challengers}

    ew_h = (field or {}).get("labels", {}).get("ew") or {"false_alarm": None, "pending": None, "acted": None,
                                                         "reasons": {}, "reason_guns": {}, "notes": [],
                                                         "disagreements": []}
    reasons = {r: {"count": ew_h["reasons"].get(r, 0),
                   "linked": {"challenger": LINKED[r], "verdict_ko": verdict_ko[LINKED[r]]} if r in LINKED else None,
                   "guns": ew_h["reason_guns"].get(r, []) if r == "센서 · 신호 문제" else [],
                   "notes": [x for x in ew_h["notes"] if x["reason"] == r]}
               for r in list(fe.REASONS) + [fe.REASON_NONE]}
    crit = critical_cases(cases)
    fl_labels = (field or {}).get("labels", {}).get("failures") or {"hit": None, "miss": None, "undetermined": None}
    report = {
        "schema": SCHEMA,
        "meta": {"generated_at": _z(pd.Timestamp(now).tz_convert("UTC").tz_localize(None)), "tag": args.tag,
                 "model": model, "guns": guns, "simulated": any(c["simulated"] for c in cases), "note_ko": NOTE_KO},
        "criteria": {k: {"value": getattr(fe, k), "source": fe.CONSTANT_SOURCES[k]} for k in CRITERIA},
        "inputs": {"state": args.state, "cases": args.cases, "history": args.history, "api": args.api,
                   "lab": args.lab, "lab_meta": args.lab_meta if args.lab else None,
                   "counts": {"windows": int(len(windows)), "outbox": len(outbox), "cases": len(cases),
                              "lab_runs": len(lab["runs"]) if lab else 0}},
        "labels": {
            "failures": {s: int((failures["status"] == s).sum()) for s in ("confirmed", "presumed", "rejected")}
            | fl_labels,
            "ew_onsets": {k: ew_h[k] for k in ("false_alarm", "pending", "acted")},
            "critical_cases": {o: sum(c.outcome == o for c in crit) for o in ("false_alarm", "real", "unknown")},
            "reasons": ew_h["reasons"], "disagreements": ew_h["disagreements"]},
        "early_warning": {
            "champion": {"lab": (lab or {}).get("champion"), "field": (field or {}).get("champion"),
                         "mismatch_windows": (field or {}).get("mismatch")},
            "reference": {"definition": fe.REFERENCE, "field_threshold": (field or {}).get("reference_threshold"),
                          "field_basis": (field or {}).get("reference_basis"), "lab": (lab or {}).get("reference"),
                          "field": (field or {}).get("reference")},
            "challengers": challengers,
            "confirmed_only": {"champion": (confirmed or {}).get("champion"),
                               "challengers": {k: v.get("summary") for k, v in
                                               ((confirmed or {}).get("challengers") or {}).items()}}},
        "critical": fe.profile_proposals(crit, read_profiles(args.state)),
        "reasons": reasons,
        "post_failure": (field or {}).get("post_failure"),
        "warnings": notes,
    }
    return _clean(report)


# ---- report.md (spec 8.2)
def _f2(x: Any) -> str:
    return "—" if x is None else f"{x:.2f}"


def _f3(x: Any) -> str:
    return "—" if x is None else f"{x:.3f}"


def _ci(ci: Any) -> str:
    return "—" if not ci else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _d(d: Any) -> str:
    return "—" if not d or d.get("D") is None else f"{d['D']:+.3f} {_ci(d['ci'])}"


def _recall_change(d: Any) -> str:
    if not d or d.get("recall_change") is None:
        return "—"
    return f"{d['recall_change']:+.3f} {_ci(d['recall_change_ci'])}"


FIELD_BASIS_KO = {"served": "지금 임계값", "lab_mean": "실험실 평균, 적용 명령이 만들 값의 추정",
                  "field_quantile": "현장 분위수, 적용 명령이 만들 값 아님"}
EW_TABLE = ("| 규칙 | 출처 | 오알람/건·일 | 적중 | 우연 | 초과분 [95% CI] | 적중 변화(지금 규칙 대비) [95% CI] "
            "| D(지금 규칙) [95% CI] | D(기준점) [95% CI] | 판정 |", "|---|---|---|---|---|---|---|---|---|---|")


def _row(name: str, source: str, s: Any, dc: Any = None, dr: Any = None, verdict: str = "") -> str:
    if not s:
        return f"| {name} | {source} | — | — | — | — | — | — | — | {verdict} |"
    ex = "—" if s.get("excess") is None else f"{s['excess']:+.3f} {_ci(s['ci'])}"
    return (f"| {name} | {source} | {_f2(s['fa_per_day'])} | {_f3(s.get('recall'))} | {_f3(s.get('chance'))} | {ex} "
            f"| {_recall_change(dc)} | {_d(dc)} | {_d(dr)} | {verdict} |")


def render_md(rep: dict) -> str:
    """report.json -> report.md (Korean, spec 8.2 order); every number comes from the json."""
    m, ew, lab_used = rep["meta"], rep["early_warning"], rep["inputs"]["lab"] is not None
    out = ["# ML 피드백 루프 보고서 (D-9)", ""]
    if m["simulated"]:
        out += [f"> {SIMULATED_KO}", ""]
    out += ["## 1. 머리말", "", f"- 만든 시각: {m['generated_at']}", f"- {m['note_ko']}",
            f"- 모델: {m['model'] or '없음(창 기록 없음)'}"]
    inp = rep["inputs"]
    given = [f"{k} `{inp[k]}`" for k in ("state", "cases", "history", "api", "lab", "lab_meta") if inp[k]]
    out += [f"- 입력: {', '.join(given)}", f"- 읽은 수: 창 {inp['counts']['windows']}개, outbox {inp['counts']['outbox']}건, "
            f"케이스 {inp['counts']['cases']}건, 실험실 실행 {inp['counts']['lab_runs']}개", ""]
    if m["guns"]:
        out += ["| 설비 | 처음 | 끝 | 창 |", "|---|---|---|---|"]
        out += [f"| {g} | {v['first']} | {v['last']} | {v['windows']} |" for g, v in m["guns"].items()] + [""]
    out += ["| 상수 | 값 | 출처 |", "|---|---|---|"]
    out += [f"| `{k}` | {v['value']} | {v['source']} |" for k, v in rep["criteria"].items()] + [""]
    if rep["warnings"]:
        out += ["경고:", ""] + [f"- {w}" for w in rep["warnings"]] + [""]

    out += ["## 2. 한눈에 보기", ""]
    for code, ko in VERDICT_ORDER:
        names = [c["name"] for c in ew["challengers"] if c["verdict"] == code]
        out.append(f"- {ko}: {', '.join(names) if names else '없음'}")
    out += [f"- critical 프로필 제안: {len(rep['critical']['proposals'])}건", ""]

    lb = rep["labels"]
    fa = lb["failures"]
    out += ["## 3. 라벨 현황", "",
            f"- 고장: 확인 {fa['confirmed']} · 추정 {fa['presumed']} · 거부 {fa['rejected']} "
            f"(지금 규칙 기준 적중 {fa['hit'] if fa['hit'] is not None else '—'} · 미탐 "
            f"{fa['miss'] if fa['miss'] is not None else '—'} · 판정 안 함 "
            f"{fa['undetermined'] if fa['undetermined'] is not None else '—'})"]
    on = lb["ew_onsets"]
    out.append("- 조기 경보 시작: " + (f"오알람 {on['false_alarm']} · 대기 {on['pending']} · 조치함 {on['acted']}"
                                  if on["false_alarm"] is not None else "해당 없음(창 기록 없음)"))
    cc = lb["critical_cases"]
    out.append(f"- critical 케이스: 오알람 {cc['false_alarm']} · 실제 {cc['real']} · 미확인 {cc['unknown']}")
    out.append("- 사유 분포: " + (", ".join(f"{r} {n}" for r, n in lb["reasons"].items()) or "없음"))
    out.append("- 사람-자동 불일치: " + (", ".join(f"{d['case_id']}({d['gun']}, {d['onset_at']})"
                                          for d in lb["disagreements"]) or "없음"))
    out.append("")

    out += ["## 4. 조기 경보", "",
            f"기준점: {ew['reference']['definition']} (현장 임계값 {_f3(ew['reference']['field_threshold'])}, "
            f"근거 {ew['reference']['field_basis'] or '—'}). 서빙 판정과 다시 계산한 지금 규칙이 다른 창: "
            f"{ew['champion']['mismatch_windows'] if ew['champion']['mismatch_windows'] is not None else '—'}", "",
            *EW_TABLE]
    if lab_used:
        out.append(_row("지금 규칙", "실험실", ew["champion"]["lab"]))
    out.append(_row("지금 규칙", "현장", ew["champion"]["field"]))
    out.append(_row("기준점", "현장", ew["reference"]["field"]))
    for c in ew["challengers"]:
        for src, key, v in (("실험실", "lab", c["lab_verdict"]), ("현장", "field", c["field_verdict"])):
            r = c[key] or {}
            if src == "실험실" and not lab_used:
                continue
            out.append(_row(c["name"], src, r.get("summary"), r.get("d_champion"), r.get("d_reference"),
                            fe.VERDICT_KO[v]))
    out.append("")
    for c in ew["challengers"]:
        th, basis = c["thresholds"], (c["field"] or {}).get("threshold_basis")
        shown = "" if th["field"] is None else (f". 임계값: 지금 {_f3(th['served'])} · 현장 평가 {_f3(th['field'])}"
                                                f"({FIELD_BASIS_KO[basis]})")
        if shown and th["lab_mean"] is not None and basis != "lab_mean":
            shown += f" · 실험실 평균 {_f3(th['lab_mean'])}"
        out.append(f"- **{c['name']}**: {c['verdict_ko']} (실험실 {fe.VERDICT_KO[c['lab_verdict']]}, 현장 "
                   f"{fe.VERDICT_KO[c['field_verdict']]}). 적용 방법: {c['apply_ko']}{shown}")
        if (c["field"] or {}).get("against_apply"):
            low = th["field"] < th["served"]
            out.append(f"  - 현장 임계값은 지금보다 {'낮은데' if low else '높은데'} 적용 명령은 임계값을 "
                       f"{'올린다' if low else '내린다'}. 이 줄의 현장 수치는 적용 결과가 아니다.")
    co = ew["confirmed_only"]
    out += ["", "확인된 고장만으로 한 현장 결과(판정 없음):", "", *EW_TABLE, _row("지금 규칙", "현장(확인)", co["champion"])]
    out += [_row(k, "현장(확인)", v) for k, v in co["challengers"].items()] + [""]

    cr = rep["critical"]
    out += ["## 5. critical 프로필", ""]
    for p in cr["proposals"]:
        cases = ", ".join(f"{x['case_id']}({x['date']})" for x in p["cases"])
        body = json.dumps(p["body"], ensure_ascii=False)
        out += [f"- **{p['gun']}**: {', '.join(p['exclude'])} 빼기. 근거 케이스 {cases}", f"  - {p['risk_ko']}",
                f"  - 적용: `PUT /guns/{p['gun']}/profile` 본문 `{body}`"]
    if not cr["proposals"]:
        out.append("- 제안 없음")
    out += [f"- {w['gun']} {w['cls']}: {w['text']}" for w in cr["watching"]]
    out += [f"- {r['gun']} {', '.join(r['exclude'])}: {r['text']} ({r.get('why', '프로필이 비게 됨')})"
            for r in cr["review"]] + [""]

    out += ["## 6. 사유별 메모", "", "| 사유 | 건수 | 연결된 판정 | 점검 권고 설비 | 메모 |", "|---|---|---|---|---|"]
    for r, v in rep["reasons"].items():
        link = f"{v['linked']['challenger']}: {v['linked']['verdict_ko']}" if v["linked"] else "—"
        memo = "; ".join(f"{x['case_id']} {x['note']}" for x in v["notes"]) or "—"
        out.append(f"| {r} | {v['count']} | {link} | {', '.join(v['guns']) or '—'} | {memo} |")
    out.append("")

    out += ["## 7. 고장 뒤 조기 경보 진단", ""]
    if rep["post_failure"]:
        out += ["| 고장 뒤(분) | 창 | 조기 경보 비율 |", "|---|---|---|"]
        out += [f"| {b['from_min']}–{b['to_min']} | {b['windows']} | {_f3(b['ew_share'])} |" for b in rep["post_failure"]]
    else:
        out.append("- 창 기록 없음")
    out.append("")

    judged = (ew["champion"]["field"] or {}).get("failures")
    out += ["## 8. 한계", "",
            f"- 판정된 현장 고장 {judged if judged is not None else 0}건, 조치함으로 뺀 시작 "
            f"{on['acted'] if on['acted'] is not None else 0}건. 조치함은 지금 규칙의 알람 구간 기준으로 모든 규칙에서 같게 뺐다.",
            "- 분위수 임계값은 그 임계값을 정한 같은 현장 창으로 평가했다. 오알람 수가 조금 좋게 나온다(B-20과 같은 방식).",
            "- 현장의 Q+ · Q-는 현장 정상 창의 분위수다. 적용 명령(학습 데이터로 다시 학습)이 만드는 임계값과 다르다. 적용 방법 줄의 임계값을 같이 본다.",
            "- D는 초과분(적중 − 우연)의 차이다. 우연이 같이 줄면 적중이 줄어도 D는 0 근처에 남는다. 놓침은 '적중 변화' 열로 본다.",
            "- 현장 판정력이 낮다. 고장 64건을 모아도 실행 하나로는 GATE만 한 차이를 39%만 잡는다(스펙 3.2).",
            "- 11.1 이전 api DB에서는 코드 없는 알림(조기 경보, 모델 단독 critical)이 설비마다 한 케이스에 묶여 있을 수 있다.",
            "- 프로필 밖 코드는 사람에게 가지 않는다. 그래서 프로필이 틀려도 루프는 알아챌 수 없다.",
            "- 종료 코드 상태가 길게 이어지는 동안의 조기 경보 시작도 오알람으로 센다(B-18과 같음).",
            "- critical의 정답은 사람 판정뿐이다. \"실제\"는 사람이 해결 종료했다는 뜻이지 검증됐다는 뜻이 아니다.", ""]
    return "\n".join(out)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--state", required=True, help="the ML server's state dir (RSW_STATE_DIR)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cases", help="the api's SQLite file (read only)")
    src.add_argument("--history", help="a saved GET /api/history response")
    src.add_argument("--api", help="the api's address, e.g. http://127.0.0.1:9000")
    ap.add_argument("--lab", help="B-19 cache dir (scores_cv_*.parquet); default: no lab evaluation")
    ap.add_argument("--lab-meta", default=os.path.join(PROJECT_ROOT, "results", "B-17", "cache", "meta.csv"))
    ap.add_argument("--tag", help="report folder name; default: UTC YYYYMMDD-HHMM")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "D-9", "runs"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(timezone.utc)
    args.tag = args.tag or now.strftime("%Y%m%d-%H%M")
    try:
        report = build_report(args, now)
    except InputError as exc:
        print(f"feedback_loop: {exc}", file=sys.stderr)
        return 2
    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_md(report))
    print(f"feedback_loop: wrote {os.path.join(out, 'report.md')} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
