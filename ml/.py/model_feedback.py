"""
D-12 operator verdicts for the model loop (spec: docs/superpowers/specs/2026-11-12-feedback-labels-design.md).

The api hands out the facts of every closed early-warning case (GET /api/ml/feedback). This module keeps them in
<registry>/feedback.json and reads one verdict out of each case:

    confirmed    closed as resolved, with at least one check marked abnormal  -> its span is not normal data
    false_alarm  closed as a false alarm                                      -> recorded (the data says so already)
    unknown      resolved without an abnormal check                           -> recorded, not used
    unmatched    the warning belongs to no preprocessed file                  -> recorded, not used

One warning per case: the one the operator selected, else the representative one. The span starts at that warning's
onset (onset_at when the api passes it, else window_start; data time, naive UTC like the preprocessed files) and lasts
SPAN_S, the time the server waits before it sends a gun's early warning again. train.py floors it to its window. The
Aura link of a case (resolution_status) is no condition.

Nothing here trains or judges: model_loop.tick gets a function (make_feedback_fn) that turns a list of files into the
spans and counts of the verdicts on them, and passes the spans on to train.py --feedback.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import pandas as pd

SPAN_S = 3600
FILE = "feedback.json"
SCHEMA = 1
VERDICTS = ("confirmed", "false_alarm", "unknown", "unmatched")

Ranges = dict[str, tuple[pd.Timestamp, pd.Timestamp]]
FeedbackFn = Callable[[list[str]], "dict[str, Any] | None"]


class FeedbackError(ValueError):
    """The verdict store cannot be read, or the api gave no usable answer. Nothing was changed."""


def naive_utc(s: Any) -> pd.Timestamp:
    """"...Z", "...+09:00" or no zone at all -> naive UTC (what manifest.csv and the preprocessed files hold)."""
    t = pd.Timestamp(s)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def load_ranges(data_dir: str) -> Ranges:
    """file -> (start, end) of every preprocessed file that was kept (manifest.csv)."""
    path = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(path):
        raise ValueError(f"{path} not found - run preprocess.py first")
    m = pd.read_csv(path, dtype={"file": str, "dropped": str})
    kept = m[(m["dropped"].isna() | (m["dropped"].str.strip() == "")) & m["start"].notna() & m["end"].notna()]
    return {f: (naive_utc(s), naive_utc(e)) for f, s, e in zip(kept["file"], kept["start"], kept["end"])}


# ------------------------------------------------------------- the verdict rule (spec 4)
def _events(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in (item.get("events") or []) if isinstance(e, dict)]


def _representative(item: dict[str, Any]) -> dict[str, Any] | None:
    return next((e for e in _events(item) if e.get("representative")), None)


def _chosen(item: dict[str, Any]) -> dict[str, Any] | None:
    """The warning the verdict is about: the selected one, else the representative one."""
    sel = item.get("selected_event_id")
    return next((e for e in _events(item) if sel and e.get("event_id") == sel), None) or _representative(item)


def _warning_time(ev: dict[str, Any]) -> pd.Timestamp | None:
    """Data time of the warning: the onset the server reports (early_warning.onset_at, the last second of the window
    in which the run was reached) when the api passes it, else the handoff's window_start. A request that holds
    several windows names the latest one as window_start, which can lie after the onset. Either is a row time:
    train.py floors the span to its window, so the warning's own window is the first one of the span."""
    for key in ("onset_at", "window_start"):
        try:
            t = naive_utc(ev[key]) if ev.get(key) else None
        except (ValueError, TypeError):
            continue
        if t is not None and not pd.isna(t):
            return t
    return None


def _abnormal(item: dict[str, Any]) -> int:
    """checks.abnormal as a count. Anything that is no count (a store edited by hand) is no abnormal check: the case
    then reads "unknown" instead of stopping the loop."""
    checks = item.get("checks")
    try:
        return int((checks.get("abnormal") if isinstance(checks, dict) else 0) or 0)
    except (TypeError, ValueError):
        return 0


def verdict(item: dict[str, Any], ranges: Ranges, span_s: int = SPAN_S) -> dict[str, Any]:
    """One closed case (an item of GET /api/ml/feedback) -> {"verdict", "event_id", "file", "start", "end"}.
    file / start / end are None for an unmatched case; start and end are naive ISO strings, end exclusive."""
    ev = _chosen(item)
    out = {"verdict": "unmatched", "event_id": (ev or {}).get("event_id"), "file": None, "start": None, "end": None}
    t = _warning_time(ev) if ev and ev.get("gun_id") else None
    if t is None:
        return out
    span = ranges.get(str(ev["gun_id"]))
    if span is None or not span[0] <= t <= span[1]:
        return out
    outcome = item.get("outcome")
    if outcome == "false_alarm":
        v = "false_alarm"
    elif outcome == "resolved" and _abnormal(item) >= 1:
        v = "confirmed"
    else:
        v = "unknown"
    return {**out, "verdict": v, "file": str(ev["gun_id"]), "start": t.isoformat(),
            "end": (t + pd.Timedelta(seconds=span_s)).isoformat()}


def collect(data: dict[str, Any] | None, ranges: Ranges, names: list[str], span_s: int = SPAN_S) -> dict[str, Any] | None:
    """The verdicts on these files: spans in train.py's --feedback layout, counts per verdict and the warnings whose
    spans are confirmed. confirmed / false_alarm / unknown count the cases on `names`; unmatched counts the whole
    store (such a case belongs to no file). None without a store: the loop then runs as it did before D-12."""
    if data is None:
        return None
    want = set(names)
    spans: dict[str, dict[str, list[list[str]]]] = {"confirmed": {}, "false_alarm": {}}
    counts = {v: 0 for v in VERDICTS}
    ids: list[str] = []
    rows = [verdict(it, ranges, span_s) for it in (data.get("cases") or {}).values()]
    for v in sorted(rows, key=lambda r: (r["start"] or "", r["event_id"] or "")):
        if v["verdict"] == "unmatched":
            counts["unmatched"] += 1
        elif v["file"] in want:
            counts[v["verdict"]] += 1
            if v["verdict"] in spans:
                spans[v["verdict"]].setdefault(v["file"], []).append([v["start"], v["end"]])
            if v["verdict"] == "confirmed":
                ids.append(v["event_id"])
    return {"spans": spans, "counts": counts, "event_ids": ids}


# ------------------------------------------------------------- the store (spec 6.1)
class Store:
    """<registry>/feedback.json: the cases as the api sent them, keyed by their representative warning (a case
    number like CASE-0001 can repeat between api databases). Cases are added and replaced, never dropped."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.path = os.path.join(self.root, FILE)

    def load(self) -> dict[str, Any] | None:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as e:
            raise FeedbackError(f"cannot read {self.path}: {e}") from e
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA or not isinstance(data.get("cases"), dict):
            raise FeedbackError(f"{self.path} is not a schema-{SCHEMA} verdict store")
        odd = [key for key, case in data["cases"].items() if not isinstance(case, dict)]
        if odd:
            raise FeedbackError(f"{self.path}: case {odd[0]!r} is not an object (the store was changed by hand?)")
        return data

    def merge(self, items: list[dict[str, Any]], source: str, at: str) -> dict[str, int]:
        """Add the items of one pull. A case that is there already is replaced when it differs (it was closed again);
        an item without a representative warning is skipped. Temp file + rename."""
        data = self.load() or {"schema_version": SCHEMA, "source": None, "pulled_at": None, "cases": {}}
        n = {"new": 0, "changed": 0, "same": 0, "skipped": 0}
        for it in items:
            rep = _representative(it) if isinstance(it, dict) else None
            key = (rep or {}).get("event_id")
            if not key:
                n["skipped"] += 1
                continue
            old = data["cases"].get(key)
            if old is not None and {k: v for k, v in old.items() if k != "received_at"} == it:
                n["same"] += 1
                continue
            n["new" if old is None else "changed"] += 1
            data["cases"][key] = {**it, "received_at": at}
        data["source"], data["pulled_at"] = source, at
        os.makedirs(self.root, exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
            for attempt in range(5):  # Windows: a reader holding feedback.json makes the rename fail briefly
                try:                  # (the same wait as model_registry.Registry.save)
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return n


def fetch(api_url: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    """The items of GET <api_url>/api/ml/feedback (schema 1.x)."""
    import httpx  # only a pull needs it

    url = api_url.rstrip("/") + "/api/ml/feedback"
    try:
        r = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as e:
        raise FeedbackError(f"no answer from {url}: {e}") from e
    if r.status_code != 200:
        raise FeedbackError(f"{url} answered {r.status_code}")
    try:
        body = r.json()
    except ValueError as e:
        raise FeedbackError(f"{url} did not answer with JSON: {e}") from e
    if not isinstance(body, dict) or not str(body.get("schema_version", "")).startswith("1.") \
            or not isinstance(body.get("items"), list):
        raise FeedbackError(f"{url} did not answer with schema 1.x items "
                            f"(schema_version {body.get('schema_version') if isinstance(body, dict) else None!r})")
    return body["items"]


def make_feedback_fn(root: str, data_dir: str, span_s: int = SPAN_S) -> FeedbackFn:
    """What model_loop.tick calls with the files of a training or a judgement. The store and the manifest are read
    at every call, so a pull between two ticks is seen by the next one."""
    store = Store(root)

    def feedback(names: list[str]) -> dict[str, Any] | None:
        data = store.load()
        return None if data is None else collect(data, load_ranges(data_dir), names, span_s)

    return feedback
