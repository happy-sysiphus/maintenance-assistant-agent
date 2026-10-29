"""
ML per-minute sensor document (schema sensor-features-v4, docs/sensor_features.schema.json). Not pushed in
operation (RAG declined it, history.md 37); /predict and GET /guns/{id}/sensor-features still serve it.

Decided 2026-10-21 (techspec D-6, WORK_LOG D-004; history.md 33-34). ML sends measured facts, never interpretations
(no situation / symptom mapping, no model score). Every scored minute, three groups:

  measurements  c1 c2 c3 c5 c6 (what the gun measures / reports) x mean / std = 10 features, ALL of them, sorted by
                |z| (largest first). Each: z = round(window feature - model normal reference, 3) - the trace / handoff
                value, NOT a standard-normal z-score - and the raw value (the minute's mean / std of the CSV numbers).
  states        c4 c10 c18 - 0/1 signals in the data (c4, c18 = 1 while the gun presses, c10 = US2 enable on): the share
                of the minute's seconds they were on (raw, 0..1). Both 0 + c10 0 = the gun stood still: ignore the z.
  setpoints     c13 c14 c15 c16 (what the controller is told): mean / std of the minute, raw and z. Setpoints change per
                weld spot (one program per spot), so a setpoint z is "another program mix", never a fault signal.
  dropped       c7 c8 c9 c17 (constant per gun), c11 c12 c19 (counters)

Plus error_code (the minute's last non-zero controller code), fault_class (E01..E04 when it is a terminal code - the
data mapping of readme 5.3.4) and gun_norm ("warming_up" in the gun's first hours: z against the global reference).

Pure Python (no numpy / pandas / pydantic).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

SCHEMA_VERSION = "sensor-features-v4"
VALUE_SPACE = "ml_handoff_z"
WINDOW_S = 60
MEASURED = ("c1", "c2", "c3", "c5", "c6")
MEASURED_FEATURES = tuple(f"{s}_{stat}" for s in MEASURED for stat in ("mean", "std"))
STATES = ("c4", "c10", "c18")
SETPOINTS = ("c13", "c14", "c15", "c16")
DROPPED = ("c7", "c8", "c9", "c17", "c11", "c12", "c19")
# terminal code -> fault class (data mapping, readme 5.3.4)
TERMINAL_TO_CLASS = {"E012": "E01", "E016": "E02", "E028": "E03", "E029": "E04"}


def utc_iso(t: Any) -> str:
    """Datetime / Timestamp / str -> ISO 8601 UTC with Z, second precision (naive = UTC, as in main.py)."""
    s = t.isoformat() if hasattr(t, "isoformat") else str(t)
    s = s.replace(" ", "T").split(".")[0].removesuffix("+00:00")
    return s if s.endswith("Z") else s + "Z"


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, str):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _mean_std(vals: list[float]) -> tuple[float, float]:
    """Mean and sample std (ddof=1, as pandas / train.window_features); std 0 for a single value."""
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _on(v: Any) -> float | None:
    """0/1 signal -> 1.0 / 0.0 (c10 arrives as 'on' / 'off', True / False or 1 / 0)."""
    if isinstance(v, str):
        s = v.strip().lower()
        return 1.0 if s in ("on", "true", "1") else 0.0 if s in ("off", "false", "0") else None
    f = _num(v)
    return None if f is None else (1.0 if f > 0 else 0.0)


def raw_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Raw rows of one minute -> {sensor: (mean, std)} for the measured sensors and setpoints, {state: on share}.
    Kept per scored minute (main.py trace) so a document can be rebuilt after the reading buffer moved on."""
    rows = list(rows)
    out: dict[str, Any] = {"n": len(rows), "stats": {}, "on": {}}
    for c in MEASURED + SETPOINTS:
        vals = [f for f in (_num(r.get(c)) for r in rows) if f is not None]
        if vals:
            out["stats"][c] = _mean_std(vals)
    for c in STATES:
        vals = [f for f in (_on(r.get(c)) for r in rows) if f is not None]
        if vals:
            out["on"][c] = sum(vals) / len(vals)
    return out


def last_code(codes: Sequence[Any]) -> str | None:
    """The last non-zero controller code of the minute, else None."""
    for c in reversed(list(codes)):
        s = str(c)
        if s not in ("0", "", "nan", "None"):
            return s
    return None


def _r(v: float | None, nd: int = 3) -> float | None:
    return None if v is None or math.isnan(v) else round(float(v), nd)


def build_doc(gun_id: str, observed_at: Any, z_by_feature: dict[str, float], raw: dict[str, Any],
              error_code: str | None, gun_norm: str, reference_id: str | None = None) -> dict[str, Any]:
    """One minute -> the sensor-features-v4 document. z_by_feature: window feature - reference for (at least) the
    measured and setpoint features; raw: raw_summary() of the minute's rows."""
    stats, on = raw.get("stats", {}), raw.get("on", {})

    def raw_value(feature: str) -> float | None:
        sensor, stat = feature.rsplit("_", 1)
        if sensor not in stats:
            return None
        return _r(stats[sensor][0 if stat == "mean" else 1])

    measurements = []
    for i, f in enumerate(MEASURED_FEATURES):
        z = _r(z_by_feature.get(f))
        measurements.append({"feature": f, "z": z, "value": raw_value(f), "_order": i})
    # largest |z| first; a missing z last; ties in catalog order
    measurements.sort(key=lambda m: (m["z"] is None, -abs(m["z"] or 0.0), m["_order"]))
    for m in measurements:
        del m["_order"]
    setpoints = []
    for c in SETPOINTS:
        setpoints.append({"setpoint": c, "mean": raw_value(f"{c}_mean"), "std": raw_value(f"{c}_std"),
                          "mean_z": _r(z_by_feature.get(f"{c}_mean")), "std_z": _r(z_by_feature.get(f"{c}_std"))})
    return {
        "schema_version": SCHEMA_VERSION,
        "gun_id": gun_id,
        "observed_at": utc_iso(observed_at),
        "window_s": WINDOW_S,
        "value_space": VALUE_SPACE,
        "reference_id": reference_id,
        "gun_norm": gun_norm,
        "error_code": error_code,
        "fault_class": TERMINAL_TO_CLASS.get(error_code or ""),
        "n_samples": int(raw.get("n", 0)),
        "measurements": measurements,
        "states": [{"signal": c, "on_share": _r(on.get(c))} for c in STATES],
        "setpoints": setpoints,
    }
