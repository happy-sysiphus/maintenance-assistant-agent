"""Validate the event and restore the observed samples from the ML handoff."""

import re

from .warning_input import (CHANNEL_NAMES, STATE_CHANNELS, clock_seconds, expand_channel,
                            required_text, timestamp)

UNIDENTIFIED_CHANNEL = "c19"
BINARY_CHANNELS = STATE_CHANNELS | {"c17"}
MIN_OBSERVED = 3


def observed_code(handoff):
    trigger = handoff.get("trigger") or {}
    explicit = handoff.get("observed_error")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise ValueError("observed_error must contain code and observed_at")
        code, time = explicit.get("code"), explicit.get("observed_at")
    else:
        if not isinstance(trigger, dict):
            raise ValueError("trigger must be an object")
        code, time = trigger.get("rule_code"), trigger.get("rule_trigger_time")
        if not code or not time:
            return None
    code = required_text(code, "observed_error.code", 4).upper()
    if not re.fullmatch(r"E[0-9]{3}", code):
        raise ValueError("observed_error.code must be an E code")
    return {"code": code, "observed_at": timestamp(time, "observed_error.observed_at").isoformat()}


def restore_series(block):
    if block is None:
        return {}
    if not isinstance(block, dict) or not isinstance(block.get("channels"), dict):
        raise ValueError("sensor_features must contain a channels object")
    end = timestamp(block.get("window_end"), "sensor_features.window_end")
    seconds = clock_seconds(block, end)
    missing = block.get("missing_s", [])
    if not isinstance(missing, list) or any(type(s) is not int or s not in seconds for s in missing):
        raise ValueError("invalid missing_s")
    result = {}
    for channel in sorted(block["channels"].keys() & CHANNEL_NAMES.keys()):
        observed = expand_channel(channel, block["channels"][channel], seconds, set(missing),
                                  state_channels=BINARY_CHANNELS)
        # Align offset zero with the end of the sensor window.
        result[channel] = {i - len(seconds) + 1: v for i, v in observed.items()}
    return result


def parse_event(handoff, scope):
    if not isinstance(handoff, dict) or handoff.get("schema_version") != "1.4":
        raise ValueError("expected a v1.4 sensor handoff")
    block = handoff.get("sensor_features")
    raw_error = observed_code(handoff)
    warning = handoff.get("early_warning") or {}
    if not isinstance(warning, dict):
        raise ValueError("early_warning must be an object")
    onset = warning.get("onset_at")
    series = restore_series(block)
    detected = onset or handoff.get("detected_at") or (block or {}).get("window_end")
    if detected is None and raw_error:
        detected = raw_error["observed_at"]
    times = [timestamp(detected, "event time")]
    if raw_error:
        times.append(timestamp(raw_error["observed_at"], "observed_error.observed_at"))

    features = handoff.get("contributing_features", [])
    if (not isinstance(features, list) or len(features) > 3
            or any(not isinstance(f, dict) or not isinstance(f.get("sensor"), str) for f in features)):
        raise ValueError("contributing_features must contain up to three objects")
    episode_id = handoff.get("episode_id")
    gun_id = handoff.get("gun_id")
    if onset and block and timestamp(onset, "onset_at") != timestamp(block["window_end"], "window_end"):
        raise ValueError("early warning sensor_features must end at onset_at")
    event = {
        "event_id": required_text(handoff.get("event_id"), "event_id"),
        "episode_id": required_text(episode_id, "episode_id") if episode_id is not None else None,
        "gun_id": required_text(gun_id, "gun_id") if gun_id is not None else None,
        "event_at": max(times).isoformat(),
        "equipment_scope": scope,
        "observed_error": raw_error,
        "sensor_features": block,
        "contributing_features": features,
    }
    return event, series


def observations(series):
    result = []
    for channel, observed in series.items():
        samples = list(observed.items())
        if not samples:
            continue
        values = [v for _, v in samples]
        changes = [{"before_offset_s": a, "after_offset_s": b, "from": x, "to": y}
                   for (a, x), (b, y) in zip(samples, samples[1:]) if x != y]
        name = "unidentified sensor" if channel == UNIDENTIFIED_CHANNEL else CHANNEL_NAMES[channel]
        result.append({"channel": channel, "name": name,
                       "observed_seconds": len(values), "first": values[0], "last": values[-1],
                       "min": min(values), "max": max(values),
                       "change_count": len(changes), "changes": changes[:3]})
    return result


def search_query(observed, code=None):
    lines = [code] if code else []
    for row in observed:
        if row["channel"] == UNIDENTIFIED_CHANNEL:
            continue
        change = "transition" if row["change_count"] else "constant"
        if row["first"] != row["last"]:
            change += " increase" if row["first"] < row["last"] else " decrease"
        lines.append(f"{row['name']} {change}")
    return "\n".join(lines)


def compare_series(current, previous):
    """Rank by aligned, per-channel distance and penalize missing coverage."""
    comparisons = []
    for channel, observed in current.items():
        if len(observed) < MIN_OBSERVED:
            continue
        past = previous.get(channel, {})
        offsets = sorted(observed.keys() & past.keys())
        if len(offsets) < max(MIN_OBSERVED, (len(observed) + 1) // 2):
            continue
        pairs = [(observed[i], past[i]) for i in offsets]
        if channel in BINARY_CHANNELS:
            distance = sum(x != y for x, y in pairs) / len(pairs)
        else:
            scale = max(abs(v) for pair in pairs for v in pair) or 1
            distance = sum(abs(x / scale - y / scale) / 2 for x, y in pairs) / len(pairs)
        coverage = len(offsets) / len(observed)
        comparisons.append({"channel": channel, "overlap_seconds": len(offsets), "offsets": offsets,
                            "coverage": coverage, "distance": distance})
    if not comparisons:
        return None
    score = sum(c["coverage"] * (1 - c["distance"]) for c in comparisons) / len(current)
    return score, comparisons
