"""Interpret ML's v1.4 onset window without judging normality."""
from datetime import datetime, timedelta
from math import isfinite
from statistics import median

PATTERN_VERSION = "direction-v1"
CHANNEL_NAMES = {
    "c1": "electrode cap offset", "c2": "electrode force", "c3": "electrode position",
    "c4": "force reached", "c5": "balance pressure", "c6": "friction",
    "c7": "maximum aperture", "c8": "maximum electrode force", "c9": "start friction",
    "c10": "US2 operating enable", "c11": "weld count", "c12": "position count",
    "c13": "balance pressure setpoint", "c14": "electrode force setpoint",
    "c15": "electrode position setpoint", "c16": "sheet thickness setpoint",
    "c17": "velocity setpoint", "c18": "close gun command", "c19": "robot offset counter",
}
STATE_CHANNELS = {"c4", "c10", "c18"}


def required_text(value, field, limit=300):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{field} must be nonempty text (max {limit} characters)")
    return value.strip()


def timestamp(value, field):
    result = datetime.fromisoformat(required_text(value, field, 50).replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return result


def clock_seconds(block, onset):
    start = timestamp(block.get("window_start"), "sensor_features.window_start")
    end = timestamp(block.get("window_end"), "sensor_features.window_end")
    if end != onset or start.microsecond or end.microsecond:
        raise ValueError("sensor_features must describe the whole-second onset window")
    duration = (end - start).total_seconds()
    if not 2 <= duration <= 59:
        raise ValueError("onset window must contain 3 to 60 seconds")
    return [(start + timedelta(seconds=i)).second for i in range(int(duration) + 1)]


def expand_channel(channel, changes, seconds, missing, *, state_channels=STATE_CHANNELS):
    if not isinstance(changes, list):
        raise ValueError(f"{channel}: expected a list of change points")
    positions = {second: index for index, second in enumerate(seconds)}
    indexed = {}
    for point in changes:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{channel}: expected [second, value]")
        second, value = point
        if (type(second) is not int or second not in positions or second in missing
                or type(value) not in (int, float) or not isfinite(value)
                or (channel in state_channels and value not in (0, 1))):
            raise ValueError(f"{channel}: invalid change point")
        index = positions[second]
        if index in indexed:
            raise ValueError(f"{channel}: duplicate change point")
        indexed[index] = value
    observed = {}
    value = None
    for index, second in enumerate(seconds):
        value = indexed.get(index, value)
        if value is not None and second not in missing:
            observed[index] = value
    return observed


def channel_observation(channel, changes, seconds, missing):
    observed = expand_channel(channel, changes, seconds, missing)
    third = len(seconds) // 3
    first = [v for i, v in observed.items() if i < third]
    last = [v for i, v in observed.items() if i >= len(seconds) - third]
    before = median(first) if len(first) * 2 >= third else None
    after = median(last) if len(last) * 2 >= third else None
    direction = None
    if before is not None and after is not None:
        if channel not in STATE_CHANNELS or (before in (0, 1) and after in (0, 1)):
            direction = "up" if after > before else "down" if after < before else None

    transitions = []
    if channel in STATE_CHANNELS:
        pairs = list(observed.items())
        for (left, a), (right, b) in zip(pairs, pairs[1:]):
            if a != b:
                transitions.append({"last_before_offset_s": left, "first_after_offset_s": right,
                                    "from": a, "to": b})
    return {"channel": channel, "name": CHANNEL_NAMES[channel], "direction": direction,
            "first_median": before, "last_median": after, "observed_seconds": len(observed),
            "first_samples": len(first), "last_samples": len(last), "transitions": transitions}


def analyze_warning(handoff):
    """Only three distinct directional channels form a reusable pattern."""
    if (not isinstance(handoff, dict) or handoff.get("event_type") != "early_warning"
            or handoff.get("schema_version") != "1.4"):
        raise ValueError("expected an early_warning handoff with schema_version 1.4")
    event_id = required_text(handoff.get("event_id"), "event_id")
    gun_id = handoff.get("gun_id")
    if gun_id is not None:
        gun_id = required_text(gun_id, "gun_id")
    info = handoff.get("early_warning")
    if not isinstance(info, dict):
        raise ValueError("early_warning details are required")
    onset = timestamp(info.get("onset_at"), "early_warning.onset_at")
    features = handoff.get("contributing_features", [])
    if not isinstance(features, list) or len(features) > 3:
        raise ValueError("contributing_features must contain up to three entries")
    channels, context = [], []
    for item in features:
        if not isinstance(item, dict):
            raise ValueError("invalid contributing feature")
        signal = required_text(item.get("sensor"), "feature.sensor")
        name = required_text(item.get("feature"), "feature.feature")
        if signal in CHANNEL_NAMES:
            if signal not in channels:
                channels.append(signal)
        elif name not in context:
            context.append(name)

    observations = []
    block = handoff.get("sensor_features")
    if block is not None:
        if not isinstance(block, dict) or not isinstance(block.get("channels"), dict):
            raise ValueError("invalid sensor_features block")
        seconds = clock_seconds(block, onset)
        missing = block.get("missing_s", [])
        if not isinstance(missing, list) or any(type(s) is not int or s not in seconds for s in missing):
            raise ValueError("invalid missing_s")
        observations = [channel_observation(c, block["channels"].get(c, []), seconds, set(missing))
                        for c in sorted(channels)]
    pairs = [f"{o['channel']}:{o['direction']}" for o in observations if o["direction"]]
    return {
        "event_id": event_id, "gun_id": gun_id, "onset_at": onset.isoformat(),
        "pattern_version": PATTERN_VERSION,
        "pattern_key": f"{PATTERN_VERSION}:" + "|".join(pairs) if len(pairs) == 3 else None,
        "selected_channels": sorted(channels), "observations": observations,
        "derived_features": sorted(context),
    }
