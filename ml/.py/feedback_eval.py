"""
D-9 ML feedback loop - the evaluator (docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md, 6-7): labels,
metrics and verdicts as pure functions on window and failure tables. It reads no files: .py/feedback_loop.py reads the
window log, the outbox, the closed cases and the lab cache, and calls these.

Labels generalise B-18's Frame (results/B-18/early_warning.py) to several failures per gun:
    lead        the time from a window's end (window start + 60 s) to the gun's next failure; inf when there is none
    excluded    windows ending up to POST_FAILURE_S after a failure (B-18 drops everything after its one failure)
    record end  pending onsets, the monitored days and the field normal population lean on the record end only for
                windows with no later failure - when a failure follows, the outcome is already known
So with one failure at the end of every gun's record (the lab) the labels are B-18's exactly, and the confidence
intervals - a gun bootstrap with A-5's seeds (results/A-5/overlap.py) - reproduce B-20 to the last digit.

Window table columns: gun, window_start (naive UTC, the minute bucket), score, warmup, non_welding_share,
error_active; the field adds early_warning (what the server decided) and ew_threshold.
Failure table columns: gun, T (naive UTC); optional status, event_id, rule_code.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---- constants (spec 7.8); calibrated ones match results/D-9/calibrate.json (tests/test_feedback_loop.py)
MARGIN, CHEAPER = 0.120, 0.843
Q_MINUS, Q_PLUS, Q_REFERENCE = 0.960, 0.980, 0.975
MIN_FAILURES, MIN_GUNS = 10, 5
PROFILE_CASES, PROFILE_DAYS = 2, 2
FA_CAP = 24
REFERENCE = "model_ng_k1 @ 0.975"
GATE_NON_WELDING = 0.5
ZONE_S, MIN_ZONE_WINDOWS, PLACEBO_DAYS = (600, 3600), 25, range(1, 7)
CALIBRATION_LEAD_S = 86400
N_BOOT, SEED_SINGLE, SEED_PAIRED = 2000, 42, 7
POST_FAILURE_S = 3600
WINDOW_S = 60
OUTBOX_RETENTION_DAYS = 30
CLASSES = ("E01", "E02", "E03", "E04")
REASONS = ("설비 정지 · 전원 꺼짐", "정상 작업 중 변동", "센서 · 신호 문제", "기타")
REASON_NONE = "사유 없음"

# the "출처" column of spec 7.8, per constant (the report's criteria)
CONSTANT_SOURCES = {
    "MARGIN": "보정(3.2): 재학습 잡음 (`calibrate.json` 0.1201)",
    "CHEAPER": "보정(3.2): 오알람 수 잡음 (`calibrate.json` 0.8430)",
    "Q_MINUS": "보정(3.2): 실험실 오알람 곡선",
    "Q_PLUS": "보정(3.2): 실험실 오알람 곡선",
    "MIN_FAILURES": "보정(3.2): 잘못된 반영 제안 ≤ 5%",
    "PROFILE_CASES": "미탐 우선 보정(3.2): 1 + 견딜 오클릭 1번, 동률 처리",
    "PROFILE_DAYS": "미탐 우선 보정(3.2): 1 + 견딜 오클릭 1번, 동률 처리",
    "FA_CAP": "사용자 결정(11-01, B-20)",
    "REFERENCE": "B-20 운영점",
    "GATE_NON_WELDING": "서빙의 일반 알람 게이트",
    "ZONE_S": "B-18 사전 고정(바꾸면 B-20과 비교가 깨짐)",
    "MIN_ZONE_WINDOWS": "B-18 사전 고정(바꾸면 B-20과 비교가 깨짐)",
    "PLACEBO_DAYS": "B-18 사전 고정(바꾸면 B-20과 비교가 깨짐)",
    "CALIBRATION_LEAD_S": "번들의 보정 모집단",
    "N_BOOT": "A-5와 같게(재현용)",
    "SEED_SINGLE": "A-5와 같게(재현용)",
    "SEED_PAIRED": "A-5와 같게(재현용)",
    "POST_FAILURE_S": "가정: 데이터로 정할 수 없음(모든 파일이 고장 10분 뒤 끝남). 8.2 진단으로 다시 정함",
    "MIN_GUNS": "가정: 데이터로 정할 수 없음(설비마다 고장 1건). 부트스트랩이 도는 최소 조건",
}

_S = np.timedelta64(1, "s")


@dataclass
class Frame:
    """The windows a rule is judged on, with their leads, run breaks and the failures' zones (spec 6.2)."""
    w: pd.DataFrame                    # kept windows, sorted by (gun, window_start); + end, lead (s), nxt
    guns: list[str]
    gun: np.ndarray                    # gun index per window
    brk: np.ndarray                    # a run starts here (first window, new gun, or not 60 s after the previous)
    ok: np.ndarray                     # not warm-up
    normal: np.ndarray                 # monitored: lead > 60 min, not warm-up, outcome known
    tail: np.ndarray                   # no later failure and within 60 min of the record end: outcome unknown
    days: np.ndarray                   # monitored days per gun
    record_end: np.ndarray             # per gun: the last window end before any exclusion
    failures: pd.DataFrame             # kept failures, ordered by (gun index, T)
    fail_gun: np.ndarray               # gun index per failure
    zones: dict[str, list[np.ndarray]]  # "real", "p1".."p6": per failure, the window positions of its zone
    valid: dict[str, np.ndarray]       # per failure: does the zone count
    excluded_windows: int
    dropped_failures: int


@dataclass
class RuleResult:
    """One rule on one frame: per failure hit / chance, per gun false alarms, pending, acted, monitored days."""
    real: np.ndarray
    chance: np.ndarray
    fa: np.ndarray
    pending: np.ndarray
    acted: np.ndarray
    days: np.ndarray
    onset: np.ndarray                  # per window: an alarm run starts here


@dataclass
class Evidence:
    """What the bootstrap and the verdicts need: per failure hit / chance (pooled over runs) and the false-alarm rate."""
    real: np.ndarray
    chance: np.ndarray
    fail_gun: np.ndarray
    n_guns: int
    fa_per_day: float


@dataclass(frozen=True)
class Challenger:
    """A phase-1 challenger (spec 7.2): window table -> score -> condition. quantile None = the current threshold."""
    name: str
    kind: str
    quantile: float | None
    reason: str | None
    apply_ko: str
    lab: bool


CHALLENGERS = (
    Challenger("GATE", "model", None, "설비 정지 · 전원 꺼짐",
               "`train.py --early-warning-rule model_k1`로 다시 학습, B-9 절차로 번들 교체", True),
    Challenger("Q+", "model_ng", Q_PLUS, "정상 작업 중 변동", "`train.py --early-warning-q 0.98`로 다시 학습, 번들 교체", True),
    Challenger("Q-", "model_ng", Q_MINUS, None, "`train.py --early-warning-q 0.96`로 다시 학습, 번들 교체", True),
    Challenger("Q0.975", "model_ng", Q_REFERENCE, None,
               '지금은 손잡이가 없다. "검토 필요"가 나오고 사람이 반영하기로 하면 임계값을 직접 지정하는 옵션을 만든다', False),
)


@contextmanager
def _quiet():
    """nanmean / nanpercentile of empty slices warn; an empty zone is a normal case here (as in B-18 / A-5)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def build_frame(windows: pd.DataFrame, failures: pd.DataFrame) -> Frame:
    """Spec 6.2 on one window table. With one failure at the end of every gun's record it is B-18's load + Frame."""
    w = windows.sort_values(["gun", "window_start"], kind="stable").reset_index(drop=True)
    start = w["window_start"].to_numpy("datetime64[ns]")
    end = start + np.timedelta64(WINDOW_S, "s")
    names = w["gun"].to_numpy()
    f = failures.sort_values(["gun", "T"], kind="stable").reset_index(drop=True)
    f_names = f["gun"].to_numpy()
    f_times = f["T"].to_numpy("datetime64[ns]")

    # leads and exclusion, gun by gun (windows and failures are sorted by gun, then time)
    lead = np.full(len(w), np.inf)
    nxt_local = np.full(len(w), -1)
    drop = np.zeros(len(w), bool)
    rec_end: dict[str, np.datetime64] = {}
    bounds = np.flatnonzero(np.r_[True, names[1:] != names[:-1], True]) if len(w) else np.array([0])
    for a, b in zip(bounds[:-1], bounds[1:]):
        g = names[a]
        e = end[a:b]
        rec_end[g] = e.max()
        ts = f_times[f_names == g]
        if not len(ts):
            continue
        j = np.searchsorted(ts, e, side="right")          # first failure with T > end
        has = j < len(ts)
        lead[a:b][has] = (ts[j[has]] - e[has]) / _S
        nxt_local[a:b][has] = j[has]
        after = j > 0
        drop[a:b][after] = (e[after] - ts[j[after] - 1]) / _S < POST_FAILURE_S

    keep = ~drop
    w = w.loc[keep].reset_index(drop=True)
    end, lead, nxt_local, names = end[keep], lead[keep], nxt_local[keep], names[keep]
    codes, uniques = pd.factorize(names, sort=True)       # B-18: pd.factorize on the sorted files
    gun = codes.astype(int)
    guns = [str(g) for g in uniques]
    gidx = {g: i for i, g in enumerate(guns)}
    kept_f = np.isin(f_names, guns)
    f = f.loc[kept_f].reset_index(drop=True)
    f_names, f_times = f_names[kept_f], f_times[kept_f]
    fail_gun = np.array([gidx[g] for g in f_names], dtype=int)
    offset = np.searchsorted(fail_gun, np.arange(len(guns)), side="left")  # first failure of each gun
    nxt = np.where(nxt_local >= 0, offset[gun] + nxt_local, -1) if len(w) else np.zeros(0, int)
    w["end"], w["lead"], w["nxt"] = end, lead, nxt

    st = w["window_start"].to_numpy("datetime64[ns]")
    same = np.r_[False, (gun[1:] == gun[:-1]) & (np.diff(st) == np.timedelta64(WINDOW_S, "s"))] if len(w) else \
        np.zeros(0, bool)
    brk = ~same
    ok = ~w["warmup"].to_numpy(bool)
    record_end = np.array([rec_end[g] for g in guns], dtype="datetime64[ns]")
    tail = np.isinf(lead) & (end > record_end[gun] - np.timedelta64(ZONE_S[1], "s")) if len(w) else np.zeros(0, bool)
    normal = (lead > ZONE_S[1]) & ok & ~tail
    days = np.bincount(gun, normal, len(guns)) / 1440

    in_real = (nxt >= 0) & (lead >= ZONE_S[0]) & (lead <= ZONE_S[1]) & ok
    real_idx = np.flatnonzero(in_real)
    zones = {"real": [real_idx[nxt[real_idx] == i] for i in range(len(f))]}
    valid = {"real": np.array([len(z) >= MIN_ZONE_WINDOWS for z in zones["real"]], dtype=bool)}
    first = np.searchsorted(gun, np.arange(len(guns)), side="left")   # windows are sorted by gun
    last = np.searchsorted(gun, np.arange(len(guns)), side="right")
    lo_s, hi_s = np.timedelta64(ZONE_S[1], "s"), np.timedelta64(ZONE_S[0], "s")
    for k in PLACEBO_DAYS:
        zk, vk = [], []
        for i in range(len(f)):
            g = fail_gun[i]
            a, b = first[g], last[g]
            anchor = f_times[i] - np.timedelta64(k * 86400, "s")
            lo = a + np.searchsorted(end[a:b], anchor - lo_s, side="left")
            hi = a + np.searchsorted(end[a:b], anchor - hi_s, side="right")
            idx = np.arange(lo, hi)
            idx = idx[ok[idx]]
            others = f_times[(fail_gun == g) & (np.arange(len(f)) != i)]
            overlap = bool(((anchor - lo_s <= others) & (others - lo_s <= anchor - hi_s)).any())
            zk.append(idx)
            vk.append(len(idx) >= MIN_ZONE_WINDOWS and not overlap)
        zones[f"p{k}"], valid[f"p{k}"] = zk, np.array(vk, dtype=bool)
    return Frame(w=w, guns=guns, gun=gun, brk=brk, ok=ok, normal=normal, tail=tail, days=days, record_end=record_end,
                 failures=f, fail_gun=fail_gun, zones=zones, valid=valid, excluded_windows=int(drop.sum()),
                 dropped_failures=int((~kept_f).sum()))


def evaluate(fr: Frame, c: np.ndarray, acted: np.ndarray | None = None) -> RuleResult:
    """One rule (a per-window condition, sustained 1 window) on a frame: B-18 Frame.evaluate generalised."""
    c = np.asarray(c, bool)
    onset = c & (fr.brk | ~np.r_[False, c[:-1]])
    a = np.zeros(len(c), bool) if acted is None else np.asarray(acted, bool)
    n = len(fr.guns)
    fa = np.bincount(fr.gun, onset & fr.normal & ~a, n)
    acted_n = np.bincount(fr.gun, onset & fr.normal & a, n)
    pending = np.bincount(fr.gun, onset & fr.tail, n)

    def hits(name: str) -> np.ndarray:
        return np.array([float(c[z].any()) if v else np.nan for z, v in zip(fr.zones[name], fr.valid[name])])

    real = hits("real")
    with _quiet():
        chance = np.nanmean(np.vstack([hits(f"p{k}") for k in PLACEBO_DAYS]), axis=0) if len(real) else np.zeros(0)
    return RuleResult(real=real, chance=chance, fa=fa, pending=pending, acted=acted_n, days=fr.days.copy(), onset=onset)


def pool(results: list[RuleResult], fail_gun: np.ndarray, n_guns: int) -> Evidence:
    """Per-failure hit / chance averaged over runs (A-5 stack); false alarms per monitored gun-day over all runs."""
    with _quiet():
        real = np.nanmean(np.vstack([r.real for r in results]), axis=0)
        chance = np.nanmean(np.vstack([r.chance for r in results]), axis=0)
    fa = sum(r.fa.sum() for r in results)
    days = sum(r.days.sum() for r in results)
    return Evidence(real=real, chance=chance, fail_gun=np.asarray(fail_gun, dtype=int), n_guns=int(n_guns),
                    fa_per_day=float(fa / days) if days > 0 else float("nan"))


def _draws(ev: Evidence, seed: int):
    """Gun bootstrap: every drawn gun brings all of its failures (A-5's draws when each gun has one failure)."""
    rng = np.random.default_rng(seed)
    fails_of = [np.flatnonzero(ev.fail_gun == g) for g in range(ev.n_guns)]
    for _ in range(N_BOOT):
        gi = rng.integers(0, ev.n_guns, ev.n_guns)
        yield np.concatenate([fails_of[g] for g in gi])


def summary(ev: Evidence) -> dict:
    """fa_per_day, recall, chance, excess and its 95% CI (seed 42), failures judged, guns with a judged failure."""
    det = np.isfinite(ev.real)
    out = {"fa_per_day": ev.fa_per_day, "recall": None, "chance": None, "excess": None, "ci": None,
           "failures": int(det.sum()), "guns": int(len(set(ev.fail_gun[det].tolist())))}
    if not det.any():
        return out
    with _quiet():
        boot = [np.nanmean(ev.real[f]) - np.nanmean(ev.chance[f]) for f in _draws(ev, SEED_SINGLE)]
        recall, chance = float(np.nanmean(ev.real)), float(np.nanmean(ev.chance))
        out.update(recall=recall, chance=chance, excess=recall - chance,
                   ci=[float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))])
    return out


def paired(a: Evidence, b: Evidence) -> dict:
    """D = excess(a) - excess(b) on the same failures, with one resample for both (seed 7), and on the same resamples
    the change in recall itself: D nets out chance, so recall can fall while D stays near 0."""
    if len(a.real) != len(b.real):
        raise ValueError("paired: the two rules must be judged on the same failures")
    if not (np.isfinite(a.real).any() and np.isfinite(b.real).any()):
        return {"D": None, "ci": None, "recall_change": None, "recall_change_ci": None}

    def d(f: np.ndarray) -> float:
        return (np.nanmean(a.real[f]) - np.nanmean(a.chance[f])) - (np.nanmean(b.real[f]) - np.nanmean(b.chance[f]))

    def dr(f: np.ndarray) -> float:
        return np.nanmean(a.real[f]) - np.nanmean(b.real[f])

    def ci(boot: list[float]) -> list[float]:
        return [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]

    every = np.arange(len(a.real))
    with _quiet():
        draws = list(_draws(a, SEED_PAIRED))
        return {"D": float(d(every)), "ci": ci([d(f) for f in draws]),
                "recall_change": float(dr(every)), "recall_change_ci": ci([dr(f) for f in draws])}


def normal_population(fr: Frame) -> np.ndarray:
    """Windows a quantile threshold is taken from (spec 7.2, the bundle's calibration population): lead > 1 day, not
    warm-up, no controller code; with no later failure, not within a day of the record end (lead not known yet)."""
    lead = fr.w["lead"].to_numpy()
    end = fr.w["end"].to_numpy("datetime64[ns]")
    unknown = np.isinf(lead) & (end > fr.record_end[fr.gun] - np.timedelta64(CALIBRATION_LEAD_S, "s")) \
        if len(lead) else np.zeros(0, bool)
    return (lead > CALIBRATION_LEAD_S) & fr.ok & ~fr.w["error_active"].to_numpy(bool) & ~unknown


def quantile_threshold(fr: Frame, q: float) -> float:
    s = fr.w["score"].to_numpy()[normal_population(fr)]
    return float(np.quantile(s, q)) if len(s) else float("nan")


def condition(fr: Frame, kind: str, thr: float, scores: np.ndarray | None = None) -> np.ndarray:
    """B-18 Frame.condition: model_ng = score >= thr; model = also non-welding share <= 0.5. Warm-up never alarms.
    `scores` replaces the logged scores (spec 9.7: a phase-2 model re-scores the window vectors)."""
    s = fr.w["score"].to_numpy() if scores is None else np.asarray(scores)
    c = s >= thr
    if kind == "model":
        c = c & (fr.w["non_welding_share"].to_numpy() <= GATE_NON_WELDING)
    elif kind != "model_ng":
        raise ValueError(f"unknown condition kind {kind}")
    return c & fr.ok


# ---- verdicts (spec 7.4, 7.5)
VERDICT_KO = {"pass": "통과", "insufficient": "증거 부족", "na": "해당 없음", "fail": "탈락"}
_DIFF, _NOLAB = "검토 필요(실험실과 현장이 다름)", "검토 필요(실험실 확인 없음)"
_AGAINST = "검토 필요(현장 임계값이 적용 방향과 반대)"
COMBINE = {
    ("pass", "pass"): ("propose", "반영 제안"),
    ("pass", "insufficient"): ("hold", "보류(현장 증거 쌓는 중)"),
    ("pass", "fail"): ("review", _DIFF),
    ("fail", "pass"): ("review", _DIFF),
    ("fail", "insufficient"): ("reject", "반영 안 함"),
    ("fail", "fail"): ("reject", "반영 안 함"),
    ("na", "pass"): ("review", _NOLAB),  # user decision 11-08: no lab check, no proposal
    ("na", "insufficient"): ("hold", "보류"),
    ("na", "fail"): ("reject", "반영 안 함"),
    ("pass", "na"): ("hold", "보류"),
    ("fail", "na"): ("reject", "반영 안 함"),
    ("na", "na"): ("not_evaluated", "평가 안 함"),
}


def source_verdict(ch: dict, d_champion: dict, d_reference: dict, fa_champion: float, fa_reference: float,
                   field: bool) -> str:
    """Spec 7.4 for one evidence source. ch = the challenger's summary; "na" is the caller's call (no evidence)."""
    if field and (ch["failures"] < MIN_FAILURES or ch["guns"] < MIN_GUNS):
        return "insufficient"
    if d_champion.get("ci") is None or d_reference.get("ci") is None:
        return "insufficient" if field else "fail"
    fa = ch["fa_per_day"]
    better = d_champion["ci"][0] > 0 and fa <= FA_CAP
    cheaper = (d_reference["ci"][0] > -MARGIN and d_reference["ci"][1] >= 0
               and fa <= CHEAPER * fa_reference and fa < fa_champion)
    return "pass" if (better or cheaper) else "fail"


def combine(lab: str, field: str, against_apply: bool = False) -> tuple[str, str]:
    """Spec 7.5: (the report's verdict code, its Korean text). A field pass measured on the other side of the served
    threshold from where the apply command moves it is no evidence for that command: it is never a proposal."""
    code, ko = COMBINE[(lab, field)]
    return ("review", _AGAINST) if against_apply and code == "propose" else (code, ko)


# ---- critical profile (spec 6.4, 7.6)
PROFILE_RISK_KO = "이 유형의 진짜 고장이 나도 경고로만 남고 케이스가 생기지 않음(C-3, `main.py:659-663`)."


@dataclass
class CriticalCase:
    """One closed-or-open critical case (a terminal-code alarm). outcome: false_alarm / real (resolved) / unknown."""
    case_id: str
    gun: str
    cls: str | None
    outcome: str
    detected_at: pd.Timestamp


def profile_proposals(cases: list[CriticalCase], profiles: dict[str, list[str] | None]) -> dict:
    """Spec 7.6: per gun, the fault classes with >= PROFILE_CASES false-alarm cases on >= PROFILE_DAYS UTC days, no real
    case and still in the gun's profile (none = every class) leave the profile. An empty profile is never proposed, and
    a class with an unconfirmed case (open or unresolved: it may be real) goes to a person instead."""
    by: dict[tuple[str, str], dict] = {}
    for c in cases:
        if c.cls not in CLASSES:
            continue
        v = by.setdefault((c.gun, c.cls), {"fa": [], "real": 0, "unknown": 0})
        if c.outcome == "false_alarm":
            v["fa"].append(c)
        elif c.outcome == "real":
            v["real"] += 1
        else:
            v["unknown"] += 1
    proposals, watching, review = [], [], []
    for gun in sorted({g for g, _ in by}):
        current = profiles.get(gun)
        current = list(CLASSES) if current is None else list(current)
        passing, evidence = [], []
        for cls in CLASSES:
            v = by.get((gun, cls))
            if not v or not v["fa"] or v["real"] or cls not in current:
                continue
            fa = sorted(v["fa"], key=lambda c: (pd.Timestamp(c.detected_at), c.case_id))
            n, m = len(fa), len({pd.Timestamp(c.detected_at).date() for c in fa})
            if n >= PROFILE_CASES and m >= PROFILE_DAYS:
                if v["unknown"]:
                    review.append({"gun": gun, "exclude": [cls], "text": "검토 필요",
                                   "why": f"미확인 케이스 {v['unknown']}건"})
                    continue
                passing.append(cls)
                evidence += [{"case_id": c.case_id, "date": str(pd.Timestamp(c.detected_at).date())} for c in fa]
            else:
                watching.append({"gun": gun, "cls": cls, "cases": n, "days": m,
                                 "text": f"지켜보는 중({PROFILE_CASES}건 중 {n}건, {PROFILE_DAYS}일 중 {m}일)"})
        if not passing:
            continue
        body = [c for c in CLASSES if c in current and c not in passing]
        if body:
            proposals.append({"gun": gun, "exclude": passing, "body": {"expected_fault_classes": body},
                              "cases": evidence, "risk_ko": PROFILE_RISK_KO})
        else:
            review.append({"gun": gun, "exclude": passing, "text": "검토 필요"})
    return {"proposals": proposals, "watching": watching, "review": review}


# ---- early-warning human labels (spec 6.3)
def reason_of(text: str | None) -> str:
    """A closure reason -> one of the 4 chips; other text -> 기타; none -> 사유 없음 (FIELD_MEMORY off, no reason)."""
    t = (text or "").strip()
    if not t:
        return REASON_NONE
    return t if t in REASONS else "기타"


@dataclass
class EwLink:
    """An early-warning handoff in a case: the case's closure and the onset window (onset_at floored to the minute)."""
    case_id: str
    outcome: str                       # false_alarm / resolved / anything else (open, unresolved, ...)
    reason: str | None
    note: str | None
    gun: str
    onset_start: pd.Timestamp | None


def onset_index(fr: Frame, gun: str, start: pd.Timestamp | None) -> int | None:
    """Position in fr.w of the gun's window starting at `start`, or None."""
    if start is None or gun not in fr.guns:
        return None
    g = fr.guns.index(gun)
    a, b = np.searchsorted(fr.gun, [g, g + 1], side="left")
    st = fr.w["window_start"].to_numpy("datetime64[ns]")[a:b]
    t = np.datetime64(pd.Timestamp(start).to_datetime64(), "ns")
    j = int(np.searchsorted(st, t))
    return int(a + j) if j < len(st) and st[j] == t else None


def onset_label(fr: Frame, rr: RuleResult, i: int) -> str:
    """The automatic label of window i for rule rr (spec 6.2)."""
    if not rr.onset[i]:
        return "not_onset"
    if fr.tail[i]:
        return "pending"
    if fr.normal[i]:
        return "false_alarm"
    return "hit_side"


def acted_mask(fr: Frame, champion_c: np.ndarray, links: list[EwLink]) -> np.ndarray:
    """Windows of the current rule's alarm runs that hold an onset of a case closed as resolved (조치함, plan
    decision 6): every rule's false-alarm onsets there count as acted on, not as false alarms."""
    c = np.asarray(champion_c, bool)
    run = np.where(c, np.cumsum(c & (fr.brk | ~np.r_[False, c[:-1]])), 0)
    acted = {int(run[i]) for i in (onset_index(fr, x.gun, x.onset_start) for x in links if x.outcome == "resolved")
             if i is not None and c[i]}
    return np.isin(run, sorted(acted)) & c


def ew_human(fr: Frame, champion: RuleResult, links: list[EwLink]) -> dict:
    """Spec 6.3: reasons of false-alarm closures on onsets the automatic label also calls false alarms, notes,
    disagreements (closed as false alarm, but a failure followed within 60 min), and links whose onset is not logged."""
    reasons: dict[str, int] = {}
    reason_guns: dict[str, set[str]] = {}
    notes, disagreements, unmatched = [], [], 0
    for x in links:
        if x.outcome not in ("false_alarm", "resolved"):
            continue
        i = onset_index(fr, x.gun, x.onset_start)
        if i is None or not champion.onset[i]:
            unmatched += 1
            continue
        if x.outcome != "false_alarm":
            continue
        label = onset_label(fr, champion, i)
        if label == "false_alarm":
            r = reason_of(x.reason)
            reasons[r] = reasons.get(r, 0) + 1
            reason_guns.setdefault(r, set()).add(x.gun)
            text = x.note or (x.reason if r == "기타" else None)
            if r in ("기타", REASON_NONE) and text:
                notes.append({"case_id": x.case_id, "gun": x.gun, "reason": r, "note": text})
        elif label == "hit_side":
            disagreements.append({"case_id": x.case_id, "gun": x.gun,
                                  "onset_at": pd.Timestamp(x.onset_start).isoformat() + "Z", "auto": label})
    return {"false_alarm": int(champion.fa.sum()), "pending": int(champion.pending.sum()),
            "acted": int(champion.acted.sum()), "reasons": reasons,
            "reason_guns": {r: sorted(g) for r, g in reason_guns.items()}, "notes": notes,
            "disagreements": disagreements, "unmatched": unmatched}


# ---- diagnostics
def post_failure(windows: pd.DataFrame, failures: pd.DataFrame) -> list[dict]:
    """Spec 8.2 (7): share of early-warning windows 0-120 min after each failure, in 10-min bins (by window end), up to
    the gun's next failure; warm-up windows left out. Re-derives POST_FAILURE_S once the field has data after failures."""
    n = np.zeros(12, int)
    ew = np.zeros(12, int)
    f = failures.sort_values(["gun", "T"]) if len(failures) else failures
    for gun, grp in (f.groupby("gun") if len(f) else []):
        w = windows[windows["gun"] == gun]
        end = w["window_start"].to_numpy("datetime64[ns]") + np.timedelta64(WINDOW_S, "s")
        on, ok = w["early_warning"].to_numpy(bool), ~w["warmup"].to_numpy(bool)
        ts = grp["T"].to_numpy("datetime64[ns]")
        for k, t in enumerate(ts):
            el = (end - t) / _S
            sel = ok & (el > 0) & (el <= 7200)
            if k + 1 < len(ts):
                sel &= end <= ts[k + 1]
            b = (np.ceil(el[sel] / 600) - 1).astype(int)
            n += np.bincount(b, minlength=12)[:12]
            ew += np.bincount(b, on[sel], minlength=12)[:12].astype(int)
    return [{"from_min": 10 * i, "to_min": 10 * i + 10, "windows": int(n[i]),
             "ew_share": float(ew[i] / n[i]) if n[i] else None} for i in range(12)]


def champion_mismatch(windows: pd.DataFrame) -> int:
    """Windows where the served early warning differs from "score >= ew_threshold and not warm-up" (spec 7.1)."""
    if "ew_threshold" not in windows or "early_warning" not in windows:
        return 0
    thr = windows["ew_threshold"].to_numpy(float)
    has = np.isfinite(thr)
    expect = (windows["score"].to_numpy(float) >= np.where(has, thr, np.inf)) & ~windows["warmup"].to_numpy(bool)
    return int(((windows["early_warning"].to_numpy(bool) != expect) & has).sum())


# ---- assembled evaluations (spec 7.1-7.3)
def _challenger_result(ev: Evidence, champ_ev: Evidence, ref_ev: Evidence, champ: dict, ref: dict, thr,
                       field: bool) -> dict:
    s = summary(ev)
    dc, dr = paired(ev, champ_ev), paired(ev, ref_ev)
    return {"threshold": thr, "summary": s, "d_champion": dc, "d_reference": dr,
            "verdict": source_verdict(s, dc, dr, champ["fa_per_day"], ref["fa_per_day"], field)}


def lab_evaluation(runs) -> dict:
    """The lab (B-19's out-of-fold runs, meta fault times): the B-20 fixed rule and GATE / Q+ / Q- per run, pooled
    over the runs (A-5). The reference point is the current rule itself; Q0.975 is the same rule here, so na."""
    names, thresholds, res = [], {}, {"champion": [], "GATE": [], "Q+": [], "Q-": []}
    guns = fail_gun = fail_key = None
    for name, w, f in runs:
        fr = build_frame(w, f)
        key = list(zip(fr.failures["gun"], fr.failures["T"]))
        if guns is None:
            guns, fail_gun, fail_key = fr.guns, fr.fail_gun, key
        if fr.guns != guns or key != fail_key:
            raise ValueError(f"lab run {name}: other guns or failures than the first run")
        thr = {"champion": quantile_threshold(fr, Q_REFERENCE), "Q+": quantile_threshold(fr, Q_PLUS),
               "Q-": quantile_threshold(fr, Q_MINUS)}
        res["champion"].append(evaluate(fr, condition(fr, "model_ng", thr["champion"])))
        res["GATE"].append(evaluate(fr, condition(fr, "model", thr["champion"])))
        res["Q+"].append(evaluate(fr, condition(fr, "model_ng", thr["Q+"])))
        res["Q-"].append(evaluate(fr, condition(fr, "model_ng", thr["Q-"])))
        names.append(name)
        thresholds[name] = thr
        del fr
    if not names:
        raise ValueError("lab: no runs")
    ev = {k: pool(v, fail_gun, len(guns)) for k, v in res.items()}
    champ = summary(ev["champion"])
    challengers = {}
    for ch in CHALLENGERS:
        if not ch.lab:
            challengers[ch.name] = {"verdict": "na"}
            continue
        challengers[ch.name] = _challenger_result(ev[ch.name], ev["champion"], ev["champion"], champ, champ, None,
                                                  field=False)
    return {"runs": names, "champion": champ, "reference": champ, "challengers": challengers,
            "thresholds": thresholds}


def field_evaluation(windows: pd.DataFrame, failures: pd.DataFrame, model_entry: dict | None,
                     links: list[EwLink]) -> dict | None:
    """The field (the window log of one model, judged failures, human links): the served decision, the B-20 reference
    point (7.1) and the challengers with field thresholds; every rule judged with the same acted-on runs."""
    if windows is None or not len(windows):
        return None
    notes: list[str] = []
    fr = build_frame(windows, failures)
    if fr.dropped_failures:
        notes.append(f"창 기록이 없는 설비의 고장 {fr.dropped_failures}건은 평가에서 뺌")
    ew = (model_entry or {}).get("early_warning") or {}
    thr_c = ew.get("threshold")
    if thr_c is None and "ew_threshold" in windows:
        vals = windows["ew_threshold"].dropna()
        thr_c = float(vals.iloc[0]) if len(vals) else None
    n_guns = len(fr.guns)
    if thr_c is None:
        notes.append("창 기록의 모델에 조기 경보 임계값이 없어 조기 경보 현장 평가는 해당 없음")
        return {"failures": int(fr.valid["real"].sum()), "guns": n_guns, "mismatch": 0, "champion_threshold": None,
                "reference_threshold": None, "reference_basis": None, "champion": None, "reference": None,
                "challengers": {ch.name: {"verdict": "na"} for ch in CHALLENGERS},
                "labels": {"failures": {"hit": 0, "miss": 0, "undetermined": len(fr.failures)}, "ew": None},
                "post_failure": None, "warnings": notes}
    q = ew.get("quantile")
    served_q = Q_REFERENCE if q is None else float(q)
    basis = "ew_threshold" if q is None or abs(float(q) - Q_REFERENCE) < 1e-9 else "field_q0.975"
    thr_r = float(thr_c) if basis == "ew_threshold" else quantile_threshold(fr, Q_REFERENCE)
    champion_c = fr.w["early_warning"].to_numpy(bool)
    acted = acted_mask(fr, champion_c, links)
    rr_c = evaluate(fr, champion_c, acted)
    ev_c = pool([rr_c], fr.fail_gun, n_guns)
    champ = summary(ev_c)
    ref = ev_r = None
    if np.isfinite(thr_r):
        ev_r = pool([evaluate(fr, condition(fr, "model_ng", thr_r), acted)], fr.fail_gun, n_guns)
        ref = summary(ev_r)
    challengers, empty = {}, []
    for ch in CHALLENGERS:
        thr = float(thr_c) if ch.quantile is None else quantile_threshold(fr, ch.quantile)
        if ev_r is None or not np.isfinite(thr):
            challengers[ch.name] = {"verdict": "na"}
            empty.append(ch.name)
            continue
        ev = pool([evaluate(fr, condition(fr, ch.kind, thr), acted)], fr.fail_gun, n_guns)
        res = _challenger_result(ev, ev_c, ev_r, champ, ref, thr, field=True)
        # the apply command retrains at this quantile: it moves the threshold up (Q+) or down (Q-) from the served one
        res["against_apply"] = bool(ch.lab and ch.quantile is not None and ch.quantile != served_q
                                    and (thr - float(thr_c)) * (ch.quantile - served_q) < 0)
        challengers[ch.name] = res
    if empty:
        notes.append(f"현장 정상 창이 없어 {', '.join(empty)}의 현장 평가는 해당 없음")
    mismatch = champion_mismatch(windows)
    if mismatch:
        notes.append(f"서빙 판정과 다시 계산한 지금 규칙이 다른 창 {mismatch}개")
    human = ew_human(fr, rr_c, links)
    if human["unmatched"]:
        notes.append(f"창 기록에서 시작 창을 찾지 못한 조기 경보 핸드오프 {human['unmatched']}건")
    real = rr_c.real
    labels = {"failures": {"hit": int((real == 1).sum()), "miss": int((real == 0).sum()),
                           "undetermined": int(np.isnan(real).sum())}, "ew": human}
    return {"failures": champ["failures"], "guns": champ["guns"], "mismatch": mismatch,
            "champion_threshold": float(thr_c), "reference_threshold": thr_r, "reference_basis": basis,
            "champion": champ, "reference": ref, "challengers": challengers, "labels": labels,
            "post_failure": post_failure(windows, fr.failures), "warnings": notes}
