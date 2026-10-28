"""
Unit tests of .py/hitl_loop.py (D-8 HITL 드라이버). 서버도 실제 CSV도 쓰지 않는다: tmp_path의 작은 CSV와
on_result를 직접 부르는 가짜 replay_fn으로 Feeder 스레드를 몰고, 그래프는 스크립트 Feeder와 메모리 안의
가짜 ML / api client(FakeML, FakeAPI)로 몬다. 수 초 안에 끝난다.
"""
import os
import re
import sys
import threading

import httpx
import pytest

pytest.importorskip("langgraph")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".py"))

import hitl_loop as hl  # noqa: E402


def write_csv(path, times):
    with open(path, "w", encoding="utf-8") as f:
        f.write("time,c1,error\n")
        for t in times:
            f.write(f"{t},1.0,0\n")
    return str(path)


def fake_replay(events, calls=None):
    """on_result를 (응답, None) 한 번 + 이벤트마다 한 번 부르는 replay_file 대역."""
    def replay_fn(client, path, gun_id=None, chunk_s=60, reset=True, on_result=None, **kw):
        if calls is not None:
            calls.append((path, gun_id, chunk_s, reset))
        on_result({"gun_id": gun_id}, None)
        for ev in events:
            on_result({"gun_id": gun_id}, ev)
        return {"gun_id": gun_id, "critical_events": list(events)}
    return replay_fn


def items_until_end(feeder, limit=20):
    out = []
    for _ in range(limit):
        it = feeder.next_item(timeout=2.0)
        assert it is not None, f"queue stalled after {out}"
        out.append(it)
        if it[0] in ("end", "error"):
            break
    return out


def test_file_span(tmp_path):
    p = write_csv(tmp_path / "g.csv", ["2021-09-04 05:00:00", "2021-09-04 02:46:08", "2021-09-06 11:00:01",
                                       "2021-09-05 00:00:00"])
    assert hl.file_span(p) == ("2021-09-04T02:46:08Z", "2021-09-06T11:00:01Z")


def test_feeder_orders_files_and_pauses(tmp_path):
    late = write_csv(tmp_path / "late.csv", ["2021-09-10 00:00:00", "2021-09-10 01:00:00"])
    early = write_csv(tmp_path / "early.csv", ["2021-09-01 00:00:00", "2021-09-01 01:00:00"])
    calls = []
    ev1, ev2 = {"window_end": "a"}, {"window_end": "b"}
    f = hl.Feeder([late, early], make_client=lambda: object(), chunk_s=600,
                  replay_fn=fake_replay([ev1, ev2], calls))
    f.start()
    try:
        assert f.next_item(timeout=2.0) == ("file", "early", early, "2021-09-01T00:00:00Z", "2021-09-01T01:00:00Z")
        assert f.next_item(timeout=2.0) == ("alert", 1, "early", ev1)
        assert f.next_item(timeout=0.2) is None  # seq 1을 놓아 주기 전에는 멈춰 있다
        f.release(1)
        assert f.next_item(timeout=2.0) == ("alert", 2, "early", ev2)
        f.release(2)
        assert f.next_item(timeout=2.0) == ("file", "late", late, "2021-09-10T00:00:00Z", "2021-09-10T01:00:00Z")
        assert f.next_item(timeout=2.0) == ("alert", 3, "late", ev1)
        f.release(3)
        assert f.next_item(timeout=2.0) == ("alert", 4, "late", ev2)
        f.release(4)
        assert f.next_item(timeout=2.0) == ("end",)
    finally:
        f.stop()
        f.join(timeout=2.0)
    assert not f.is_alive()
    assert calls == [(early, "early", 600, True), (late, "late", 600, True)]


def test_feeder_error_item(tmp_path):
    p = write_csv(tmp_path / "g.csv", ["2021-09-01 00:00:00"])

    def boom(client, path, gun_id=None, chunk_s=60, reset=True, on_result=None, **kw):
        raise RuntimeError("HTTP 500")

    f = hl.Feeder([p], make_client=lambda: object(), replay_fn=boom)
    f.start()
    items = items_until_end(f)
    f.join(timeout=2.0)
    assert items[0][0] == "file"
    assert items[-1][0] == "error"
    assert "RuntimeError" in items[-1][1]


def test_feeder_no_pause_dry_run(tmp_path):
    p = write_csv(tmp_path / "g.csv", ["2021-09-01 00:00:00"])
    evs = [{"window_end": str(i)} for i in range(3)]
    f = hl.Feeder([p], make_client=lambda: object(), pause=False, replay_fn=fake_replay(evs))
    f.start()
    items = items_until_end(f)
    f.join(timeout=2.0)
    assert [it[:2] for it in items[1:]] == [("alert", 1), ("alert", 2), ("alert", 3), ("end",)]


def test_feeder_stop_while_waiting(tmp_path):
    p = write_csv(tmp_path / "g.csv", ["2021-09-01 00:00:00"])
    closed = []

    class Client:
        def close(self):
            closed.append(True)

    f = hl.Feeder([p], make_client=Client, replay_fn=fake_replay([{"window_end": "a"}, {"window_end": "b"}]))
    f.start()
    assert f.next_item(timeout=2.0)[0] == "file"
    assert f.next_item(timeout=2.0)[:2] == ("alert", 1)
    f.stop()
    f.join(timeout=2.0)
    assert not f.is_alive()
    assert f.next_item(timeout=0.1) is None  # 멈춤은 error도 end도 아니다
    assert closed == [True]


# ---- 그래프와 실행기 (Task 5) ----

UI = "http://ui"
END_ISO = "2021-09-04T12:00:00Z"  # 파일 끝: 05:00 trigger에서 25200초 → "far"
FILE_ITEM = ("file", "test_0", "x/test_0.csv", "2021-09-04T00:00:00Z", END_ISO)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def handoff(eid, code="E029", gun="test_0", t="2021-09-04T05:00:00Z"):
    return {"event_id": eid, "gun_id": gun, "detected_at": t,
            "trigger": {"source": "rule", "rule_code": code, "rule_trigger_time": t}}


class FakeML:
    """ML 서버 대역: /health와 /handoffs(outbox)만."""

    def __init__(self, outbox=None, health=None):
        self.outbox = list(outbox or [])
        self.health = health

    def get(self, path, params=None):
        if path == "/health":
            return FakeResponse(200, self.health or {
                "guns_tracked": 0, "rag_url": None,
                "outbox": {"pending": 0, "delivered": 0, "failed": 0, "pull_only": len(self.outbox)}})
        assert path == "/handoffs", path
        return FakeResponse(200, [{"handoff": h, "delivery": "pull_only"} for h in self.outbox])


def r1_note(gun, code, fa, real):
    """api/field_memory.py의 R1 문장(브리프 그대로)."""
    return (f"같은 설비·코드({gun} {code})에서 사람이 오알람으로 닫은 기록 {fa}건 · 고장 확인 {real}건 "
            f"→ 오알람(운영 원인)인지부터 확인. 아니면 카드의 점검 순서대로.")


class FakeAPI:
    """api 대역: 메모리 안에서 ingest(같은 설비·코드의 열린 케이스면 반복으로 묶기), 닫기, labels·alarms·checks 계산.
    안내는 E 코드 카드 하나(candidates[0].situation_id = 코드), 현장 기억은 GET 케이스의 최상위 field_memory.
    닫기는 사람이 화면에서 하는 것처럼 /api/history를 polls번째 조회할 때 일어난다."""

    def __init__(self, ml, memory_on=True, empty_syncs=0, ml_status="ok", annotate=True):
        self.ml = ml
        self.memory_on = memory_on
        self.annotate = annotate  # False면 GET 케이스에 field_memory가 없다(플래그 꺼진 api처럼)
        self.ml_status = ml_status  # /api/health의 ml (api가 본 ML 상태)
        self.empty_syncs = empty_syncs  # 처음 이만큼은 sync가 {}를 돌려준다(ML을 아직 못 읽은 것처럼)
        self.cases = {}
        self.seen = set()
        self.closing = {}  # case_id -> [남은 history 조회 수, outcome, records 추가분]
        self.sync_calls = 0
        self.paths = []  # 조회한 주소 (시작 점검 순서·상세 조회 횟수 확인용)

    def ingest(self, h):
        if h["event_id"] in self.seen:
            return "duplicate"
        self.seen.add(h["event_id"])
        code = h["trigger"]["rule_code"]
        for d in self.cases.values():
            # api/store.py open_case처럼 끝나지 않은 케이스(도움 요청 포함)면 묶는다
            if (d["status"] not in ("resolved", "unresolved", "false_alarm") and d["event"]["gun_id"] == h["gun_id"]
                    and d["event"]["trigger"]["rule_code"] == code):
                d["repeats"].append({"event_id": h["event_id"], "detected_at": h["detected_at"]})
                return "repeat"
        cid = f"CASE-{len(self.cases) + 1:04d}"
        card = code or "E029"
        self.cases[cid] = {
            "case_id": cid, "status": "new", "event": h, "repeats": [],
            "guidance": {"status": "ok", "candidates": [
                {"situation_id": card, "checks": [{"id": f"{card}-K1"}, {"id": f"{card}-K2"}], "causes": []}]},
            "records": {"judgments": [], "results": [], "checks": []}}
        return "new"

    def sync(self):
        counts = {}
        for h in self.ml.outbox:
            k = self.ingest(h)
            counts[k] = counts.get(k, 0) + 1
        return counts

    def close_after(self, case_id, outcome, polls=2, **records):
        """닫기 대역: /api/history를 polls번째 조회할 때 닫힌다. records(checks·judgments·results)도 그때 들어간다."""
        self.closing[case_id] = [polls, outcome, records]

    def _summary(self, d):
        return {"case_id": d["case_id"], "status": d["status"], "event": d["event"],
                "repeat_count": len(d["repeats"])}

    @staticmethod
    def _key(d):
        return f"{d['event']['gun_id']}|{d['event']['trigger']['rule_code'] or ''}"

    @staticmethod
    def _label(d):
        return {"false_alarm": "false_alarm", "resolved": "real"}.get((d["records"].get("closure") or {}).get("outcome"))

    def _memory(self):
        labels, alarms, checks = {}, {}, {}
        for d in self.cases.values():
            if not d["records"].get("closure"):
                continue
            for c in d["records"].get("checks", []):
                n = checks.setdefault(c["check_id"], {"check_id": c["check_id"], "abnormal": 0, "normal": 0})
                n[c["result"]] += 1
            label = self._label(d)
            if label is None:
                continue
            labels[d["case_id"]] = label
            key = self._key(d)
            a = alarms.setdefault(key, {"key": key, "gun_id": d["event"]["gun_id"],
                                        "code": d["event"]["trigger"]["rule_code"], "false_alarm": 0, "real": 0,
                                        "case_ids": []})
            a[label] += 1
            a["case_ids"].append(d["case_id"])
        return {"enabled": True, "labeled": len(labels), "labels": labels,
                "alarms": [alarms[k] for k in sorted(alarms)], "checks": [checks[k] for k in sorted(checks)],
                "manual_causes": [], "missed_checks": []}

    def _field_memory(self, d):
        """api/field_memory.py annotate의 대역: 같은 설비·코드의 사람 라벨(자기 제외)과 R1 문장만."""
        code = d["event"]["trigger"]["rule_code"]
        n = {"false_alarm": 0, "real": 0}
        for o in self.cases.values():
            if o is not d and self._key(o) == self._key(d) and self._label(o):
                n[self._label(o)] += 1
        r1 = bool(code) and n["false_alarm"] > n["real"]
        return {"alarm_key": self._key(d), "alarm": n, "applied": ["R1"] if r1 else [],
                "note_ko": r1_note(d["event"]["gun_id"], code, n["false_alarm"], n["real"]) if r1 else None,
                "checks": {}, "labeled": len(self._memory()["labels"])}

    def get(self, path, params=None):
        self.paths.append(path)
        if path == "/api/health":
            return FakeResponse(200, {"status": "ok", "ml": self.ml_status, "rag": "mapping_only",
                                      **({"memory": "on"} if self.memory_on else {})})
        if path == "/api/cases":
            return FakeResponse(200, {"cases": [self._summary(d) for d in self.cases.values()
                                                if d["status"] not in ("resolved", "false_alarm")]})
        if path == "/api/history":
            for cid, c in list(self.closing.items()):
                c[0] -= 1
                if c[0] == 0:
                    del self.closing[cid]
                    d = self.cases[cid]
                    d["status"] = c[1]
                    d["records"].update(c[2])
                    d["records"]["closure"] = {"outcome": c[1], "at": "2026-10-28T00:00:00Z"}
            return FakeResponse(200, {"items": [{**self._summary(d), "closure": d["records"]["closure"]}
                                                for d in self.cases.values() if d["records"].get("closure")]})
        if path == "/api/memory":
            return FakeResponse(200, self._memory()) if self.memory_on else FakeResponse(404, {})
        d = self.cases[path.rsplit("/", 1)[1]]
        out = dict(d)
        if self.memory_on and self.annotate:
            out["field_memory"] = self._field_memory(d)
        return FakeResponse(200, out)

    def post(self, path, json=None):
        assert path == "/api/ml/sync", path
        self.sync_calls += 1
        if self.empty_syncs:
            self.empty_syncs -= 1
            return FakeResponse(200, {})
        return FakeResponse(200, self.sync())


class ScriptFeeder:
    """미리 정한 항목을 차례로 내는 Feeder 대역. 알림 항목을 낼 때 그 알림의 핸드오프를 ML outbox에 넣는다."""

    def __init__(self, ml, script, ui=None):
        self.ml = ml
        self.script = list(script)  # [(item, [handoff, ...]), ...]
        self.ui = ui  # 주어지면 UI가 먼저 /api/cases로 당겨 간 것처럼 바로 sync한다
        self.released = []
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def next_item(self, timeout=0.5):
        assert self.script, "script exhausted"
        item, hs = self.script.pop(0)
        self.ml.outbox.extend(hs)
        if hs and self.ui is not None:
            self.ui.sync()
        return item

    def release(self, seq):
        self.released.append(seq)

    def stop(self):
        self.stopped = True


class Recorder:
    """on_update 대역: 노드 순서와 그때의 state를 남기고, 중단이 오면 케이스 닫기를 예약한다.
    outcomes[case_id]는 outcome 문자열이나 (outcome, records 추가분) 쌍."""

    def __init__(self, api, outcomes=None):
        self.api = api
        self.outcomes = outcomes or {}
        self.calls = []

    def __call__(self, node, update, state):
        self.calls.append((node, update, state))
        if node == "__interrupt__":
            o = self.outcomes.get(update["case_id"], "false_alarm")
            outcome, records = o if isinstance(o, tuple) else (o, {})
            self.api.close_after(update["case_id"], outcome, **records)

    @property
    def nodes(self):
        return [c[0] for c in self.calls]


def make_run(script, outcomes=None, ui_first=False, **api_kw):
    ml = FakeML()
    api = FakeAPI(ml, **api_kw)
    feeder = ScriptFeeder(ml, script, ui=api if ui_first else None)
    rec = Recorder(api, outcomes)
    deps = hl.Deps(feeder=feeder, ml=ml, api=api, ui_url=UI)
    return deps, api, feeder, rec


ALERT_H1 = [(FILE_ITEM, []), (("alert", 1, "test_0", {}), [handoff("h1")]), (("end",), [])]


def test_loop_waits_and_resumes():
    deps, api, feeder, rec = make_run(ALERT_H1)
    final = hl.run_loop(deps, rec, poll_s=0, thread_id="t1")
    # 이어 간 operator의 {"resume": ...} 청크는 on_update로 넘기지 않는다(브리프 순서)
    assert rec.nodes == ["advance", "ingest", "guide", "__interrupt__",
                         "learn", "evaluate", "advance", "report"]
    assert final["resume"]["closure"]["outcome"] == "false_alarm"  # state에는 남는다
    assert rec.calls[3][1] == {"case_id": "CASE-0001", "url": "http://ui/cases/CASE-0001", "gun_id": "test_0",
                               "code": "E029", "order": ["E029"], "note_ko": None}
    row = final["alerts"][0]
    assert row["status"] == "closed" and row["human_label"] == "false_alarm"
    assert row["case_id"] == "CASE-0001" and row["closure"]["outcome"] == "false_alarm"
    assert row["order"] == ["E029"] and row["applied"] == [] and "baseline" not in row
    assert row["memory_delta"] == {"alarms": [{"key": "test_0|E029", "gun_id": "test_0", "code": "E029",
                                               "false_alarm": 1, "real": 0, "case_ids": ["CASE-0001"]}],
                                   "checks": []}
    assert final["files"]["test_0"] == {"path": "x/test_0.csv", "start": "2021-09-04T00:00:00Z", "end": END_ISO}
    assert final["done"] is True and final["error"] is None and final["queue"] == []
    assert feeder.started and feeder.stopped and feeder.released == [1]
    assert api.closing == {}  # 두 번째 history 조회에서 닫혔다
    # 상세는 guide에서 한 번, 닫힌 뒤 한 번만: 닫힘은 /api/history로 기다린다(상세 GET은 RAG를 부른다)
    assert api.paths.count("/api/cases/CASE-0001") == 2


def test_second_alert_gets_note():
    # 사람이 오알람으로 닫으면 같은 설비·코드의 다음 알림에 R1 문장이 붙는다(카드는 그대로)
    script = [(FILE_ITEM, []), (("alert", 1, "test_0", {}), [handoff("h1")]),
              (("alert", 2, "test_0", {}), [handoff("h2", t="2021-09-05T05:00:00Z")]), (("end",), [])]
    deps, api, feeder, rec = make_run(script)
    final = hl.run_loop(deps, rec, poll_s=0)
    waits = [c[1] for c in rec.calls if c[0] == "__interrupt__"]
    assert [w["order"] for w in waits] == [["E029"], ["E029"]]
    assert waits[0]["note_ko"] is None and waits[1]["note_ko"] == r1_note("test_0", "E029", 1, 0)
    first, second = final["alerts"]
    assert first["applied"] == [] and second["applied"] == ["R1"]
    assert second["memory_delta"]["alarms"][0]["false_alarm"] == 2


def test_checks_memory_delta():
    deps, api, feeder, rec = make_run(
        [(FILE_ITEM, []), (("alert", 1, "test_0", {}), [handoff("h1", code="E012")]), (("end",), [])],
        outcomes={"CASE-0001": ("resolved", {"checks": [{"check_id": "E012-K1", "result": "abnormal"}]})})
    final = hl.run_loop(deps, rec, poll_s=0)
    row = final["alerts"][0]
    assert row["order"] == ["E012"] and row["human_label"] == "real"
    assert row["memory_delta"]["checks"] == [{"check_id": "E012-K1", "abnormal": 1, "normal": 0}]
    assert final["memory"]["checks"] == [{"check_id": "E012-K1", "abnormal": 1, "normal": 0}]


def test_guide_without_field_memory():
    deps, api, feeder, rec = make_run(ALERT_H1, annotate=False)
    final = hl.run_loop(deps, rec, poll_s=0)
    row = final["alerts"][0]
    assert row["order"] == ["E029"] and row["applied"] == [] and row["note_ko"] is None


def test_folded_alert_continues():
    deps, api, feeder, rec = make_run(ALERT_H1)
    api.ingest(handoff("h0", t="2021-09-04T04:00:00Z"))  # 이미 열려 있는 같은 설비·코드 케이스
    final = hl.run_loop(deps, rec, poll_s=0)
    assert rec.nodes == ["advance", "ingest", "advance", "report"]
    row = final["alerts"][0]
    assert row["status"] == "folded" and row["folded_into"] == "CASE-0001" and row["case_id"] is None
    assert feeder.released == [1]


def test_missing_alert_continues(monkeypatch):
    monkeypatch.setattr(hl, "_SYNC_RETRY_S", 0)
    deps, api, feeder, rec = make_run(ALERT_H1)
    api.ml = FakeML()  # api가 ML outbox를 못 본다: sync는 계속 {}
    final = hl.run_loop(deps, rec, poll_s=0)
    assert api.sync_calls == 4  # 처음 1회 + 다시 3회
    assert [r["status"] for r in final["alerts"]] == ["missing"]
    assert "__interrupt__" not in rec.nodes


def test_case_already_ingested_by_ui():
    deps, api, feeder, rec = make_run(ALERT_H1, ui_first=True)
    final = hl.run_loop(deps, rec, poll_s=0)
    assert api.sync_calls == 1  # {"duplicate": 1}은 빈 결과가 아니라 다시 시도하지 않는다
    assert rec.nodes.count("__interrupt__") == 1
    assert final["alerts"][0]["case_id"] == "CASE-0001" and final["alerts"][0]["status"] == "closed"


def test_sync_retried_when_empty(monkeypatch):
    monkeypatch.setattr(hl, "_SYNC_RETRY_S", 0)
    deps, api, feeder, rec = make_run(ALERT_H1, empty_syncs=2)
    final = hl.run_loop(deps, rec, poll_s=0)
    assert api.sync_calls == 3
    assert final["alerts"][0]["status"] == "closed"


def test_two_handoffs_one_alert():
    deps, api, feeder, rec = make_run(
        [(FILE_ITEM, []), (("alert", 1, "test_0", {}), [handoff("h1"), handoff("h2", code="E012")]),
         (("end",), [])], outcomes={"CASE-0002": "resolved"})
    final = hl.run_loop(deps, rec, poll_s=0)
    assert rec.nodes == ["advance", "ingest", "guide", "__interrupt__", "learn", "evaluate",
                         "guide", "__interrupt__", "learn", "evaluate", "advance", "report"]
    assert [c[1]["case_id"] for c in rec.calls if c[0] == "__interrupt__"] == ["CASE-0001", "CASE-0002"]
    assert [(r["case_id"], r["human_label"]) for r in final["alerts"]] == [("CASE-0001", "false_alarm"),
                                                                           ("CASE-0002", "real")]
    assert feeder.released == [1]


def test_data_label_hidden_until_closed():
    deps, api, feeder, rec = make_run(ALERT_H1)
    hl.run_loop(deps, rec, poll_s=0)
    states = {node: state for node, _, state in rec.calls}
    waiting = states["__interrupt__"]["alerts"][0]
    assert waiting["status"] == "waiting" and waiting["data_label"] is None and waiting["ttf_s"] is None
    assert states["learn"]["alerts"][0]["data_label"] is None  # 닫혔어도 evaluate 전에는 없다
    done = states["evaluate"]["alerts"][0]  # on_update의 state는 이 노드의 갱신까지 반영돼 있다
    assert done["data_label"] == "far" and done["ttf_s"] == 25200 and done["agree"] is True


def test_error_item_ends_run():
    deps, api, feeder, rec = make_run([(FILE_ITEM, []), (("error", "Traceback ...\nRuntimeError: boom"), [])])
    final = hl.run_loop(deps, rec, poll_s=0)
    assert "boom" in final["error"] and final["done"] is True
    assert rec.nodes == ["advance", "report"] and final["alerts"] == []
    assert feeder.stopped


def test_wait_for_closure_polls():
    api = FakeAPI(FakeML())
    api.ingest(handoff("h1"))
    judged = [{"situation_id": "E029", "answer": "yes"}]
    results = [{"outcome": "resolved"}]
    api.close_after("CASE-0001", "resolved", polls=3, judgments=judged, results=results)
    out = hl.wait_for_closure(api, "CASE-0001", poll_s=0)
    assert out == {"case_id": "CASE-0001", "closure": {"outcome": "resolved", "at": "2026-10-28T00:00:00Z"},
                   "judgments": judged, "results": results}
    assert api.closing == {}
    # 닫힘은 /api/history로만 기다리고, 상세(GET 때마다 RAG를 부를 수 있다)는 닫힌 뒤 한 번만 읽는다
    assert api.paths == ["/api/history"] * 3 + ["/api/cases/CASE-0001"]
    api.ingest(handoff("h2", code="E012"))
    api.paths = []
    stop = threading.Event()
    stop.set()
    assert hl.wait_for_closure(api, "CASE-0002", poll_s=0, stop=stop)["closure"] is None  # 멈추면 닫힘 없이 돌아온다
    assert api.paths == ["/api/history"]


def test_wait_for_closure_ignores_other_closed_cases():
    api = FakeAPI(FakeML())
    api.ingest(handoff("h1"))
    api.ingest(handoff("h2", code="E012"))
    api.close_after("CASE-0001", "false_alarm", polls=1)
    api.close_after("CASE-0002", "resolved", polls=3)
    assert hl.wait_for_closure(api, "CASE-0002", poll_s=0)["closure"]["outcome"] == "resolved"
    assert api.paths == ["/api/history"] * 3 + ["/api/cases/CASE-0002"]


CLEAN_HEALTH = {"guns_tracked": 0, "rag_url": None, "outbox": {"pending": 0, "delivered": 0, "failed": 0, "pull_only": 0}}


def test_startup_check_clean():
    ml = FakeML()
    api = FakeAPI(ml)
    assert hl.startup_check(ml, api) == []
    assert api.paths == ["/api/health", "/api/cases", "/api/history"]


@pytest.mark.parametrize("health, word", [
    ({**CLEAN_HEALTH, "guns_tracked": 2}, "guns_tracked"),
    ({**CLEAN_HEALTH, "outbox": {"pending": 0, "delivered": 0, "failed": 0, "pull_only": 3}}, "outbox"),
])
def test_startup_check_ml_dirty(health, word):
    ml = FakeML(health=health)
    api = FakeAPI(ml)
    problems = hl.startup_check(ml, api)
    assert len(problems) == 1 and word in problems[0]
    assert "새 RSW_STATE_DIR 폴더(예: state/d8-<새 이름>)로 ML 서버를 다시 켜세요" in problems[0]
    assert "비우고" not in problems[0]
    assert api.paths == ["/api/health"]  # ML이 더러우면 /api/cases를 부르지 않는다(api가 옛 핸드오프를 당겨 감)


def test_startup_check_rag_url():
    ml = FakeML(health={**CLEAN_HEALTH, "rag_url": "http://rag"})
    problems = hl.startup_check(ml, FakeAPI(ml))
    assert len(problems) == 1 and "RSW_RAG_URL" in problems[0]


def test_startup_check_api_problems():
    ml = FakeML()
    off = hl.startup_check(ml, FakeAPI(ml, memory_on=False))
    assert len(off) == 1 and "memory" in off[0] and "FIELD_MEMORY=1" in off[0]
    down = hl.startup_check(ml, FakeAPI(ml, ml_status="down"))
    assert len(down) == 1 and "ML_URL" in down[0]


@pytest.mark.parametrize("closure_status", ["new", "false_alarm"])
def test_startup_check_api_not_empty(closure_status):
    ml = FakeML()
    api = FakeAPI(ml)
    api.ingest(handoff("h0"))
    if closure_status == "false_alarm":  # 닫힌 케이스는 history에만 있다
        api.close_after("CASE-0001", "false_alarm", polls=1)
        api.get("/api/history")
        api.paths = []
    problems = hl.startup_check(ml, api)
    assert len(problems) == 1
    assert ("cases" if closure_status == "new" else "history") in problems[0]
    assert "새 API_DB 파일(예: state/d8-<새 이름>.db)로 api 서버를 다시 켜세요" in problems[0]
    assert "비우고" not in problems[0]


def test_startup_check_unreachable():
    class Down:
        base_url = "http://127.0.0.1:8000"

        def get(self, path, params=None):
            raise httpx.ConnectError("connection refused")

    ml = FakeML()
    problems = hl.startup_check(Down(), FakeAPI(ml))
    assert len(problems) == 1 and "http://127.0.0.1:8000" in problems[0] and "먼저 그 서버를 켜세요" in problems[0]
    problems = hl.startup_check(ml, Down())
    assert len(problems) == 1 and "api" in problems[0] and "먼저 그 서버를 켜세요" in problems[0]
    assert "Traceback" not in " ".join(problems)


def test_wait_for_closure_rides_out_api_errors():
    api = FakeAPI(FakeML())
    api.ingest(handoff("h1"))
    api.close_after("CASE-0001", "false_alarm", polls=1, judgments=[{"situation_id": "E029", "answer": "no"}])
    errors = {"/api/history": [httpx.ConnectError("refused"),
                               httpx.Response(503, request=httpx.Request("GET", "http://api/x"))],
              "/api/cases/CASE-0001": [httpx.ReadTimeout("slow RAG"),
                                       httpx.Response(502, request=httpx.Request("GET", "http://api/x"))]}
    get = api.get

    def flaky(path, params=None):
        if errors.get(path):
            e = errors[path].pop(0)
            if isinstance(e, Exception):
                raise e
            return e
        return get(path, params)

    api.get = flaky
    out = hl.wait_for_closure(api, "CASE-0001", poll_s=0)
    assert out["closure"]["outcome"] == "false_alarm" and out["judgments"] == [{"situation_id": "E029", "answer": "no"}]
    assert errors == {"/api/history": [], "/api/cases/CASE-0001": []}
    # 닫힘을 본 뒤에는 history를 다시 부르지 않고 상세만 다시 시도한다
    assert api.paths == ["/api/history", "/api/cases/CASE-0001"]
    api.get = lambda path, params=None: httpx.Response(404, request=httpx.Request("GET", "http://api/x"))
    with pytest.raises(httpx.HTTPStatusError):  # 4xx는 그대로 올린다
        hl.wait_for_closure(api, "CASE-0001", poll_s=0)


def test_handover_fold_warns():
    # 도움 요청으로 넘긴 케이스는 열린 채 남아 다음 같은 알림을 반복으로 묶는다: 경고를 남기고 멈추지 않는다
    script = [(FILE_ITEM, []), (("alert", 1, "test_0", {}), [handoff("h1")]),
              (("alert", 2, "test_0", {}), [handoff("h2", t="2021-09-05T05:00:00Z")]), (("end",), [])]
    deps, api, feeder, rec = make_run(script, outcomes={"CASE-0001": "handed_over"})
    final = hl.run_loop(deps, rec, poll_s=0)
    assert rec.nodes.count("__interrupt__") == 1
    first, second = final["alerts"]
    assert first["closure"]["outcome"] == "handed_over" and first["human_label"] is None
    assert "CASE-0001를 도움 요청으로 넘겼습니다" in first["warn"]
    assert second["status"] == "folded" and second["folded_into"] == "CASE-0001"
    assert "CASE-0001가 도움 요청 상태라" in second["warn"] and "오알람으로 닫기" in second["warn"]
    view = hl.render_view(final, None, None, UI, None)
    rows = [r for r in view.split("<tr>") if "2021-09-05T05:00:00Z" in r]
    assert rows and f'<td style="color:{hl.AMBER}">반복으로 묶임 (CASE-0001)</td>' in rows[0]
    assert view.count("⚠ ") == 2
    md = hl.render_summary(final)
    assert "## 경고" in md and md.count("- ⚠ ") == 2
    assert "## 경고" not in hl.render_summary(view_state())


# ---- 현황·요약 출력과 CLI (Task 6) ----

FOOTER = "재생 데이터 시연 · 현장 아님 · 데이터 기준 라벨은 답을 아는 것이라 성능 아님"


def view_state():
    """닫힌 행 하나(CASE-0001)와 사람 확인을 기다리는 행 하나(CASE-0002)."""
    closed = hl._row(handoff("h1"), "closed", case_id="CASE-0001")
    closed.update(order=["E029"], applied=["R1"], note_ko="R1 · 오알람 1건 <참고>", human_label="false_alarm",
                  data_label="far", ttf_s=25200.0, agree=True, closure={"outcome": "false_alarm"},
                  memory_delta={"alarms": [{"key": "test_0|E029", "gun_id": "test_0", "code": "E029",
                                            "false_alarm": 1, "real": 0, "case_ids": ["CASE-0001"]}],
                                "checks": [{"check_id": "E029-K1<", "abnormal": 0, "normal": 1}]})
    waiting = hl._row(handoff("h2", t="2021-09-04T11:30:00Z"), "waiting", case_id="CASE-0002")
    waiting["order"] = ["E029"]
    waiting["data_label"] = "pre_failure"  # 닫히기 전에는 값이 있어도 보이지 않아야 한다
    memory = {"labeled": 1, "labels": {"CASE-0001": "false_alarm"},
              "alarms": [{"key": "test_0|E029", "gun_id": "test_0", "code": "E029", "false_alarm": 1, "real": 0,
                          "case_ids": ["CASE-0001"]}],
              "checks": [{"check_id": "E012-K1", "abnormal": 1, "normal": 0},
                         {"check_id": "E029-K1<", "abnormal": 0, "normal": 1}]}
    return {"files": {}, "seen": ["h1", "h2"], "queue": ["h2"], "alerts": [closed, waiting], "memory": memory,
            "seq": 2, "resume": None, "done": False, "error": None}


def test_write_atomic_utf8(tmp_path):
    p = tmp_path / "v.html"
    hl.write_atomic(str(p), "⏸ 사람 확인")
    hl.write_atomic(str(p), "⏸ 사람 확인")  # 덮어쓰기
    assert p.read_text(encoding="utf-8") == "⏸ 사람 확인"
    assert os.listdir(tmp_path) == ["v.html"]  # tmp 파일이 남지 않는다


def test_view_utf8_korean(tmp_path):
    state = view_state()
    waiting = {"case_id": "CASE-0002", "url": "http://localhost:5173/cases/CASE-0002", "gun_id": "test_0",
               "code": "E029", "order": ["E029"], "note_ko": None}
    waiting["note_ko"] = "같은 설비·코드(test_0 E029)에서 사람이 오알람으로 닫은 기록 1건 <참고> & 확인"
    html = hl.render_view(state, "operator", waiting, "http://localhost:5173", "2021-09-04T11:30:00Z")
    assert '<meta charset="utf-8">' in html and '<meta http-equiv="refresh" content="2">' in html
    assert "⏸ 사람 확인 대기" in html and "CASE-0002" in html
    assert '<a href="http://localhost:5173/cases/CASE-0002" target="d8-ui">' in html  # 링크는 탭 하나를 다시 쓴다
    assert "· 안내 카드 E029</p>" in html and "후보 순서" not in html
    # 안내 문장은 대기 줄 아래에 escape해서
    assert ('<p class="note">같은 설비·코드(test_0 E029)에서 사람이 오알람으로 닫은 기록 1건 &lt;참고&gt; &amp; 확인</p>'
            in html)
    assert html.count("<a ") == 1  # 행마다 링크는 없다
    assert FOOTER in html
    assert "#2563eb" in html  # 현재 노드
    assert "&lt;참고&gt;" in html and "<참고>" not in html and "E029-K1<" not in html  # 동적 문자열은 escape
    cells = {c[5]: c for c in (re.findall(r"<td[^>]*>(.*?)</td>", r) for r in html.split("<tr>")) if len(c) == 14}
    assert cells["CASE-0002"][10] == "—" and "pre_failure" not in html and "고장 직전" not in html  # 대기 행은 —
    assert cells["CASE-0001"][10] == "먼 알림" and cells["CASE-0001"][9] == "오알람"
    assert cells["CASE-0001"][8] == "오알람으로 닫음"
    assert cells["CASE-0001"][6] == cells["CASE-0002"][6] == "E029"  # 안내 카드
    assert cells["CASE-0001"][7] == "R1"  # 기억 반영
    assert cells["CASE-0001"][12] == "R1 · 오알람 1건 &lt;참고&gt;"  # 안내 문장
    assert cells["CASE-0001"][13] == "test_0|E029 오알람 1 · 고장 확인 0; E029-K1&lt; 이상 0 · 정상 1"
    assert cells["CASE-0002"][7] == cells["CASE-0002"][12] == cells["CASE-0002"][13] == "—"
    assert "<th>오알람</th><th>고장 확인</th>" in html and "오경보" not in html
    assert "<th>점검 항목</th><th>이상</th><th>정상</th>" in html
    assert "<tr><td>E012-K1</td><td>1</td><td>0</td></tr>" in html
    assert "<tr><td>E029-K1&lt;</td><td>0</td><td>1</td></tr>" in html
    assert "지난 확정 원인" not in html and "기본 순서" not in html
    p = tmp_path / "loop_view.html"
    hl.write_atomic(str(p), html)
    assert p.read_text(encoding="utf-8") == html

    playing = hl.render_view(state, "advance", None, "http://localhost:5173", "2021-09-04T11:30:00Z")
    assert "재생 중 · 마지막 알림 데이터 시각 2021-09-04T11:30:00Z" in playing and "⏸ 사람 확인 대기" not in playing


def test_view_ended():
    state = view_state()
    done = hl.render_view(state, None, None, UI, None, ended="끝남 · 알림 2건 · 요약 summary.md")
    assert f'<p class="status" style="color:{hl.GRAY}">끝남 · 알림 2건 · 요약 summary.md</p>' in done
    assert blue_box(done) == [] and "<a " not in done and "재생 중" not in done
    stopped = hl.render_view(state, None, None, UI, None, ended="중단됨: 사용자가 멈춤(Ctrl+C)")
    assert f'<p class="status" style="color:{hl.RED}">중단됨: 사용자가 멈춤(Ctrl+C)</p>' in stopped


def blue_box(html):
    """노드 사슬에서 파랑(지금 도는 노드) 상자의 이름들."""
    return re.findall(r'<span class="node" style="background:#2563eb;[^"]*">(.*?)</span>', html)


def test_view_highlights_running_node():
    # on_update는 노드가 끝난 뒤 불린다: evaluate 뒤 queue가 비면 그래프는 advance(긴 재생)를 돌고 있다
    state = view_state()
    state["queue"] = []
    current = hl.running_node("evaluate", state)
    assert current == "advance"
    assert blue_box(hl.render_view(state, current, None, "http://localhost:5173", None)) == ["재생 진행"]
    assert hl.running_node("ingest", state) == "advance"
    state["queue"] = ["h2"]
    assert hl.running_node("evaluate", state) == "guide" and hl.running_node("ingest", state) == "guide"
    assert hl.running_node("advance", {"done": False}) == "ingest"
    assert hl.running_node("advance", {"done": True}) == "report"
    assert hl.running_node("guide", state) == "operator" and hl.running_node("__interrupt__", state) == "operator"
    assert hl.running_node("learn", state) == "evaluate"
    assert hl.running_node("report", {"done": True}) is None
    assert blue_box(hl.render_view(state, None, None, "http://localhost:5173", None)) == []


def test_summary_has_rows_and_footer():
    md = hl.render_summary(view_state())
    cells = {c[5]: c for c in (ln.strip("| ").split(" | ") for ln in md.splitlines()) if len(c) == 14}
    assert cells["CASE-0001"][6:11] == ["E029", "R1", "오알람으로 닫음", "오알람", "먼 알림"]
    assert cells["CASE-0002"][10] == "—" and "pre_failure" not in md
    assert "| " + " | ".join(hl._COLS) + " |" in md
    assert "| 설비\\|코드 | 오알람 | 고장 확인 | 케이스 |" in md
    assert r"| test_0\|E029 | 1 | 0 | CASE-0001 |" in md  # 기억 (칸 안의 |는 escape)
    assert "| 점검 항목 | 이상 | 정상 |" in md and "| E012-K1 | 1 | 0 |" in md
    assert FOOTER in md


def test_columns():
    assert hl._COLS == ["데이터 시각", "설비", "코드", "출처", "상태", "케이스", "안내 카드", "기억 반영", "닫힘",
                        "사람 라벨", "데이터 라벨", "일치", "안내 문장", "기억 변화"]
    assert hl._COLS[4] == "상태"  # _html_table이 글자색을 칠하는 칸
    assert hl._MEM_COLS == ["설비|코드", "오알람", "고장 확인", "케이스"]
    assert hl._CHECK_COLS == ["점검 항목", "이상", "정상"]
    row = hl._row(handoff("h1"), "waiting", case_id="CASE-0001")
    assert "baseline" not in row and len(hl._cells(row)) == len(hl._COLS)


class DryFeeder:
    """드라이런용 Feeder 대역: 파일 하나, 알림 두 개, 끝."""
    made = []

    def __init__(self, paths, make_client=None, pause=True, **kw):
        self.paths, self.pause = paths, pause
        self.items = [("file", "test_0", paths[0], "2021-09-04T00:00:00Z", END_ISO),
                      ("alert", 1, "test_0", {"window_end": "2021-09-04T05:00:00Z", "error_code": "E029"}),
                      ("alert", 2, "test_0", {"window_end": "2021-09-04T06:00:00Z", "error_code": "E012"}),
                      ("end",)]
        DryFeeder.made.append(self)

    def start(self):
        pass

    def next_item(self, timeout=0.5):
        return self.items.pop(0) if self.items else None

    def stop(self):
        pass

    def join(self, timeout=None):
        pass


def test_cli_dry_run_counts(tmp_path, monkeypatch, capsys):
    csv = write_csv(tmp_path / "test_0.csv", ["2021-09-04 00:00:00"])
    outbox = [handoff(f"h{i}") for i in range(5)] + [handoff("h5", code="E012")]
    clients = []

    class Client(FakeML):
        def __init__(self, base_url, timeout=None):
            super().__init__(outbox)
            self.base_url = base_url
            self.params = []
            clients.append(self)

        def get(self, path, params=None):
            self.params.append(params)
            return super().get(path, params)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    DryFeeder.made = []
    monkeypatch.setattr(hl, "Feeder", DryFeeder)
    monkeypatch.setattr(hl, "httpx", type("httpx", (), {"Client": Client}))
    monkeypatch.setattr(sys, "argv", ["hitl_loop.py", csv, "--dry-run", "--ml", "http://ml", "--api", "http://api",
                                      "--out", str(tmp_path / "out")])
    hl.main()
    out = capsys.readouterr().out
    assert "test_0: 6 (rule E029 x5, rule E012 x1)" in out
    assert DryFeeder.made[0].pause is False
    assert [c.base_url for c in clients] == ["http://ml"]  # api는 부르지 않는다
    assert clients[0].params[0]["gun_id"] == "test_0"
    assert not (tmp_path / "out").exists()  # 드라이런은 산출물을 쓰지 않는다


class NullClient:
    """main()의 httpx.Client 대역: 아무 것도 부르지 않는다(startup_check·run_loop는 테스트가 바꿔 낀다)."""

    def __init__(self, base_url=None, timeout=None):
        self.base_url = base_url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_main(tmp_path, monkeypatch, fake_run_loop, problems=()):
    csv = write_csv(tmp_path / "test_0.csv", ["2021-09-04 00:00:00"])
    out = tmp_path / "out"
    monkeypatch.setattr(hl, "httpx", type("httpx", (), {"Client": NullClient}))
    monkeypatch.setattr(hl, "startup_check", lambda ml, api: list(problems))
    monkeypatch.setattr(hl, "Feeder", lambda *a, **kw: object())
    monkeypatch.setattr(hl, "run_loop", fake_run_loop)
    monkeypatch.setattr(sys, "argv", ["hitl_loop.py", csv, "--ui", "http://ui/", "--out", str(out)])
    return out


def interrupted_state():
    state = view_state()
    return state, {"case_id": "CASE-0002", "url": "http://ui/cases/CASE-0002", "gun_id": "test_0", "code": "E029",
                   "order": ["E029"], "note_ko": "오알람으로 닫은 기록 1건"}


def test_main_startup_failure_writes_nothing(tmp_path, monkeypatch):
    out = run_main(tmp_path, monkeypatch, lambda deps, on_update: {}, problems=["ML 서버에 연결하지 못했습니다"])
    with pytest.raises(SystemExit, match="시작 점검 실패"):
        hl.main()
    assert not out.exists()  # 점검 전에는 산출물 폴더를 만들지 않는다


def test_main_normal_end(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_loop(deps, on_update):
        seen["ui"] = deps.ui_url
        state, payload = interrupted_state()
        on_update("__interrupt__", payload, state)
        return {**state, "queue": [], "done": True}

    out = run_main(tmp_path, monkeypatch, fake_run_loop)
    out.mkdir()
    (out / "loop_log.jsonl").write_text("old run\n", encoding="utf-8")
    hl.main()
    console = capsys.readouterr().out
    assert seen["ui"] == "http://ui" and "UI http://ui\n" in console  # 끝의 /를 뗀다
    assert "  ⏸ CASE-0002 사람 확인 대기 · 안내 카드 E029: http://ui/cases/CASE-0002" in console
    assert "\n      오알람으로 닫은 기록 1건\n" in console
    log = (out / "loop_log.jsonl").read_text(encoding="utf-8")
    assert "old run" not in log and log.count("\n") == 1  # 시작할 때 비운다
    view = (out / "loop_view.html").read_text(encoding="utf-8")
    assert f'<p class="status" style="color:{hl.GRAY}">끝남 · 알림 2건 · 요약 summary.md</p>' in view
    assert blue_box(view) == [] and "<a " not in view
    assert (out / "summary.md").exists() and (out / "workflow.mmd").exists()


def test_main_error_noted_and_reraised(tmp_path, monkeypatch, capsys):
    calls = []
    real_write = hl.write_atomic

    def flaky_write(path, text, retries=5):
        calls.append(path)
        if len(calls) == 1:  # 첫 현황 쓰기가 실패해도 실행은 이어 간다
            raise PermissionError("locked")
        real_write(path, text, retries)

    def fake_run_loop(deps, on_update):
        state, payload = interrupted_state()
        on_update("__interrupt__", payload, state)
        on_update("learn", {}, state)
        raise RuntimeError("HTTP 500 boom")

    out = run_main(tmp_path, monkeypatch, fake_run_loop)
    monkeypatch.setattr(hl, "write_atomic", flaky_write)
    with pytest.raises(RuntimeError, match="boom"):
        hl.main()
    assert "  ⚠ 현황 파일을 쓰지 못했습니다(locked)" in capsys.readouterr().out
    view = (out / "loop_view.html").read_text(encoding="utf-8")
    assert f'<p class="status" style="color:{hl.RED}">중단됨: RuntimeError: HTTP 500 boom</p>' in view
    assert blue_box(view) == [] and "<a " not in view
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "재생이 오류로 끝났습니다" in summary and "Traceback" in summary and "boom" in summary


def test_main_ctrl_c(tmp_path, monkeypatch):
    def fake_run_loop(deps, on_update):
        raise KeyboardInterrupt

    out = run_main(tmp_path, monkeypatch, fake_run_loop)
    with pytest.raises(KeyboardInterrupt):
        hl.main()
    assert "중단됨: 사용자가 멈춤(Ctrl+C)" in (out / "loop_view.html").read_text(encoding="utf-8")
    assert "사용자가 멈춤(Ctrl+C)" in (out / "summary.md").read_text(encoding="utf-8")
