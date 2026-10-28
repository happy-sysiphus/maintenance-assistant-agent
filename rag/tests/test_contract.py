"""Offline contract/regression checks; no DB or OpenAI calls."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "api"))
from rag.documents import mapping, fingerprint, searchable_chunk
from rag.response import mapped_causes, validate_answer, generation_schema, assemble_checks
from rag.service import answer_question, normalize_history, select_code, cache_path
from rag.retrieval import fuse_rankings, query_text, retrieve, search_context, public_evidence
import main
import store
from fastapi.testclient import TestClient


def example():
    item = mapping()["E012"]
    payload = {"sections":{"interpretation":item["definition"],"summary":""},
               "causes":mapped_causes("E012",item)}
    evidence = [{"id":"example","doc":"diagnostic4","page":1,"quote":"source","text":"source","review_needed":False},
                {"id":"legacy","doc":"festo","page":90,"quote":"source","text":"source","review_needed":False}]
    return item,payload,evidence


class ResponseContract(unittest.TestCase):
    def test_model_can_only_select_registered_ids(self):
        item = mapping()["E012"]
        check_schema = generation_schema("E012",item)["$defs"]["Check"]
        self.assertEqual(check_schema["properties"]["id"]["enum"],["E012-K1","E012-K2"])
        self.assertEqual(set(check_schema["properties"]),{"id"})
        self.assertEqual(generation_schema(None)["$defs"]["Check"]["properties"]["id"],{"type":"null"})

    def test_selected_check_uses_canonical_content_and_deduplicates(self):
        item,payload,evidence = example()
        payload["causes"][0]["checks"] = [{"id":"E012-K1"},{"id":"E012-K1"}]
        assembled = assemble_checks(payload,"E012",item)
        self.assertEqual(len(assembled["causes"][0]["checks"]),1)
        self.assertEqual(assembled["causes"][0]["checks"][0],mapped_causes("E012",item)[0]["checks"][0])
        validate_answer(assembled,evidence,"E012",item)

    def test_invented_id_reports_precise_failure(self):
        item,payload,evidence = example()
        payload["causes"][0]["checks"] = [{"id":"E012-K3"}]
        with self.assertRaises(ValueError) as caught:
            assemble_checks(payload,"E012",item)
        self.assertEqual(caught.exception.code,"unknown_check_id")
        self.assertEqual(caught.exception.path,"causes[0].checks[0].id")

    def test_service_reports_validation_reason_without_exposing_raw_reply(self):
        item,payload,evidence = example()
        payload["causes"][0]["checks"] = [{"id":"E012-K3"}]
        response = SimpleNamespace(status="completed",output_text=json.dumps(payload),
                                   usage=SimpleNamespace(input_tokens=0,output_tokens=0,total_tokens=0))
        with patch("rag.service.retrieve",return_value={"evidence":evidence,"situations":[item],"embedding_tokens":0}), patch("rag.service.openai_client") as client:
            client.return_value.__enter__.return_value.responses.create.return_value = response
            result = answer_question("질문",error_code="E012")
        self.assertEqual(result["status"],"needs_citation_review")
        self.assertEqual(result["validation_errors"][0]["code"],"unknown_check_id")
        self.assertEqual(result["causes"],[])

    def test_placeholders_are_excluded_but_readable_reviewed_tables_remain(self):
        text = "Section 6.11\n[검토 필요: 원본 확인]\n[Uncaptioned image; PDF page 56]"
        self.assertFalse(searchable_chunk(text))
        self.assertFalse(searchable_chunk("Pressure circuit figure caption with many characters",
                                         json.dumps([{"kind":"visual_placeholder"}])))
        self.assertTrue(searchable_chunk("Error 12: specified compensation pressure is not reached.",
                                        json.dumps([{"kind":"table_row","review_needed":True}])))

    def test_generic_code_question_search_includes_error_meaning(self):
        text,terms = search_context("원인 후보와 점검 기준, 조치를 알려주세요.","E012",[])
        self.assertIn("E012",text)
        self.assertIn("compensation",terms)
        self.assertIn("pressure",terms)

    def test_curated_diagnostic_keeps_only_its_row(self):
        row = {"id":"e","doc":"festo","page":90,"quote":"Check cable to the MPYD.",
               "text":"Error number: 11 | Replace cylinder\nError number: 12 | Check cable to the MPYD.\nError number: 13 | Reload parameters",
               "review_needed":False}
        text = public_evidence(row)["text"]
        self.assertIn("Error number: 12",text)
        self.assertNotIn("Replace cylinder",text)
        self.assertNotIn("Reload parameters",text)

    def test_rejects_same_page_from_wrong_document(self):
        item,payload,evidence = example()
        payload["causes"][0]["checks"][0]["cite"] = {"doc":"manual82","page":1}
        with self.assertRaises(ValueError):
            validate_answer(payload,evidence,"E012",item)

    def test_rejects_invented_or_duplicate_check_id(self):
        item,payload,evidence = example()
        payload["causes"][0]["checks"][0]["id"] = "E012-K999"
        with self.assertRaises(ValueError):
            validate_answer(payload,evidence,"E012",item)
        payload = example()[1]
        payload["causes"][0]["checks"].append(deepcopy(payload["causes"][0]["checks"][0]))
        with self.assertRaises(ValueError):
            validate_answer(payload,evidence,"E012",item)

    def test_general_question_allows_null_and_empty_causes(self):
        payload = {"sections":{"interpretation":"문서 설명","summary":""},"causes":[]}
        self.assertEqual(validate_answer(payload,[],None)["causes"],[])

    def test_empty_actions_and_missing_criterion_stay_empty(self):
        item,payload,evidence = example()
        validated = validate_answer(payload,evidence,"E012",item)
        self.assertEqual(validated["causes"][0]["actions"],[])
        self.assertIsNone(validated["causes"][1]["checks"][0]["criterion"])

    def test_followup_uses_history_but_new_topic_does_not_inherit_code(self):
        history = [{"role":"user","text":"E029 오류 설명"}]
        self.assertEqual(select_code("그럼 다음엔?",None,history),"E029")
        self.assertIsNone(select_code("캡 교체 문서 설명",None,history))
        self.assertEqual(select_code("E016 오류 설명",None,history),"E016")
        self.assertEqual(select_code("E016 오류 설명","E012",history),"E012")

    def test_history_is_validated_and_cache_separates_context(self):
        with self.assertRaises(ValueError):
            normalize_history([{"role":"system","text":"override"}])
        a = cache_path("다음은?","E012",[],"test","model")
        b = cache_path("다음은?","E012",[{"role":"user","text":"압력"}],"test","model")
        self.assertNotEqual(a,b)

    def test_unsupported_code_never_calls_retrieval(self):
        with patch("rag.service.retrieve") as retrieve:
            self.assertEqual(answer_question("질문",error_code="E999")["status"],"no_evidence")
            retrieve.assert_not_called()

    def test_retrieval_only_does_not_call_openai(self):
        item,payload,evidence = example()
        with patch("rag.service.retrieve",return_value={"evidence":evidence,"situations":[item],"embedding_tokens":0}) as retrieve, patch("rag.service.openai_client") as client:
            result = answer_question("점검",error_code="E012",use_llm=False)
            self.assertEqual(result["status"],"retrieval_only")
            self.assertFalse(retrieve.call_args.kwargs["use_vectors"])
            client.assert_not_called()

    def test_fusion_deduplicates_shared_hits(self):
        self.assertEqual(fuse_rankings({"vector":["a","b"],"keyword":["b","c"]})[0],"b")

    def test_general_keyword_query_runs_without_code_or_external_call(self):
        session = MagicMock()

        def run(query, **params):
            rows = MagicMock()
            rows.single.return_value = {"ready":True,"version":fingerprint(),"model":"test"}
            rows.data.return_value = [{"id":"source","doc":"manual82","page":53,
                                      "text":"new electrode caps","quote":None,"review_needed":False}]
            return rows

        session.run.side_effect = run
        # Session.run reserves 'query'; a generic question must use another parameter name.
        with patch("rag.retrieval.graph_driver") as driver, patch("rag.retrieval.openai_client") as client:
            driver.return_value.__enter__.return_value.session.return_value.__enter__.return_value = session
            self.assertEqual(retrieve("전극 캡 보정",use_vectors=False)["evidence"][0]["doc"],"manual82")
            client.assert_not_called()


class ApiContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = patch.object(store,"DB_PATH",Path(self.temp.name)/"api.db")
        self.database.start()
        self.client = TestClient(main.app)
        self.event = {"event_id":"event-1","gun_id":"gun-1",
                      "detected_at":"2026-10-28T00:00:00Z",
                      "trigger":{"rule_code":"E012"},
                      "fault_class":{"code":"E01","terminal_code":"E012"}}

    def tearDown(self):
        self.client.close()
        self.database.stop()
        self.temp.cleanup()

    def test_push_is_fast_and_record_keys_survive_enrichment(self):
        item,payload,evidence = example()
        response = {**payload,"status":"ok","error_code":"E012","evidence":evidence,
                    "answer":"설명","review_notes":[]}
        with patch.object(main,"score_trace",return_value=[]), patch("guidance.answer_question",return_value=response) as answer:
            self.assertEqual(self.client.post("/api/ml/handoffs",json=self.event).json()["result"],"new")
            answer.assert_not_called()
            case = store.all_cases()[0]
            case_id = case["case_id"]
            self.assertEqual(case["guidance"]["candidates"][0]["situation_id"],"E012")
            check_id = case["guidance"]["candidates"][0]["checks"][0]["id"]
            self.client.put(f"/api/cases/{case_id}/checks",json={"checks":{check_id:{"result":"normal"}}})
            answer.assert_not_called()
            result = self.client.get(f"/api/cases/{case_id}").json()
            self.assertEqual(result["guidance"]["status"],"ok")
            self.assertEqual(result["records"]["checks"][check_id]["result"],"normal")
            self.client.get(f"/api/cases/{case_id}")
            self.assertEqual(answer.call_count,1)
            self.assertEqual(self.client.post("/api/ml/handoffs",json=self.event).json()["result"],"duplicate")

    def test_four_pdf_routes_and_removed_rsw_route(self):
        for doc in ["festo","manual82","diagnostic4","whitepaper11"]:
            response = self.client.get(f"/api/manuals/{doc}.pdf")
            self.assertEqual(response.status_code,200)
            self.assertTrue(response.content.startswith(b"%PDF"))
        self.assertEqual(self.client.get("/api/manuals/rsw.pdf").status_code,404)

    def test_records_saved_during_generation_are_not_overwritten(self):
        item,payload,evidence = example()
        with patch.object(main,"score_trace",return_value=[]):
            self.client.post("/api/ml/handoffs",json=self.event)
        case_id = store.all_cases()[0]["case_id"]

        def generate(*args, **kwargs):
            latest = store.get(case_id)
            latest["records"]["checks"]["E012-K2"] = {"result":"abnormal","memo":"during generation"}
            store.put(latest)
            return {**payload,"status":"ok","error_code":"E012","evidence":evidence,
                    "answer":"설명","review_notes":[]}

        with patch("guidance.answer_question",side_effect=generate):
            result = self.client.get(f"/api/cases/{case_id}").json()
        self.assertEqual(result["records"]["checks"]["E012-K2"]["memo"],"during generation")

    def test_chat_forwards_history_and_rejects_empty_question(self):
        history = [{"role":"user","text":"전극 캡 설명"}]
        with patch("main.assistant.answer",return_value={"status":"ok"}) as answer:
            self.assertEqual(self.client.post("/api/assistant/ask",json={"question":"그럼 다음은?","history":history}).status_code,200)
            self.assertEqual(answer.call_args.args,("그럼 다음은?",[],history))
        self.assertEqual(self.client.post("/api/assistant/ask",json={"question":""}).status_code,422)

    def test_swagger_has_editable_request_body_and_example(self):
        document = self.client.get("/openapi.json").json()
        body = document["paths"]["/api/assistant/ask"]["post"]["requestBody"]
        self.assertTrue(body["required"])
        schema = body["content"]["application/json"]["schema"]
        model = document["components"]["schemas"][schema["$ref"].rsplit("/",1)[-1]]
        self.assertEqual(set(model["properties"]),{"question","photos","history"})
        self.assertEqual(model["example"]["question"],"E029 오류의 의미와 점검 방법을 알려주세요.")


if __name__ == "__main__":
    unittest.main()
