"""Offline checks for budget enforcement, dataset constraints and reporting."""
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from budget import Budget, BudgetExceeded
from report import build_report, factual_breakdown
from run import QA, validate_qa, source_packs


class EvaluationChecks(unittest.TestCase):
    def test_factual_breakdown_matches_recorded_official_f1(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            verdicts = [[1, 1, 0, 0], [1, 1]]
            messages = [{"sample_id": "q1", "metric": "factual_correctness",
                         "message": {"content": json.dumps({"statements": [{"verdict": v} for v in vs]})}}
                        for vs in verdicts]
            (p / "judge_traces.jsonl").write_text("".join(json.dumps(m) + "\n" for m in messages))
            result = factual_breakdown(p, [{"id": "q1", "metric": "factual_correctness", "value": 0.67}])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["reference_claim_coverage_ratio"], 1)
            self.assertEqual(result[0]["response_reference_support_ratio"], .5)

    def test_source_packs_keep_error_rows_separate(self):
        packs = source_packs()
        self.assertEqual(sum(p["count"] for p in packs), 50)
        self.assertEqual(sum(p["count"] for p in packs if not p["code"]), 18)
        for p in packs:
            if not p["code"]:
                continue
            diagnostic = next(s["text"] for s in p["sources"] if s["doc"] == "diagnostic4")
            self.assertTrue(diagnostic.startswith("Error " + str(int(p["code"][1:])) + ":"))
            self.assertNotIn("Memory Error", diagnostic)
            self.assertNotIn("Direction of Motion", diagnostic)

    def test_official_metric_constructors_without_network(self):
        import asyncio
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import (Faithfulness, AnswerRelevancy, ContextPrecision,
                                               ContextRecall, FactualCorrectness)
        async def check():
            async with AsyncOpenAI(api_key="offline-placeholder") as client:
                llm = llm_factory("gpt-4.1-mini", client=client, temperature=0, max_tokens=4096, max_retries=1)
                emb = embedding_factory("openai", model="text-embedding-3-small", client=client)
                metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb),
                           ContextPrecision(llm=llm), ContextRecall(llm=llm), FactualCorrectness(llm=llm)]
                self.assertEqual(len(metrics), 5)
                self.assertTrue(all(callable(m.ascore) for m in metrics))
        asyncio.run(check())

    def test_auth_error_stops_following_requests(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d) / "costs.jsonl")
            request = httpx.Request("POST", "https://api.openai.com/v1/responses", json={"model": "gpt-5.4-mini"})
            data, reserve = b.reserve(request)
            b.settle(data, reserve, httpx.Response(401, json={"error": {"code": "token_invalidated"}}))
            with self.assertRaises(RuntimeError):
                b.reserve(request)

    def test_budget_blocks_before_request(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d) / "costs.jsonl", limit=0.000001)
            request = httpx.Request("POST", "https://api.openai.com/v1/responses",
                                    json={"model": "gpt-5.4-mini", "max_output_tokens": 4096})
            with self.assertRaises(BudgetExceeded):
                b.reserve(request)
            self.assertEqual(b.spent, 0)

    def test_usage_counts_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "costs.jsonl"
            b = Budget(p)
            request = httpx.Request("POST", "https://api.openai.com/v1/responses",
                                    json={"model": "gpt-5.4-mini"})
            data, reserve = b.reserve(request)
            b.settle(data, reserve, httpx.Response(200, json={"usage": {"input_tokens": 1000, "output_tokens": 100}}))
            self.assertAlmostEqual(b.spent, 0.0012)
            self.assertEqual(b.reserved, 0)
            self.assertAlmostEqual(Budget(p).spent, b.spent)
            self.assertNotIn("request", p.read_text())

    def test_unknown_model_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d) / "costs.jsonl")
            with self.assertRaises(ValueError):
                b.reserve(httpx.Request("POST", "https://api.openai.com/v1/responses", json={"model": "unknown"}))

    def test_reference_quote_must_exist(self):
        pack = {"code": None, "sources": [{"doc": "festo", "page": 1, "text": "A sufficiently long original manual quote."}]}
        qa = QA(question="매뉴얼 설명은?", reference="설명", intent="definition",
                sources=[{"doc": "festo", "page": 1, "quote": "A sufficiently long original manual quote."}])
        validate_qa(qa, pack)
        qa.sources[0].quote = "Fabricated quote not present on page"
        with self.assertRaises(ValueError):
            validate_qa(qa, pack)

    def test_missing_scores_are_reported_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "dataset.jsonl").write_text(json.dumps({"id": "q1", "question": "q", "category": "general"}) + "\n")
            (p / "metrics.jsonl").write_text(json.dumps({"id": "q1", "category": "general", "metric": "faithfulness", "value": None, "error": "timeout"}) + "\n")
            summary = build_report(p)
            value = summary["metrics"]["all"]["faithfulness"]
            self.assertIsNone(value["mean"])
            self.assertEqual(value["failed_or_undefined"], 1)


if __name__ == "__main__":
    unittest.main()
