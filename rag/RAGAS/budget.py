"""Meter every OpenAI HTTP response; block requests before exceeding the budget.

Ledger contains token counts, never keys or request text. Unknown usage is charged
at its reserved upper bound. Prices are USD/1M tokens, checked 2026-10-28.
One runner process per results directory; threads share the reservation lock.
"""
import contextvars
import json
from datetime import datetime, timezone
from pathlib import Path
import threading

import httpx

PHASE = contextvars.ContextVar("phase", default="setup")
SAMPLE = contextvars.ContextVar("sample", default="")
PRICES = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path, limit=3.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.lock = threading.Lock()
        self.reserved = 0.0
        self.spent = sum(r["usd"] for r in self.rows())
        self.contexts = {}
        self.auth_failed = False

    def rows(self):
        return [json.loads(s) for s in self.path.read_text(encoding="utf-8").splitlines()] if self.path.exists() else []

    def reserve(self, request):
        data = json.loads(request.content)
        model = data["model"]
        if model not in PRICES or data.get("stream"):
            raise ValueError("Unpriced model or streaming request is not allowed")
        input_rate, output_rate = PRICES[model]
        max_output = data.get("max_output_tokens", data.get("max_completion_tokens", data.get("max_tokens", 4096)))
        # UTF-8 byte count is a conservative token upper bound, plus framing.
        reserve = ((len(request.content) + 2048) * input_rate + max_output * output_rate) / 1e6
        with self.lock:
            if self.auth_failed:
                raise RuntimeError("OpenAI authentication failed; update key before resuming")
            if self.spent + self.reserved + reserve > self.limit:
                raise BudgetExceeded("Evaluation budget would be exceeded")
            self.reserved += reserve
            if PHASE.get() == "rag" and request.url.path.endswith("/responses"):
                self.contexts[SAMPLE.get()] = json.loads(data["input"][-1]["content"])
        return data, reserve

    def settle(self, data, reserve, response=None):
        usage = None
        if response is not None:
            try:
                usage = response.json().get("usage")
            except (ValueError, AttributeError):
                pass
        if usage:
            inp = usage.get("input_tokens", usage.get("prompt_tokens", usage.get("total_tokens", 0)))
            out = usage.get("output_tokens", usage.get("completion_tokens", 0))
            rates = PRICES[data["model"]]
            usd = (inp * rates[0] + out * rates[1]) / 1e6
        else:
            inp = out = None
            usd = reserve  # conservative even if a network failure was not billed
        row = {"at": datetime.now(timezone.utc).isoformat(), "phase": PHASE.get(),
               "sample_id": SAMPLE.get(), "model": data["model"],
               "input_tokens": inp, "output_tokens": out, "usd": usd,
               "estimated_upper_bound": not bool(usage),
               "http_status": response.status_code if response is not None else None}
        with self.lock:
            if response is not None and response.status_code == 401:
                self.auth_failed = True
            self.reserved -= reserve
            self.spent += usd
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if response is not None and PHASE.get() in {
                "faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"
            }:
                try:
                    choices = response.json().get("choices", [])
                    if choices:
                        trace = {"sample_id": SAMPLE.get(), "metric": PHASE.get(),
                                 "message": choices[0]["message"]}
                        with (self.path.parent / "judge_traces.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                except (ValueError, KeyError):
                    pass


class MeteredTransport(httpx.BaseTransport):
    def __init__(self, budget, verify):
        self.budget = budget
        self.inner = httpx.HTTPTransport(verify=verify, retries=0)

    def handle_request(self, request):
        data, reserve = self.budget.reserve(request)
        response = None
        try:
            response = self.inner.handle_request(request)
            response.read()
            return response
        finally:
            self.budget.settle(data, reserve, response)

    def close(self):
        self.inner.close()


class AsyncMeteredTransport(httpx.AsyncBaseTransport):
    def __init__(self, budget, verify):
        self.budget = budget
        self.inner = httpx.AsyncHTTPTransport(verify=verify, retries=0)

    async def handle_async_request(self, request):
        data, reserve = self.budget.reserve(request)
        response = None
        try:
            response = await self.inner.handle_async_request(request)
            await response.aread()
            return response
        finally:
            self.budget.settle(data, reserve, response)

    async def aclose(self):
        await self.inner.aclose()
