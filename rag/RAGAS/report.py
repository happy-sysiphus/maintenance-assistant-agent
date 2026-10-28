"""Offline, repeatable aggregation; failed/undefined scores remain visible."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
from budget import PRICES


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def factual_breakdown(out, scores):
    """Explain F1 from existing NLI judgments, without new LLM calls.

    Verify the reconstructed F1 matches official output before reporting ratios.
    These descriptive ratios are not additional official RAGAS metrics.
    """
    verdicts = defaultdict(list)
    for r in rows(out / "judge_traces.jsonl"):
        if r["metric"] != "factual_correctness":
            continue
        try:
            message = r["message"]
            content = message.get("content")
            if not content and message.get("tool_calls"):
                content = message["tool_calls"][0]["function"]["arguments"]
            payload = json.loads(content or "{}")
            items = payload.get("statements", [])
            if items and all(isinstance(x, dict) and "verdict" in x for x in items):
                verdicts[r["sample_id"]].append([bool(x["verdict"]) for x in items])
        except (ValueError, KeyError):
            continue
    official = {r["id"]: r["value"] for r in scores if r["metric"] == "factual_correctness"}
    result = []
    for sample_id, pairs in verdicts.items():
        if len(pairs) != 2 or official.get(sample_id) is None:
            continue
        response_claims, reference_claims = pairs
        tp = sum(response_claims)
        fp = len(response_claims) - tp
        fn = len(reference_claims) - sum(reference_claims)
        reconstructed = round(2 * tp / (2 * tp + fp + fn + 1e-8), 2)
        if abs(reconstructed - official[sample_id]) > .011:
            continue
        result.append({"id": sample_id, "response_claims": len(response_claims),
            "response_claims_supported_by_reference": tp,
            "reference_claims": len(reference_claims),
            "reference_claims_supported_by_response": sum(reference_claims),
            "response_reference_support_ratio": tp / len(response_claims),
            "reference_claim_coverage_ratio": sum(reference_claims) / len(reference_claims),
            "official_factual_f1": official[sample_id]})
    (out / "factual_breakdown.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def build_report(out):
    out = Path(out)
    dataset = rows(out / "dataset.jsonl")
    answers = rows(out / "responses.jsonl")
    # Resumable records: latest record for a sample/metric wins.
    attempts = rows(out / "metrics.jsonl")
    scores = list({(r["id"], r["metric"]): r for r in attempts}.values())
    breakdown = factual_breakdown(out, scores)
    costs = rows(out / "costs.jsonl")
    audits = rows(out / "reference_audit.jsonl")
    run_status = json.loads((out / "run_status.json").read_text(encoding="utf-8")) if (out / "run_status.json").exists() else {}
    phases = defaultdict(float)
    for c in costs:
        phases[c["phase"]] += c["usd"]
    total = sum(c["usd"] for c in costs)
    answer_usd = []
    for a in answers:
        result = a["result"]
        usage = result.get("usage", {})
        rate = PRICES.get(result.get("model"))
        if rate:
            answer_usd.append((usage.get("input_tokens", 0) * rate[0]
                               + usage.get("output_tokens", 0) * rate[1]
                               + usage.get("embedding_tokens", 0) * PRICES["text-embedding-3-small"][0]) / 1e6)
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]
    metric_summary = {}
    for group in ("all", "coded", "E012", "E016", "E028", "E029", "general"):
        def belongs(row):
            return group == "all" or row["category"] == group or (group == "coded" and row["category"] != "general")
        expected = sum(1 for r in dataset if belongs(r))
        metric_summary[group] = {}
        for metric in metric_names:
            records = [r for r in scores if r["metric"] == metric and belongs(r)]
            valid = [r["value"] for r in records if r["value"] is not None]
            metric_summary[group][metric] = {"mean": statistics.mean(valid) if valid else None,
                "std": statistics.stdev(valid) if len(valid) > 1 else None,
                "valid": len(valid), "failed_or_undefined": len(records) - len(valid),
                "pending": expected - len(records)}
    summary = {"dataset_count": len(dataset), "answer_count": len(answers),
        "run_status": run_status,
        "answer_statuses": dict(Counter(a["result"]["status"] for a in answers)),
        "metric_records": len(scores), "metrics": metric_summary,
        "metric_attempts": len(attempts), "failed_attempts": sum(r.get("value") is None for r in attempts),
        "factual_breakdown_samples": len(breakdown),
        "reference_claim_coverage_mean": statistics.mean(r["reference_claim_coverage_ratio"] for r in breakdown) if breakdown else None,
        "response_reference_support_mean": statistics.mean(r["response_reference_support_ratio"] for r in breakdown) if breakdown else None,
        "reference_audit": {"checked": len(audits), "supported": sum(a["supported"] for a in audits),
                            "flagged_ids": [a["id"] for a in audits if not a["supported"]]},
        "recorded_usd": total, "cost_by_phase": dict(phases),
        "usage_confirmed_usd": sum(c["usd"] for c in costs if not c["estimated_upper_bound"]),
        "final_answer_cost_usd": sum(answer_usd),
        "final_answer_mean_cost_usd": statistics.mean(answer_usd) if answer_usd else None,
        "unknown_usage_requests": sum(bool(c["estimated_upper_bound"]) for c in costs),
        "rag_seconds_mean": statistics.mean(a["result"]["seconds"] for a in answers) if answers else None,
        "rag_seconds_p95": sorted(a["result"]["seconds"] for a in answers)[max(0, __import__('math').ceil(len(answers)*.95)-1)] if answers else None}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["id", "category", "question", "status", "seconds"] + metric_names + ["reference", "response", "reference_sources", "reference_audit"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        lookup = {(r["id"], r["metric"]): r["value"] for r in scores}
        by_id = {a["id"]: a for a in answers}
        audit_by_id = {a["id"]: a["supported"] for a in audits}
        for r in dataset:
            a = by_id.get(r["id"], {})
            writer.writerow({"id": r["id"], "category": r["category"], "question": r["question"],
                             "status": a.get("result", {}).get("status", "pending"),
                             "seconds": a.get("result", {}).get("seconds", ""),
                             "reference": r.get("reference", ""), "response": a.get("response", ""),
                             "reference_sources": "; ".join(f"{s['doc']} p.{s['page']}" for s in r.get("sources", [])),
                             "reference_audit": audit_by_id.get(r["id"], "pending"),
                             **{m: lookup.get((r["id"], m)) for m in metric_names}})
    text = ["# RAGAS 평가 결과", "", "평가일: 2026-10-28 (Asia/Seoul)", "",
            f"- 실행 상태: **{run_status.get('status', 'partial')}**. {run_status.get('reason', '')}".rstrip(),
            f"- 데이터셋: {len(dataset)}/50문항, 실제 답변: {len(answers)}건, 채점 기록: {len(scores)}/250건.",
            f"- 답변 상태: {summary['answer_statuses']}",
            f"- 이번 평가 기록 비용: **${total:.5f}** (표준 단가 추정, 입력 캐시 할인 미적용).",
            f"- 실제 토큰 사용량이 확인된 호출 비용: **${summary['usage_confirmed_usd']:.5f}**. 나머지는 실패 요청의 보수적 예약 금액입니다.",
            f"- 토큰 사용량 미확인 요청: {summary['unknown_usage_requests']}건. 해당 요청은 예약 상한으로 보수적으로 합산.",
            f"- 합성 기준 답변 추가 자동 검토: {len(audits)}건 중 {sum(a['supported'] for a in audits)}건 지지 판정. 사람이 검증한 GT는 아닙니다.",
            f"- 채점 시도 총 {len(attempts)}건, 실패/미정의 시도 {summary['failed_attempts']}건. 일시 오류 재시도 시 이전 기록을 보존하고 최신 결과를 집계합니다.",
            f"- 재시도 입력의 수학 기호 표기 정규화(+/-·>=·<=) 적용: {sum(bool(r.get('ascii_math')) for r in attempts)}건. 평가자 유니코드 반복·출력 잘림에 대응한 동등 표기이며 원본 데이터는 보존했습니다. 재시도 출력 한도는 8192, 최초 한도는 4096입니다.",
            "", "## 지표별 결과", "", "평균은 유효 점수만 사용합니다. 실패·미정의 점수는 0으로 바꾸지 않으며 분모를 함께 보고합니다.", "",
            "coded는 E 코드 질문 32건, general은 코드 없는 일반 질문 18건입니다. 아래 표의 n은 계획된 문항 수이며 실제 채점 분모는 상세 표를 확인하세요.", "",
            "| 그룹 | n | Faithfulness | Relevancy | Context Precision | Context Recall | Factual F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",]
    for group, values in metric_summary.items():
        n = sum(group == "all" or r["category"] == group or (group == "coded" and r["category"] != "general") for r in dataset)
        text.append(f"| {group} | {n} | " + " | ".join(f"{values[m]['mean']:.4f}" if values[m]["mean"] is not None else "—" for m in metric_names) + " |")
    text += ["", "### 유효 건수와 표준편차", "",
            "| 그룹 | 지표 | 평균 | 표준편차 | 유효 | 실패/미정의 | 미실행 |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for group, values in metric_summary.items():
        for metric, v in values.items():
            mean = f"{v['mean']:.4f}" if v["mean"] is not None else "—"
            std = f"{v['std']:.4f}" if v["std"] is not None else "—"
            text.append(f"| {group} | {metric} | {mean} | {std} | {v['valid']} | {v['failed_or_undefined']} | {v['pending']} |")
    text += ["", "## 시간 및 비용", "", f"- RAG 응답 평균: {summary['rag_seconds_mean']}초.",
             f"- RAG 응답 p95: {summary['rag_seconds_p95']}초. 예비 2건·확대 3건 병렬 환경에서 측정했으며 단일 사용자 지연과 다를 수 있습니다.", "",
             f"- 최종 데이터셋 답변 생성 비용: ${summary['final_answer_cost_usd']:.6f}, 답변 1건 평균: {summary['final_answer_mean_cost_usd']} USD. 평가자 비용·제외한 초안 비용은 이 평균에서 제외했습니다.", "",
             "| 단계 | 비용(USD) |", "|---|---:|"]
    text += [f"| {phase} | ${value:.6f} |" for phase, value in sorted(phases.items())]
    text += ["", "## Factual Correctness F1 해석 보조", "",
        "아래 값은 저장된 평가자 NLI 판단에서 파생한 설명용 비율이며 추가 공식 RAGAS 지표가 아닙니다.",
        "공식 F1 점수 재계산이 일치하는 문항만 집계하고 새 LLM 호출은 하지 않았습니다.", "",
        f"- 분석 가능한 문항: {len(breakdown)}개.",
        f"- 기준 답변 주장 중 실제 RAG 답변이 지지한다고 평가된 비율의 평균: {summary['reference_claim_coverage_mean']}.",
        f"- RAG 답변 주장 중 짧은 기준 답변이 지지한다고 평가된 비율의 평균: {summary['response_reference_support_mean']}.",
        "- 앞 비율보다 뒤 비율이 낮다면 정답의 핵심은 포함하지만 기준에 없는 보충 설명이 많아 F1이 낮아지는 양상이 있을 수 있습니다. 보충 설명이 옳다는 보장은 아니며 Faithfulness·원문 대조와 함께 봐야 합니다.",
        "- 주장별 수와 분모는 factual_breakdown.json에 있습니다."]
    text += ["", "## 낮은 점수 사례", "", "정답 여부의 확정 판정이 아니라 우선 검토 대상입니다.", ""]
    dataset_by_id = {r["id"]: r for r in dataset}
    for metric in metric_names:
        low = sorted((r for r in scores if r["metric"] == metric and r["value"] is not None), key=lambda r: r["value"])[:3]
        text.append(f"### {metric}")
        text.append("")
        text += [f"- {r['id']}: {r['value']:.4f} — {dataset_by_id[r['id']]['question']}" for r in low]
        text.append("")
    text += ["## 기준 답변 자동 검토에서 표시된 문항", ""]
    for a in audits:
        if not a["supported"]:
            text.append(f"- {a['id']}: {'; '.join(a['issues'])}")
    text += ["", "표시된 문항도 삭제·수정하지 않고 전체 점수에 포함했습니다. 해당 reference를 먼저 검토해야 합니다.", "", "## 평가 해석과 한계", "",
        "- 기준 답변은 PDF 원문에서 별도로 생성한 합성 reference입니다. 사람이 검증한 gold GT가 아닙니다.",
        "- 생성기는 층화 비율을 맞추는 자체 PDF 기반 생성기이며 RAGAS TestsetGenerator를 실행한 결과는 아닙니다. 채점은 설치한 공식 ragas 라이브러리의 지표를 그대로 사용합니다.",
        "- 최종 질문·기준 답변 생성기는 gpt-5.4-mini, 평가자는 gpt-4.1-mini입니다. 생성기와 RAG가 같은 모델 계열이며 자동 평가의 편향이 가능하고 독립 전문가 평가가 아닙니다.",
        "- 사전 품질검사에서 다른 코드 질문·불완전한 기준 답변·시간 임계값 오독이 발견된 초기 초안은 최종 평가 전에 제외했습니다. 생성기를 강화하고 오류별 표 행만 추출했으며, 제외된 초안의 API 비용도 모두 포함했습니다.",
        "- E 코드 32건, 일반 질문 18건이며 history·사진·센서·미지원 질문·답변 보류 전용 세트는 평가하지 않습니다.",
        "- 같은 오류 코드를 반복하는 질문들은 상관되어 있습니다. 50문항을 완전히 독립인 50개 문제로 해석하지 않습니다.",
        "- 기준 답변 원문 인용이 해당 PDF 페이지에 존재하는지는 자동 검사했습니다. 이것만으로 기준 답변 전체의 정확성을 보증하지 않습니다.",
        "- Faithfulness에는 실제 생성 입력의 검색 원문과 검토 매핑·적용 범위를 포함합니다. Context Precision/Recall에는 순서가 보존된 검색 원문만 사용합니다.",
        "- Context Precision은 reference 기준, Context Recall은 reference의 사실 회수율입니다. Factual Correctness는 짧은 reference와의 F1으로, 정당한 보충 설명도 감점될 수 있습니다.",
        "- 공식 기본 영문 평가 프롬프트를 유지했습니다. Answer Relevancy의 역생성 질문은 영어로 출력되는 경우가 있어 한국어 질문과의 다국어 임베딩 유사도 영향을 받습니다. 이 점수를 단독으로 한국어 답변 품질의 절대 척도로 해석하지 않습니다.",
        "- 사용자에게 전달되는 전체 answer(점검·기준·조치 포함)를 평가합니다. 일부만 골라 점수를 높이지 않습니다.",
        "- 캐시는 사용하지 않았고, 기존 서비스의 실패는 그대로 보존합니다. 점수에 맞춰 프롬프트나 KG를 수정하지 않았습니다.",
        "- 숫자는 LLM 기반 자동 평가의 추정치이며 0.9 같은 보편적 합격선은 적용하지 않았습니다.",
        "- 지표별 점수는 metrics.jsonl, 평가자 판단 원문은 judge_traces.jsonl, 질문·합성 기준은 dataset.jsonl, RAG 원본 응답은 responses.jsonl, 비용은 costs.jsonl에 있습니다.",
        "", "## 공식 근거", "",
        "- [Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/)",
        "- [Response Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)",
        "- [Context Precision](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/)",
        "- [Context Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)",
        "- [Factual Correctness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/)",
        "- [Testset Generation](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/)",
        "- [평가자 단가](https://developers.openai.com/api/docs/models/gpt-4.1-mini)", ""]
    (out / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import sys
    build_report(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results")
