"""Capture the real browser workflow against running ML, API and UI servers."""

import argparse
import json
import time
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright


def wait_for(check, *, seconds=100):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.3)
    raise TimeoutError("expected state did not arrive")


def capture(args):
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(Path(__file__).with_name("dataset.json").read_text(encoding="utf-8"))
    results, screens = [], []
    with httpx.Client(timeout=20) as http, sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.browser)
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        page.set_default_timeout(120_000)
        page_errors = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        def screenshot(name, label):
            page.screenshot(path=args.output / f"{name}.png", animations="disabled")
            screens.append({"file": f"{name}.png", "label": label})

        def get(url):
            response = http.get(url)
            response.raise_for_status()
            return response.json()

        assert get(f"{args.ml}/health")["mode"] == "synthetic_handoff_replay"
        for item in dataset["events"]:
            event_id = item["handoff"]["event_id"]
            prefix = event_id[-1].lower()
            response = http.post(f"{args.ml}/demo/handoffs/{event_id}")
            response.raise_for_status()

            def delivered():
                result = get(f"{args.ml}/handoffs/{event_id}")
                if result["delivery"] == "failed":
                    raise RuntimeError(result["delivery_detail"])
                return result if result["delivery"] == "delivered" else None

            transfer = wait_for(delivered)
            assert transfer["rag_response"]["result"] == "new"
            page.goto(args.ui)
            row = page.get_by_role("row").filter(has_text=item["title"])
            expect(row).to_be_visible(timeout=30_000)
            screenshot(f"{prefix}-inbox", "ML 전송 후 작업함에 도착한 경보")
            row.get_by_role("link", name="점검 시작").click()
            page.wait_for_url("**/cases/**")
            case_id = page.url.rsplit("/", 1)[1]
            route = f"{args.api}/api/cases/{case_id}/warning"
            params = f"?event_id={event_id}"

            def assessed():
                value = get(route + params)
                return value if value["status"] != "awaiting_assessment" else None

            decision = wait_for(assessed)
            expected = item["expected"]
            assert decision["status"] in expected["acceptable_statuses"], (event_id, decision["status"], decision.get("model_reason"))
            assert decision["past_event_id"] in (expected["applicable_past_event_ids"] or [None])
            expected_message = {
                "unseen": "현재 경보에 적용할 이전 해결 사례를 찾지 못했습니다.",
                "seen": "현재 경보와 비교할 수 있는 이전 해결 사례를 찾았습니다.",
                "uncertain": "이력 재사용 판단을 보류했습니다. 매뉴얼에서 점검 근거를 찾을 수 있습니다.",
            }[decision["status"]]
            expect(page.get_by_text(expected_message, exact=True)).to_be_visible()
            if decision.get("candidate_comparisons"):
                page.get_by_text("비교한 과거 사례와 검색 점수", exact=True).click()
            screenshot(f"{prefix}-assessment", "케이스를 열어 확인한 이력 판정과 비교 결과")
            result = {"event_id": event_id, "case_id": case_id, "ml_delivery": transfer["delivery"],
                      "assessment": decision}

            if "simulated_resolution" in item:
                with page.expect_response(lambda r: r.url.endswith("/warning/candidates") and r.request.method == "POST",
                                          timeout=120_000) as answer:
                    page.get_by_role("button", name="원인 후보 보기").click()
                candidates = answer.value.json()
                message = {
                    "ok": "매뉴얼 후보를 가져왔습니다.",
                    "no_evidence": "맞는 매뉴얼 근거가 부족합니다. 직접 검색해 주세요.",
                    "generation_error": "후보를 생성하지 못했습니다. 다시 시도하거나 직접 검색해 주세요.",
                }[candidates["status"]]
                expect(page.get_by_text(message, exact=True)).to_be_visible()
                screenshot(f"{prefix}-candidates", "원인 후보 보기로 매뉴얼 검색")

                resolution = item["simulated_resolution"]
                code = resolution["manual_case_id"].split(":")[1].split("-")[0]
                page.get_by_label("직접 매뉴얼 검색").fill(f"{code} 참고 항목. {item['manual_question']}")
                with page.expect_response(lambda r: r.url.endswith("/warning/manual") and r.request.method == "POST",
                                          timeout=120_000) as answer:
                    page.get_by_role("button", name="검색", exact=True).click()
                search = answer.value.json()
                card = next(c for c in search["cards"] if c["manual_case_id"] == resolution["manual_case_id"])
                source = card["source"]
                article = page.get_by_role("article").filter(has=page.get_by_role("heading", name=card["title"], exact=True))
                article.scroll_into_view_if_needed()
                expect(article.get_by_role("link", name=f"{source['doc']} PDF p.{source['page']}")).to_be_visible()
                screenshot(f"{prefix}-manual", "수동 검색으로 찾은 적용 가능한 검토 사례와 PDF")
                article.get_by_role("button", name="이 사례 선택").click()
                expect(article.get_by_role("button", name="선택됨")).to_be_visible()
                page.get_by_label("작업자", exact=True).fill("시연 작업자")
                page.get_by_label("실제 수행한 조치").fill(resolution["actual_action"])
                page.get_by_label("실제 조치 후 해결됐음을 확인했습니다.").check()
                page.get_by_label("위 정비일지를 확인하고 승인합니다.").check()
                screenshot(f"{prefix}-confirm", "적용 근거와 실제 조치를 입력하고 해결과 일지를 확인")
                page.get_by_role("button", name="해결 결과 저장·종료").click()
                expect(page.get_by_text("공용 해결 이력 반영 완료", exact=True)).to_be_visible(timeout=45_000)
                screenshot(f"{prefix}-saved", "해결 종료와 공용 Aura 반영 완료")
                result.update(manual_candidates=candidates, manual_search=search, recording_status=get(route + params)["recording_status"])
            else:
                assert decision["card"]["repair"]["event_id"] == expected["applicable_past_event_ids"][0]
                saved_item = next(e for e in dataset["events"] if e["handoff"]["event_id"] == decision["past_event_id"])
                assert decision["card"]["repair"]["actual_action"] == saved_item["simulated_resolution"]["actual_action"]
                page.get_by_role("button", name="맞아요 · 이 사례 선택").click()
                expect(page.get_by_role("button", name="선택됨", exact=True)).to_be_visible()
                screenshot(f"{prefix}-selected", "이전 해결 카드의 조치와 PDF를 확인한 뒤 선택")
                result["selected"] = get(route + params)["selection"]
                assert result["selected"]["manual_case_id"] == saved_item["simulated_resolution"]["manual_case_id"]

            results.append(result)
            print(event_id, decision["status"], decision["past_event_id"], result.get("recording_status"), flush=True)
        assert not page_errors, page_errors
        browser.close()
    (args.output / "results.json").write_text(json.dumps({"results": results, "screens": screens}, ensure_ascii=False, indent=2),
                                             encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml", default="http://127.0.0.1:19111")
    parser.add_argument("--api", default="http://127.0.0.1:19112")
    parser.add_argument("--ui", default="http://127.0.0.1:19113")
    parser.add_argument("--browser", default="msedge")
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
