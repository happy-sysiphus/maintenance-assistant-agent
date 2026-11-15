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
            cases = get(f"{args.api}/api/cases")["cases"]
            case_id = next(c["case_id"] for c in cases if c["event"]["event_id"] == event_id)
            page.goto(args.ui + "/?tab=early")
            row = page.get_by_role("row").filter(has=page.locator(f'a[href="/cases/{case_id}"]'))
            expect(row).to_be_visible(timeout=30_000)
            screenshot(f"{prefix}-inbox", "ML 전송 후 작업함에 도착한 경보")
            row.get_by_role("link", name="확인하기").click()
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
                "unseen": "Unseen · 적용할 이전 해결 사례를 찾지 못했어요.",
                "seen": "Seen · 비교할 수 있는 이전 해결 사례를 찾았어요.",
                "uncertain": "판단 보류 · 매뉴얼에서 점검 근거를 확인해 주세요.",
            }[decision["status"]]
            expect(page.get_by_text(expected_message, exact=True)).to_be_visible()
            if decision.get("candidate_comparisons"):
                page.get_by_text("비교한 과거 사례 보기", exact=True).click()
            screenshot(f"{prefix}-assessment", "케이스를 열어 확인한 이력 판정과 비교 결과")
            result = {"event_id": event_id, "case_id": case_id, "ml_delivery": transfer["delivery"],
                      "assessment": decision}

            if "simulated_resolution" in item:
                with page.expect_response(lambda r: r.url.endswith("/warning/candidates") and r.request.method == "POST",
                                          timeout=120_000) as answer:
                    page.get_by_role("button", name="원인 후보 보기").click()
                candidates = answer.value.json()
                screenshot(f"{prefix}-candidates", "원인 후보 보기로 매뉴얼 검색")

                resolution = item["simulated_resolution"]
                code = resolution["manual_case_id"].split(":")[1].split("-")[0]
                page.get_by_role("link", name="직접 찾기", exact=False).click()
                page.get_by_label("질문", exact=True).fill(f"{code} 참고 항목. {item['manual_question']}")
                with page.expect_response(lambda r: r.url.endswith("/warning/manual") and r.request.method == "POST",
                                          timeout=120_000) as answer:
                    page.get_by_role("button", name="찾기", exact=True).click()
                search = answer.value.json()
                card = next(c for c in search["cards"] if c["manual_case_id"] == resolution["manual_case_id"])
                article = page.get_by_text(card["title"], exact=True).locator("..")
                screenshot(f"{prefix}-manual", "수동 검색으로 찾은 등록 사례와 PDF")
                article.get_by_role("button", name="이 사례로 점검").click()
                page.get_by_role("navigation", name="진행 단계").wait_for()
                # This is a synthetic maintenance scenario, explicitly confirmed by its operator.
                radios = page.get_by_role("radio", name="이상", exact=True)
                for index in range(radios.count()):
                    radios.nth(index).check()
                page.get_by_role("button", name="다음", exact=True).click()
                page.get_by_role("button", name="기타", exact=True).click()
                page.get_by_placeholder("실제로 한 작업을 적어 주세요").fill(resolution["actual_action"])
                page.get_by_placeholder("이름", exact=True).fill("시연 작업자")
                page.get_by_role("button", name="다음", exact=True).click()
                page.get_by_label("조치 후 정상으로 돌아온 것을 확인했어요").check()
                screenshot(f"{prefix}-confirm", "실제 조치를 입력하고 해결을 확인한 정비일지")
                page.get_by_role("button", name="일지 승인하고 해결 종료").click()
                expect(page.get_by_text("✓ 해결로 닫았어요", exact=True)).to_be_visible(timeout=45_000)
                wait_for(lambda: get(route + params)["recording_status"] == "saved", seconds=100)
                screenshot(f"{prefix}-saved", "해결 종료와 공용 Aura 반영 완료")
                result.update(manual_candidates=candidates, manual_search=search, recording_status=get(route + params)["recording_status"])
            else:
                assert decision["card"]["repair"]["event_id"] == expected["applicable_past_event_ids"][0]
                saved_item = next(e for e in dataset["events"] if e["handoff"]["event_id"] == decision["past_event_id"])
                assert decision["card"]["repair"]["actual_action"].endswith(saved_item["simulated_resolution"]["actual_action"])
                page.get_by_role("button", name="이 방법으로 점검").click()
                page.get_by_role("navigation", name="진행 단계").wait_for()
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
