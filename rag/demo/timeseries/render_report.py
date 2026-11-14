"""Create a portable report from an actual browser run."""

import argparse
import base64
import json
from html import escape
from pathlib import Path

SCENARIOS = {
    "A": ("첫 압력 저하 경보", "이력이 없는 경보에서 매뉴얼을 찾아 첫 해결 기록을 남긴다."),
    "B": ("압력 저하 재발", "A의 실제 조치와 PDF 근거를 확인하고 적용할 사례를 선택한다."),
    "C": ("비슷한 압력, 다른 위치 움직임", "A와 압력은 비슷하지만, 위치가 계속 이동해 별개의 조치를 기록한다."),
    "D": ("위치 유지 이상 재발", "A와 C가 모두 있을 때 현재 상황과 가까운 C의 해결 기록을 확인한다."),
}

STEPS = {
    "inbox": ("점검 시작", "ML이 19채널 경보를 API로 전송했다. 작업함에서 이 경보의 점검을 시작한다."),
    "assessment": ("이력 판정 확인", "케이스를 열면 UI가 판정을 요청한다. 검색 점수가 있으면 비교한 과거 사례를 펼쳐 볼 수 있다."),
    "candidates": ("원인 후보 보기", "RAG가 센서 관측을 바탕으로 PDF를 검색한다. 시연 작업자는 자동 후보를 검토하고 수동 검색을 이어갔다."),
    "manual": ("검색 → 이 사례 선택", "직접 매뉴얼 검색에 질문을 입력한다. 적용할 점검과 PDF 페이지를 확인한 뒤 사례를 선택한다."),
    "confirm": ("해결 확인 → 일지 승인", "작업자와 수행한 합성 조치를 입력한다. 해결 확인과 일지 승인을 체크하면 저장·종료 버튼을 누를 수 있다."),
    "saved": ("해결 결과 저장·종료", "API가 종료와 전송 대기를 기록했다. 화면의 공용 해결 이력 반영 완료는 Aura에 저장됐음을 뜻한다."),
    "selected": ("맞아요 · 이 사례 선택", "과거 조치와 PDF 근거를 검토하고 현재 점검에 적용할 사례로 선택했다."),
}

CSS = """
:root{font:16px/1.6 system-ui,sans-serif;color:#172334;background:#f2f5f9}
*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:32px 20px 56px}
header{background:#172b43;color:#fff;padding:28px 32px;border-radius:18px}
h1,h2,p{margin:0}h1{font-size:clamp(26px,4vw,44px);line-height:1.25;margin-bottom:12px}
header p{color:#dbe8f7;max-width:68ch}.tabs,.controls{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
button{font:inherit;cursor:pointer;color:#20344b;background:#fff;border:1px solid #c4cfdf;border-radius:8px;padding:8px 13px}
button[aria-pressed=true]{background:#214b7e;border-color:#214b7e;color:#fff}button:disabled{opacity:.5;cursor:default}
button:focus-visible,a:focus-visible{outline:3px solid #3873bd;outline-offset:3px}
section{background:#fff;border:1px solid #dbe3ee;border-radius:14px;padding:20px;margin-top:12px}
.intro{color:#40536c;margin-top:4px}.result{margin:12px 0;padding:9px 12px;background:#eaf1fb;border-radius:8px}
figure{margin:12px 0 0}img{display:block;width:100%;border:1px solid #d6dfea;border-radius:8px}
figcaption{margin:8px 0;color:#475c76}.click{display:inline-block;margin-bottom:6px;font-weight:700;color:#214b7e}
footer{margin:18px 0 0;color:#40536c;font-size:.9rem}[hidden]{display:none}
@media(max-width:640px){main{padding:14px 10px}header,section{padding:14px}}
"""

SCRIPT = """
const panels = [...document.querySelectorAll('[data-scenario]')];
const tabs = [...document.querySelectorAll('[data-tab]')];
const prev = document.querySelector('#prev'), next = document.querySelector('#next');
let scenario = 0, step = 0;
const frames = index => [...panels[index].querySelectorAll('[data-frame]')];
function show(nextScenario, nextStep) {
  scenario = nextScenario;
  step = nextStep;
  const current = frames(scenario);
  panels.forEach((panel, index) => panel.hidden = index !== scenario);
  tabs.forEach((tab, index) => tab.setAttribute('aria-pressed', String(index === scenario)));
  current.forEach((frame, index) => frame.hidden = index !== step);
  panels[scenario].querySelector('[data-step]').textContent = '화면 ' + (step + 1) + ' / ' + current.length;
  prev.disabled = scenario === 0 && step === 0;
  next.disabled = scenario === panels.length - 1 && step === current.length - 1;
}
tabs.forEach((tab, index) => tab.addEventListener('click', () => show(index, 0)));
next.addEventListener('click', () => step + 1 < frames(scenario).length ? show(scenario, step + 1) : show(scenario + 1, 0));
prev.addEventListener('click', () => step > 0 ? show(scenario, step - 1) : show(scenario - 1, frames(scenario - 1).length - 1));
show(0, 0);
"""


def render(source, destination):
    report = json.loads((source / "results.json").read_text(encoding="utf-8"))
    panels = []
    for result in report["results"]:
        letter = result["event_id"][-1]
        title, description = SCENARIOS[letter]
        decision = result["assessment"]
        info = [f"{result['case_id']} · {decision['status']}"]
        if decision["past_event_id"]:
            info.append(f"과거 경보 {decision['past_event_id']}")
        if result.get("recording_status") == "saved":
            info.append("Aura 반영 완료")
        frames = []
        for screen in report["screens"]:
            prefix, step = Path(screen["file"]).stem.split("-", maxsplit=1)
            if prefix != letter.lower():
                continue
            action, note = STEPS[step]
            if step == "candidates":
                status = result["manual_candidates"]["status"]
                if status == "generation_error":
                    note = "자동 후보 요청은 오류 상태를 표시했다. 작업자는 아래 수동 검색으로 계속 진행했다."
                elif status == "no_evidence":
                    note = "자동으로 제시할 근거가 부족하다는 안내가 나왔다. 작업자는 수동으로 매뉴얼을 검색했다."
            image = base64.b64encode((source / screen["file"]).read_bytes()).decode("ascii")
            frames.append(f"""<figure data-frame>
              <figcaption><span class="click">{escape(action)}</span><br>{escape(note)}</figcaption>
              <img alt="{escape(screen['label'])}" src="data:image/png;base64,{image}">
            </figure>""")
        panels.append(f"""<section data-scenario>
          <h2>{letter} · {escape(title)}</h2><p class="intro">{escape(description)}</p>
          <p class="result">{escape(' · '.join(info))}</p><p data-step aria-live="polite"></p>
          {''.join(frames)}
        </section>""")
    tabs = "".join(f'<button type="button" data-tab>{letter} · {escape(title)}</button>'
                   for letter, (title, _) in SCENARIOS.items())
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>정비 이력 RAG A~D 시연</title><style>{CSS}</style></head>
<body><main><header><h1>정비 이력이 다음 경보에 쓰이는 과정</h1>
<p>ML이 보낸 합성 경보 A~D를 실제 작업 화면에서 처리한 기록이다. 경보 탭이나 다음 화면 버튼을 눌러 작업 순서를 확인할 수 있다.</p></header>
<nav class="tabs" aria-label="경보 시나리오">{tabs}</nav>
<div class="controls"><button type="button" id="prev">이전 화면</button><button type="button" id="next">다음 화면</button></div>
{''.join(panels)}
<footer>React·API·ML 서버, Claude, 팀 공용 Aura를 연결해 화면을 캡처했다. 센서와 작업자의 조치는 합성 시나리오다.</footer>
</main><script>{SCRIPT}</script></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(args.source, args.destination)
