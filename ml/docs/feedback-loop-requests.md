# ML 피드백 루프(D-9): 다른 파트 요청

2026-11-08 · ML

1단계를 막는 요청은 없다. 루프(`ml/.py/feedback_loop.py`)는 지금 화면 그대로 돈다. 설계는 `ml/docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md`이고, 이 문서는 그 11절이다.

api는 담당이 따로 없어 ML이 직접 고쳤다(스펙 11.1, `api/README.md`의 "연결과 저장"). 요청이 아니라 알림이다.

- 조기 경보 핸드오프(`event_type` = `early_warning`)로 만든 케이스의 `severity`는 `warning`(화면의 "주의")이다. 나머지는 지금처럼 `critical`이다.
- 같은 설비라도 조기 경보와 모델 단독 critical은 서로 다른 케이스로 묶인다(묶음 키 `store.group_key`).
- 케이스 목록의 `event`에 `event_type`, `/api/history` 항목에 `repeat_event_ids`가 붙는다.

조기 경보 핸드오프는 ML에서 기본으로 꺼져 있다(`RSW_EARLY_WARNING_HANDOFFS`). 켜지 않으면 응답에 필드 둘이 늘 뿐이다.

## FE 요청

### 1. 조기 경보 케이스를 "조기 경보 · 확정 아님"으로 보여 주기

- 무엇: `event.event_type`이 `early_warning`인 케이스를 "조기 경보 · 확정 아님"으로 보여 준다.
- 왜: api 변경 뒤에는 `severity`가 "주의"로 보이지만, 문구는 여전히 "코드 없음 · 새 고장"이다(11-08 화면 점검). 조기 경보는 "몇십 분 안에 고장이 날 수 있음"을 알리는 정밀도 낮은 경고라, 확정된 고장처럼 보이면 안 된다.
- 어디: 케이스 목록과 상세의 문구. `ui/src/types/case.ts:35`의 `CaseEvent`에 `event_type`이 아직 없다. 주의 표시는 이미 `ui/src/components/Tag.tsx:38-42`에 있다.

### 2. 조기 경보 케이스 닫기 선택지

- 무엇: "이상 없음(오알람)" + 사유 칩, 그리고 "이상 찾아 조치함". 뒤쪽은 지금의 해결 종료로 저장해도 된다. 루프는 해결 종료를 "조치함"으로 읽는다.
- 왜: 루프는 조기 경보의 오알람 사유와 조치 여부를 라벨로 쓴다(스펙 6.3). 조치함으로 닫힌 경보는 오알람 셈에서 빠진다.
- 어디: 케이스 닫기 화면. 지금 해결 종료는 일지 승인을 요구한다(`api/main.py:358-360`, 409 "log not approved"). 이 흐름이 조기 경보에도 맞는지는 FE가 판단한다.

### 3. 오알람 사유 칩 4종을 확정하면 알려 주기

- 무엇: 팀이 사유 칩을 확정하면 ML에 알려 준다.
- 왜: 루프의 사유 표(스펙 7.7, `ml/.py/feedback_eval.py`의 `REASONS`)가 칩 문구와 글자까지 같아야 한다. 목록 밖의 글은 "기타"로, 사유가 없는 닫기는 "사유 없음"으로 센다.
- 어디: `ui/src/types/case.ts:325`의 `FALSE_ALARM_REASONS`.
