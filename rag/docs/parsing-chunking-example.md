# 매뉴얼 파싱·청킹 개선 전후: 실제 청크 사례

진단 문서의 **오류 17(Friction Low)** 이 PDF 1쪽에서 2쪽으로 이어지는 사례다. 파싱 구조를 설명하기 위한 사례이며, 서비스 지원 오류 코드 E012·E016·E028·E029에 새 코드를 추가한 것은 아니다. 쪽 번호는 PDF 파일에서 1부터 센다.

아래 JSON은 실제 저장된 chunks.jsonl 레코드에서 **설명에 필요한 필드만 발췌**했다. 표시한 필드명과 값, 본문은 원본 그대로이며 일부 메타데이터와 개인 PC의 절대 경로는 생략했다. 개선 전의 두 레코드는 비교하기 쉽게 JSON 배열로 묶었다.

- PDF: diagnostic4, festo_bmw_2gen_error_messages_en.pdf, 1~2쪽.
- 개선 전: 기존 파서·청커로 생성한 backup의 rag/var/chunks.jsonl.
- 개선 후: 개선된 파서·청커로 생성한 현재 rag/var/chunks.jsonl.
- 두 JSONL은 Git에서 제외되는 로컬 산출물이다. 아래 발췌본은 산출물이 없어도 비교할 수 있도록 남겼다.

## 1. 처리 흐름의 차이

| 단계 | 개선 전 | 개선 후 |
|---|---|---|
| 표 추출 | 주로 pdfplumber의 표·셀 추출과 매뉴얼별 규칙 사용 | Docling 레이아웃·표 구조 인식 후 겹치는 pdfplumber 표가 있으면 셀 경계와 텍스트 결합 |
| 머리글 | 일부 알려진 표는 수동 대응. 이 사례처럼 식별하지 못하면 Column 1 등 사용 | Docling 머리글과 원본 셀을 결합해 Nr, Error, Description 등으로 표현. 불확실한 경우 검토 표시 |
| 페이지 경계 | 페이지별 처리로 이어지는 행이 별도 청크에 들어갈 수 있음 | 위치·절·머리글·열 수 등 조건이 맞는 연속 페이지의 행을 연결 |
| 청킹 | 행 단위를 사용하지만 일반 표 행은 여러 행이 같은 청크에 모일 수 있음 | 식별된 오류 행을 error_row로 취급하고 표·행 ID와 페이지별 출처를 보존 |
| 크기 | 목표 상한 1200자 | 목표 상한 1200자 유지. 긴 원자 단위는 중간에서 자르지 않을 수 있음 |

Docling과 pdfplumber의 모든 셀을 정답처럼 비교하는 것은 아니다. 같은 위치의 표를 좌표 겹침으로 연결하고 결과를 규칙에 따라 결합한다. 대응하는 표 경계를 찾지 못하면 Docling 결과를 사용한다.

## 2. 개선 전 실제 결과

오류 17의 첫 부분은 1쪽의 오류 16과 **같은 청크**에 있다. 이어지는 설명은 2쪽의 **별도 청크**로 저장됐으며 그 본문에는 오류 번호 17이 없다.

~~~json
[
  {
    "chunk_id": "chunk-794f438ac97d51c997f3",
    "chunk_index": 818,
    "page_content": "7.1 Error Numbers Servopneumatic Welding Gun System CPX-WELD-2G-SA\n[Table; PDF page 1]\nColumn 1: 16 | Column 2: Electrode Broken | Column 3: The electrode position is smaller than the zero position from the reference travel + the electrode monitoring limit value written in WinSPAS during the build-up of force for at least 12ms. The error is not monitored during path calibration and homing. | Column 4: X\nColumn 1: 17 | Column 2: Friction Low | Column 3: The friction measured during reference travel is | Column 4: X",
    "metadata": {
      "manual_id": "diagnostic4",
      "page_number": 1,
      "unit_kind": "table_row",
      "unit_count": 2,
      "error_numbers": [],
      "source_page": 1,
      "char_count": 519
    }
  },
  {
    "chunk_id": "chunk-0d1fb5100ce1cd733f8b",
    "chunk_index": 819,
    "page_content": "Section 7.1\n[Table; PDF page 2]\nColumn 3: smaller than the limit parameterized in WinSPAS. After the path calibration or after resetting the detection in WinSPAS, a friction mean value is formed during the first 10 homing runs and compared against this from the 11th homing run. If the friction is at least 3 times lower than the stored value, it will generate an error.",
    "metadata": {
      "manual_id": "diagnostic4",
      "page_number": 2,
      "unit_kind": "prose",
      "unit_count": 1,
      "error_numbers": [],
      "source_page": 2,
      "char_count": 370
    }
  }
]
~~~

1쪽은 Column 1 등의 일반 머리글을 사용한다. 2쪽에 있는 “저장된 값보다 3배 이상 낮으면 오류 발생”이라는 설명은 첫 청크만 검색하면 함께 전달되지 않을 수 있다. 반대로 2쪽 청크만 가져오면 해당 문장이 오류 17에 속한다는 정보가 본문에 직접 들어 있지 않다.

unit_kind가 prose인 것은 이전 청커의 분류 결과다. PDF에서 이 내용이 표 밖의 일반 본문이었다는 뜻은 아니다.

## 3. 개선 후 실제 결과

오류 17을 **하나의 오류 행 청크**로 만들었다. page_content에는 두 쪽의 설명을 연결하고 source_parts에는 각 부분이 원래 나온 쪽과 표 위치를 따로 기록했다.

~~~json
{
  "chunk_id": "chunk-85bfd4a9801b907cc212",
  "chunk_index": 862,
  "page_content": "7.1 Error Numbers Servopneumatic Welding Gun System CPX-WELD-2G-SA\n[Table diagnostic4:tables:0: 7.1 Error Numbers Servopneumatic Welding Gun System CPX-WELD-2G-SA]\nNr: 17 | Error: Friction Low | Description: The friction measured during reference travel is smaller than the limit parameterized in WinSPAS. After the path calibration or after resetting the detection in WinSPAS, a friction mean value is formed during the first 10 homing runs and compared against this from the 11th homing run. If the friction is at least 3 times lower than the stored value, it will generate an error. | Monitoring / CPX: X",
  "metadata": {
    "manual_id": "diagnostic4",
    "page_number": 1,
    "unit_kind": "error_row",
    "unit_count": 1,
    "error_numbers": [
      "17"
    ],
    "source_page": 1,
    "source_pages": [
      1,
      2
    ],
    "char_count": 607,
    "max_chars": 1200,
    "unit_sources": [
      {
        "kind": "error_row",
        "table_id": "diagnostic4:tables:0",
        "row_id": "diagnostic4:tables:0:row:17",
        "source_parts": [
          {
            "doc": "diagnostic4",
            "page": 1,
            "bbox": [
              65.39414634146341,
              124.51999999999998,
              538.8890813636364,
              782.5559999999999
            ],
            "text": "Nr: 17 | Error: Friction Low | Description: The friction measured during reference travel is | Monitoring / CPX: X"
          },
          {
            "doc": "diagnostic4",
            "page": 2,
            "bbox": [
              65.39172413793104,
              71.03999999999996,
              538.8922483870966,
              779.4359999999999
            ],
            "text": "Description: smaller than the limit parameterized in WinSPAS. After the path calibration or after resetting the detection in WinSPAS, a friction mean value is formed during the first 10 homing runs and compared against this from the 11th homing run. If the friction is at least 3 times lower than the stored value, it will generate an error."
          }
        ]
      }
    ]
  }
}
~~~

- unit_kind: error_row는 식별된 오류 표 행을 뜻한다.
- error_numbers: ["17"]은 추출된 표 행의 오류 번호다.
- source_pages: [1, 2]는 청크의 내용이 두 PDF 페이지에서 왔음을 나타낸다.
- table_id와 row_id는 표와 행을 추적하는 식별자다.
- source_parts는 페이지별 원문 부분과 해당 표의 위치다. bbox는 **표 영역 좌표**이며 각 글자나 셀 하나의 좌표가 아니다.
- max_chars: 1200은 청킹의 목표 상한이다. 모든 청크가 정확히 1200자라는 뜻은 아니다.

## 4. 연결 절차

1. Docling이 PDF 페이지를 제목·본문·표 등으로 나누고 표의 머리글과 행·열 구조를 인식한다. 현재 설정에서는 OCR을 사용하지 않는다.
2. 같은 페이지에서 pdfplumber가 찾은 표와 Docling 표의 **위치가 충분히 겹치면** 같은 표 후보로 선택한다. 그리드와 머리글을 결합하고 빈 열·하위 머리글 등을 조건에 따라 보정한다.
3. 앞 페이지의 마지막 표와 다음 페이지의 첫 표에 대해 **페이지 연속성, 페이지 가장자리와의 거리, 좌우 위치, 절, 머리글과 열 수**를 확인한다.
4. 연결 조건이 맞고 다음 페이지 첫 행의 앞 셀이 비어 있으면 앞 행의 이어지는 내용으로 합친다. 이 사례에서 오류 17의 나머지 설명이 연결됐다.
5. 청커가 연결된 행을 하나의 error_row로 저장하면서 원래의 페이지별 source_parts를 남긴다. 검색은 합쳐진 내용을 활용하고, 인용 근거는 페이지별로 제공한다.

위 조건은 프로젝트 문서 구조를 고려한 **휴리스틱**이다. 모든 표 연결을 정확하게 판단하거나 모든 형태의 이어지는 표를 연결한다고 보장하지 않는다.

## 5. 해석상의 주의점

- 한 가지 실제 사례이며 모든 표의 전사나 연결이 정확하다는 증거는 아니다.
- 그림은 존재·위치만 기록한다. 회전된 비트 열과 일부 불확실한 표·수식은 원본 확인 대상으로 남긴다.
- 검색에 쓰는 완성된 page_content와 페이지별 source_parts는 목적이 다르다. 전자는 검색 문맥을 이어 주고 후자는 원래 출처를 보존한다.
- 전체 198쪽 중 대표 20쪽의 핵심 항목을 원본과 대조했고, 별도 확인이 필요한 표 블록 18개가 남아 있다.
- 청크 크기 비교와 평가 범위는 [개선 전후 비교 보고서](../RAGAS/results-20261104/COMPARISON.md), 구현은 [파싱 코드](../pipeline/parse.py)와 [청킹 코드](../pipeline/chunk.py), 검토 대상은 [표 검토 목록](../RAGAS/results-20261104/parsing-audit.json)에서 확인할 수 있다.
