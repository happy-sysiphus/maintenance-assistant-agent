import glob
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .py/ -> repo root

# 1. 파일 경로 설정
folder_paths = [os.path.join(PROJECT_ROOT, "train"), os.path.join(PROJECT_ROOT, "test")]
all_files = []

for folder in folder_paths:
    files = glob.glob(os.path.join(folder, "*.csv"))
    all_files.extend([p.replace("\\", "/") for p in files])
    
# 모든 고유한 에러 값을 저장할 집합(set) 생성
unique_errors = set()

print(f"총 {len(all_files)}개의 파일 검사를 시작합니다...\n")

for file in all_files:
    filename = os.path.basename(file)
    
    try:
        # 2. usecols=['error']를 통해 메모리 낭비 없이 해당 컬럼만 읽어옴
        df = pd.read_csv(file, usecols=['error'], low_memory=False)
        
        # 3. 결측치(NaN)를 제외한 고유값들을 추출하여 set에 업데이트 (자동 중복 제거)
        # NaN 값도 하나의 종류로 보고 싶다면 .dropna()를 제거하세요.
        errors_in_file = df['error'].dropna().unique()
        unique_errors.update(errors_in_file)
        
    except ValueError:
        # 파일에 'error' 컬럼이 아예 존재하지 않을 때 발생하는 에러 처리
        print(f"[{filename}] 'error' 컬럼이 존재하지 않습니다. 건너뜁니다.")
    except Exception as e:
        print(f"[{filename}] 처리 실패 - 원인: {e}")

# 4. 결과 출력
print("-" * 50)
print("=== 🔍 발견된 모든 error 값의 종류 ===")

# # 집합(set)을 리스트로 변환 후 정렬하여 보기 좋게 출력
# sorted_errors = sorted(list(unique_errors))

if unique_errors:
    for i, err in enumerate(unique_errors, 1):
        print(f"{i}. {err}")
else:
    print("기록된 에러 값이 하나도 없거나 모두 결측치(NaN)입니다.")