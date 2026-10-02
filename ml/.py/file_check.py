import glob
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .py/ -> repo root

# 1. 파일 경로 설정 (유저 환경에 맞게 폴더 경로를 수정해 주세요)
folder_paths = [os.path.join(PROJECT_ROOT, "train"), os.path.join(PROJECT_ROOT, "test")]
all_files = []

for folder in folder_paths:
    files = glob.glob(os.path.join(folder, "*.csv"))
    all_files.extend([p.replace("\\", "/") for p in files])

summary_data = []
time_col = 'time'

for file in all_files:
    filename = os.path.basename(file)
    
    try:
        # 2. low_memory=False 추가로 DtypeWarning(혼합 타입 경고) 해결
        df = pd.read_csv(file, low_memory=False)
        
        # 3. 'Z'가 포함된 ISO 8601 날짜 형식을 안전하게 datetime으로 변환
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        
        actual_count = len(df)
        start_time = df[time_col].iloc[0]
        end_time = df[time_col].iloc[-1]
        
        duration = end_time - start_time
        duration_sec = duration.total_seconds()
        
        expected_count = int(duration_sec) + 1
        missing_count = expected_count - actual_count
        missing_rate = (missing_count / expected_count) * 100 if expected_count > 0 else 0
        
        summary_data.append({
            'File Name': filename,
            'Start Time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'End Time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Duration': str(duration),
            'Expected Count': expected_count,
            'Actual Count': actual_count,
            'Missing Count': missing_count,
            'Missing Rate (%)': round(missing_rate, 2)
        })
        
    except Exception as e:
        print(f"[{filename}] 처리 실패 - 원인: {e}")

# 4. 데이터프레임 변환 및 출력 (.py 스크립트 환경용)
summary_df = pd.DataFrame(summary_data)

if not summary_df.empty:
    summary_df = summary_df.sort_values(by='File Name').reset_index(drop=True)
    
    print("\n=== 📂 전체 파일 데이터 검증 요약 ===")
    print(f"총 검사한 파일 수: {len(summary_df)}개")
    print(f"전체 평균 결측률: {summary_df['Missing Rate (%)'].mean():.2f}%\n")
    print("-" * 80)
    
    print(summary_df.head(80).to_string())