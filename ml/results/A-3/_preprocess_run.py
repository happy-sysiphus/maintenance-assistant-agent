"""preprocess.py를 그대로 실행하되 --gap-fill-limit 0(채우지 않음)도 되게 한다.
pandas의 ffill(limit=0)은 오류가 나서, 0일 때만 '빈 행은 버리고 채우지 않음'으로 바꾼다. 0보다 크면 원래 함수 그대로."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import preprocess  # noqa: E402

_original = preprocess.fill_gaps


def fill_gaps(x, present, limit):
    if limit > 0:
        return _original(x, present, limit)
    x = x[np.asarray(present, dtype=bool)]  # 빈 초는 버리고 끊는다 (세그먼트가 나뉨)
    return x[x.notna().all(axis=1)]


preprocess.fill_gaps = fill_gaps
if __name__ == "__main__":
    preprocess.main()
