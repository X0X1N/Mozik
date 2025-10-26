import os
import shutil
import glob
import re

SRC = os.path.join('detected_face', 'authorized_face.jpg')
DST_DIR = 'dataset'

if not os.path.exists(SRC):
    print(f"[ERROR] 소스 파일이 없습니다: {SRC}\n먼저 'create_authorized_face.py'로 얼굴을 캡처해 주세요.")
    raise SystemExit(1)

os.makedirs(DST_DIR, exist_ok=True)

# 기존 dataset 파일들에서 인덱스 최대값을 찾아 다음 인덱스를 결정
pattern = os.path.join(DST_DIR, 'user.*.*')
candidates = glob.glob(pattern)
max_idx = 0
for p in candidates:
    bn = os.path.basename(p)
    m = re.match(r'user\.(\d+)\.(\d+)\..*', bn)
    if m:
        try:
            idx = int(m.group(2))
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            continue

new_idx = max_idx + 1

# 안전하게 존재하지 않는 파일명 찾기
dst_name = f'user.0.{new_idx}.jpg'
dst_path = os.path.join(DST_DIR, dst_name)
i = new_idx
while os.path.exists(dst_path):
    i += 1
    dst_name = f'user.0.{i}.jpg'
    dst_path = os.path.join(DST_DIR, dst_name)

shutil.copy2(SRC, dst_path)
print(f"[OK] '{SRC}'을(를) '{dst_path}'로 복사했습니다. 이제 'deepface_realtime.py'를 실행하여 인식 테스트를 해보세요.")
