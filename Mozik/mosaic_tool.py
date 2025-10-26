import cv2
import csv
import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple

# -----------------------------
# 데이터 구조와 유틸
# -----------------------------

@dataclass
class Box:
    frame: int
    t_ms: int   # timestamp in milliseconds
    id: int     # 간단 추적 ID
    x: int
    y: int
    w: int
    h: int
    source: str = "auto"  # "auto" 또는 "manual"

def ensure_dir(path: str):
    # 파일 경로일 수도 있으니 상위 폴더를 생성
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def expand_box(x, y, w, h, pad_x: float, pad_y: float, W: int, H: int):
    # 얼굴 박스를 좌우/상하로 확장
    x_new = int(round(x - w * pad_x))
    y_new = int(round(y - h * pad_y))
    w_new = int(round(w * (1 + 2*pad_x)))
    h_new = int(round(h * (1 + 2*pad_y)))
    x_new = clamp(x_new, 0, W-1)
    y_new = clamp(y_new, 0, H-1)
    w_new = clamp(w_new, 1, W - x_new)
    h_new = clamp(h_new, 1, H - y_new)
    return x_new, y_new, w_new, h_new

def pixelate_region(img, x, y, w, h, strength: int):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    s = max(1, strength)
    small_w = max(1, w // s)
    small_h = max(1, h // s)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = mosaic

def gaussian_blur_region(img, x, y, w, h, strength: int):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    k = max(1, strength)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y:y+h, x:x+w] = blurred

# 간단한 중심 거리 기반 ID 할당(프레임 간 근접한 박스를 같은 ID로 가정)
def assign_ids(prev_boxes: List[Box], curr_rects: List[Tuple[int,int,int,int]], frame_idx: int, t_ms: int) -> List[Box]:
    assigned: List[Box] = []
    used_prev = set()
    next_id_base = (prev_boxes[-1].id + 1) if prev_boxes else 0

    def center(x,y,w,h): return (x + w/2.0, y + h/2.0)

    for (x,y,w,h) in curr_rects:
        cx, cy = center(x,y,w,h)
        best = None
        best_d = 1e9
        for i, pb in enumerate(prev_boxes):
            if i in used_prev:
                continue
            pcx, pcy = center(pb.x, pb.y, pb.w, pb.h)
            d = math.hypot(cx - pcx, cy - pcy)
            if d < best_d:
                best_d = d
                best = (i, pb)
        th = max(w, h) * 1.5
        if best is not None and best_d < th:
            i, pb = best
            used_prev.add(i)
            assigned.append(Box(frame_idx, t_ms, pb.id, x,y,w,h))
        else:
            assigned.append(Box(frame_idx, t_ms, next_id_base, x,y,w,h))
            next_id_base += 1
    return assigned

# Haar 모델을 여러 후보 경로에서 안전하게 탐색
def load_haar_face():
    candidates = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
        os.path.join(os.path.dirname(cv2.__file__), "cv2", "data", "haarcascade_frontalface_default.xml"),
    ]
    tried = []
    for p in candidates:
        if p and os.path.exists(p):
            face_cascade = cv2.CascadeClassifier(p)
            if not face_cascade.empty():
                return face_cascade
            tried.append(p + " (empty)")
        else:
            tried.append(p + " (missing)")
    raise RuntimeError("Haar cascade 로드 실패. 확인 경로: " + " | ".join(tried))

def detect_faces(frame_gray, face_cascade, scale_factor: float, min_neighbors: int, min_size: int):
    rects = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(min_size, min_size)
    )
    return rects

def write_csv(csv_path: str, boxes: List[Box]):
    ensure_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame","t_ms","id","x","y","w","h","source"])
        for b in boxes:
            w.writerow([b.frame, b.t_ms, b.id, b.x, b.y, b.w, b.h, b.source])

def read_csv(csv_path: str) -> List[Box]:
    out: List[Box] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(Box(
                frame=int(row["frame"]),
                t_ms=int(row["t_ms"]),
                id=int(row["id"]),
                x=int(row["x"]),
                y=int(row["y"]),
                w=int(row["w"]),
                h=int(row["h"]),
                source=row.get("source","manual")
            ))
    return out

# -----------------------------
# 1) 스캔(탐지 → CSV)
# -----------------------------
def cmd_scan(args):
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 동영상 없음: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없음: {args.input}")
        sys.exit(1)

    try:
        face_cascade = load_haar_face()
    except Exception as e:
        print(f"[ERROR] 얼굴 모델 로드 실패: {e}")
        sys.exit(1)

    boxes_all: List[Box] = []
    prev_boxes: List[Box] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detect_faces(
            gray,
            face_cascade,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=args.min_face
        )

        curr_boxes = assign_ids(prev_boxes, rects, frame_idx, t_ms)
        for b in curr_boxes:
            b.source = "auto"
        boxes_all.extend(curr_boxes)
        prev_boxes = curr_boxes
        frame_idx += 1

        if args.preview and frame_idx % args.preview_stride == 0:
            preview = frame.copy()
            for (x,y,w,h) in rects:
                x2,y2,w2,h2 = expand_box(x,y,w,h, args.pad_x, args.pad_y, preview.shape[1], preview.shape[0])
                cv2.rectangle(preview, (x2,y2), (x2+w2, y2+h2), (0,255,0), 2)
            cv2.imshow("scan preview (q로 미리보기 종료)", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                args.preview = False
                cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()
    write_csv(args.output_csv, boxes_all)
    print(f"[OK] 탐지 결과 CSV 저장: {args.output_csv}")
    print("CSV에서 좌표(x,y,w,h)를 수정/삭제하여 섬세 조정 가능.")

# -----------------------------
# 2) 렌더(CSV 기반 또는 즉시 탐지 → 모자이크)
# -----------------------------
def cmd_render(args):
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 동영상 없음: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없음: {args.input}")
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(args.output)
    out = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
    if not out.isOpened():
        print(f"[ERROR] 출력 비디오 생성 실패: {args.output}")
        sys.exit(1)

    boxes_by_frame: dict[int, List[Box]] = {}
    face_cascade = None

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"[ERROR] CSV 없음: {args.csv}")
            sys.exit(1)
        for b in read_csv(args.csv):
            boxes_by_frame.setdefault(b.frame, []).append(b)
        print(f"[OK] CSV에서 {sum(len(v) for v in boxes_by_frame.values())}개 박스 로드.")
    else:
        print("[INFO] CSV 없이 즉시 자동 탐지로 렌더.")
        try:
            face_cascade = load_haar_face()
        except Exception as e:
            print(f"[ERROR] 얼굴 모델 로드 실패: {e}")
            sys.exit(1)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if boxes_by_frame:
            curr = boxes_by_frame.get(frame_idx, [])
            rects = [(b.x, b.y, b.w, b.h) for b in curr]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detect_faces(
                gray,
                face_cascade,
                scale_factor=args.scale_factor,
                min_neighbors=args.min_neighbors,
                min_size=args.min_face
            )

        # ★ 모든 얼굴에 대해 모자이크 처리
        for (x,y,w,h) in rects:
            x2,y2,w2,h2 = expand_box(x,y,w,h, args.pad_x, args.pad_y, frame.shape[1], frame.shape[0])
            if args.mode == "pixelate":
                pixelate_region(frame, x2,y2,w2,h2, strength=args.strength)
            else:
                gaussian_blur_region(frame, x2,y2,w2,h2, strength=args.strength)

            if args.draw_box:
                cv2.rectangle(frame, (x2,y2), (x2+w2, y2+h2), (0,255,0), 2)

        out.write(frame)
        frame_idx += 1

        if args.preview and frame_idx % args.preview_stride == 0:
            cv2.imshow("render preview (q로 미리보기 종료)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                args.preview = False
                cv2.destroyAllWindows()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[OK] 모자이크 비디오 저장: {args.output}")

# -----------------------------
# 3) 실시간 프리뷰(키보드/슬라이더) - 기존 방식
# -----------------------------
def cmd_preview(args):
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 동영상 없음: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없음: {args.input}")
        sys.exit(1)

    try:
        face_cascade = load_haar_face()
    except Exception as e:
        print(f"[ERROR] 얼굴 모델 로드 실패: {e}")
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    win = "Face Mosaic Preview (q:종료, m:모드, b:박스, r:녹화)"
    cv2.namedWindow(win)
    cv2.createTrackbar("Strength", win, 18, 60, lambda v: None)
    cv2.createTrackbar("PadX%", win, int(args.pad_x*100), 50, lambda v: None)
    cv2.createTrackbar("PadY%", win, int(args.pad_y*100), 50, lambda v: None)
    cv2.createTrackbar("MinFace", win, args.min_face, 200, lambda v: None)
    cv2.createTrackbar("Neighbors", win, args.min_neighbors, 15, lambda v: None)

    mode_pixelate = True
    draw_box = False
    recording = False
    writer = None

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            strength = max(1, cv2.getTrackbarPos("Strength", win))
            pad_x = cv2.getTrackbarPos("PadX%", win) / 100.0
            pad_y = cv2.getTrackbarPos("PadY%", win) / 100.0
            min_face = max(1, cv2.getTrackbarPos("MinFace", win))
            neighbors = max(0, cv2.getTrackbarPos("Neighbors", win))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detect_faces(gray, load_haar_face(), args.scale_factor, neighbors, min_face)

            vis = frame.copy()
            for (x,y,w,h) in rects:
                x2,y2,w2,h2 = expand_box(x,y,w,h, pad_x, pad_y, vis.shape[1], vis.shape[0])
                if mode_pixelate:
                    pixelate_region(vis, x2,y2,w2,h2, strength=strength)
                else:
                    gaussian_blur_region(vis, x2,y2,w2,h2, strength=strength)
                if draw_box:
                    cv2.rectangle(vis, (x2,y2), (x2+w2, y2+h2), (0,255,0), 2)

            if recording:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_path = args.output if args.output else "preview_record.mp4"
                    ensure_dir(out_path)
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
                    if not writer.isOpened():
                        print("[ERROR] 녹화 파일 생성 실패.")
                        recording = False
                        writer = None
                if writer is not None:
                    writer.write(vis)

            overlay = vis.copy()
            txt = f"Mode: {'PIXELATE' if mode_pixelate else 'GAUSSIAN'}  |  Box:{'ON' if draw_box else 'OFF'}  |  Rec:{'ON' if recording else 'OFF'}"
            cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 220, 10), 2, cv2.LINE_AA)
            cv2.imshow(win, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mode_pixelate = not mode_pixelate
            elif key == ord('b'):
                draw_box = not draw_box
            elif key == ord('r'):
                recording = not recording
                if not recording and writer is not None:
                    writer.release()
                    writer = None
                    print("[OK] 녹화 저장 완료.")
            frame_idx += 1
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

# -----------------------------
# 4) GUI 모드 (버튼: 재생/일시정지/저장/종료)
# -----------------------------
def cmd_gui(args):
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 동영상 없음: {args.input}")
        sys.exit(1)

    try:
        import PySimpleGUI as sg
    except ImportError:
        print("[ERROR] PySimpleGUI가 필요함. 먼저 설치:  pip install PySimpleGUI")
        sys.exit(1)

    # 초기 파라미터
    mode_pixelate = True
    draw_box = False
    is_playing = False
    is_saving = False

    # 레이아웃
    control_col = [
        [sg.Text("모자이크 모드")],
        [sg.Button("모드 전환 (픽셀↔블러)", key="-MODE-")],
        [sg.Checkbox("박스 표시", default=False, key="-BOX-")],
        [sg.Text("강도"), sg.Slider(range=(1,60), default_value=18, orientation="h", size=(28,15), key="-STRENGTH-")],
        [sg.Text("PadX%"), sg.Slider(range=(0,50), default_value=int(args.pad_x*100), orientation="h", size=(28,15), key="-PADX-")],
        [sg.Text("PadY%"), sg.Slider(range=(0,50), default_value=int(args.pad_y*100), orientation="h", size=(28,15), key="-PADY-")],
        [sg.Text("MinFace"), sg.Slider(range=(1,200), default_value=args.min_face, orientation="h", size=(28,15), key="-MINFACE-")],
        [sg.Text("Neighbors"), sg.Slider(range=(0,15), default_value=args.min_neighbors, orientation="h", size=(28,15), key="-NEIGH-")],
        [sg.HorizontalSeparator()],
        [sg.Button("재생/일시정지", key="-PLAY-"), sg.Button("저장 시작/중지", key="-SAVE-")],
        [sg.Button("처음으로", key="-REWIND-"), sg.Button("종료", key="-EXIT-")],
        [sg.Text("상태: 정지", key="-STATUS-", size=(34,1))]
    ]
    layout = [
        [sg.Image(key="-IMAGE-"), sg.Column(control_col, vertical_alignment='top')]
    ]
    window = sg.Window("Face Mosaic GUI", layout, finalize=True)

    # 비디오 준비
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sg.popup_error(f"영상을 열 수 없음: {args.input}")
        window.close()
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        face_cascade = load_haar_face()
    except Exception as e:
        sg.popup_error(f"얼굴 모델 로드 실패: {e}")
        window.close()
        sys.exit(1)

    # 저장 준비
    writer = None
    out_path = args.output if args.output else "gui_output.mp4"

    # 메인 루프
    try:
        while True:
            event, values = window.read(timeout=1)  # 1ms 폴링
            if event in (sg.WIN_CLOSED, "-EXIT-"):
                break

            if event == "-MODE-":
                mode_pixelate = not mode_pixelate
            if event == "-PLAY-":
                is_playing = not is_playing
                window["-STATUS-"].update(f"상태: {'재생' if is_playing else '정지'}")
            if event == "-SAVE-":
                is_saving = not is_saving
                if is_saving and writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    ensure_dir(out_path)
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
                    if not writer.isOpened():
                        sg.popup_error("저장 파일 생성 실패.")
                        is_saving = False
                        writer = None
                if not is_saving and writer is not None:
                    writer.release()
                    writer = None
                window["-STATUS-"].update(f"상태: {'저장중' if is_saving else ('재생' if is_playing else '정지')}")
            if event == "-REWIND-":
                # 처음으로 이동: 캡쳐 재열기
                cap.release()
                cap = cv2.VideoCapture(args.input)
                is_playing = False
                window["-STATUS-"].update("상태: 처음으로 이동 (정지)")

            draw_box = values["-BOX-"]
            strength = int(values["-STRENGTH-"])
            pad_x = values["-PADX-"] / 100.0
            pad_y = values["-PADY-"] / 100.0
            min_face = int(values["-MINFACE-"])
            neighbors = int(values["-NEIGH-"])

            if is_playing:
                ok, frame = cap.read()
                if not ok:
                    # 파일 끝 → 정지
                    is_playing = False
                    window["-STATUS-"].update("상태: 끝 (정지)")
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rects = detect_faces(gray, face_cascade, args.scale_factor, neighbors, min_face)

                    vis = frame.copy()
                    # ★ 모든 얼굴에 대해 모자이크 처리
                    for (x,y,w,h) in rects:
                        x2,y2,w2,h2 = expand_box(x,y,w,h, pad_x, pad_y, vis.shape[1], vis.shape[0])
                        if mode_pixelate:
                            pixelate_region(vis, x2,y2,w2,h2, strength=strength)
                        else:
                            gaussian_blur_region(vis, x2,y2,w2,h2, strength=strength)
                        if draw_box:
                            cv2.rectangle(vis, (x2,y2), (x2+w2, y2+h2), (0,255,0), 2)

                    # 저장 토글 시 파일로 기록
                    if is_saving and writer is not None:
                        writer.write(vis)

                    # 화면 표시 (PIL 없이 PNG 인코딩으로 표시)
                    imgbytes = cv2.imencode(".png", vis)[1].tobytes()
                    window["-IMAGE-"].update(data=imgbytes)

            else:
                # 정지 상태에서는 마지막 프레임 유지
                pass

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        window.close()

# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Face Mosaic Tool (scan & render & preview & gui)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_opts(sp):
        sp.add_argument("--scale-factor", type=float, default=1.1, help="Haar scaleFactor(기본 1.1)")
        sp.add_argument("--min-neighbors", type=int, default=5, help="Haar minNeighbors(기본 5)")
        sp.add_argument("--min-face", type=int, default=24, help="최소 얼굴 크기(px, 기본 24)")
        sp.add_argument("--pad-x", type=float, default=0.15, help="가로 패딩 비율(한쪽, 기본 0.15)")
        sp.add_argument("--pad-y", type=float, default=0.20, help="세로 패딩 비율(한쪽, 기본 0.20)")
        sp.add_argument("--preview", action="store_true", help="(scan/render) 주기적 미리보기 창 표시")
        sp.add_argument("--preview-stride", type=int, default=15, help="(scan/render) N프레임마다 미리보기")

    # 스캔
    s = sub.add_parser("scan", help="얼굴 자동 탐지 후 CSV로 내보내기")
    s.add_argument("-i","--input", required=True, help="입력 동영상 경로")
    s.add_argument("-o","--output-csv", default="faces_auto.csv", help="출력 CSV 경로")
    add_common_opts(s)

    # 렌더
    r = sub.add_parser("render", help="CSV(선택) 기반으로 모자이크 영상 렌더링")
    r.add_argument("-i","--input", required=True, help="입력 동영상 경로")
    r.add_argument("-o","--output", default="mosaic_output.mp4", help="출력 동영상 경로")
    r.add_argument("--csv", help="스캔 결과 CSV 경로(없으면 자동 탐지)")
    r.add_argument("--mode", choices=["pixelate","gaussian"], default="pixelate", help="모자이크 방식 선택")
    r.add_argument("--strength", type=int, default=18, help="강도: 픽셀 크기 또는 블러 커널(기본 18)")
    r.add_argument("--draw-box", action="store_true", help="박스 시각화(디버깅)")
    add_common_opts(r)

    # 프리뷰(키보드/슬라이더)
    pvw = sub.add_parser("preview", help="실시간 프리뷰(키보드/슬라이더, r=저장)")
    pvw.add_argument("-i","--input", required=True, help="입력 동영상 경로")
    pvw.add_argument("-o","--output", default="preview_record.mp4", help="녹화 파일 경로(r키로 녹화)")
    pvw.add_argument("--scale-factor", type=float, default=1.1, help="Haar scaleFactor(기본 1.1)")
    pvw.add_argument("--min-neighbors", type=int, default=5, help="초기 Neighbors")
    pvw.add_argument("--min-face", type=int, default=24, help="초기 MinFace")
    pvw.add_argument("--pad-x", type=float, default=0.15, help="초기 PadX")
    pvw.add_argument("--pad-y", type=float, default=0.20, help="초기 PadY")

    # GUI(버튼 포함)
    gui = sub.add_parser("gui", help="GUI(재생/일시정지/저장/종료 버튼)")
    gui.add_argument("-i","--input", required=True, help="입력 동영상 경로")
    gui.add_argument("-o","--output", default="gui_output.mp4", help="저장 파일 경로")
    gui.add_argument("--scale-factor", type=float, default=1.1, help="Haar scaleFactor(기본 1.1)")
    gui.add_argument("--min-neighbors", type=int, default=5, help="초기 Neighbors")
    gui.add_argument("--min-face", type=int, default=24, help="초기 MinFace")
    gui.add_argument("--pad-x", type=float, default=0.15, help="초기 PadX")
    gui.add_argument("--pad-y", type=float, default=0.20, help="초기 PadY")

    return p

def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(2)
    args = parser.parse_args()
    try:
        if args.cmd == "scan":
            cmd_scan(args)
        elif args.cmd == "render":
            cmd_render(args)
        elif args.cmd == "preview":
            cmd_preview(args)
        elif args.cmd == "gui":
            cmd_gui(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        # 사용자가 Ctrl+C 눌러도 깔끔 종료
        print("\n[INFO] 사용자 중단. 안전 종료.")
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()
