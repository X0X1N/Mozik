import os, cv2, time, threading, collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Deque, Set
from PIL import Image, ImageTk  # pillow

# -----------------------------
# 크래시 방지: 비디오 백엔드/스레드 설정 (가장 먼저!)
# -----------------------------
# Windows 기본 MSMF 백엔드가 HEVC/가변프레임 등에서 튕기는 경우가 많아 FFmpeg 우선으로 전환
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
# 불필요한 로그 억제
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
# OpenCV 내부 스레드로 인한 교착/크래시 방지
cv2.setNumThreads(0)

# 내부 모듈 (프로젝트에 이미 존재)
from .core import clamp, expand_box, pixelate_region, gaussian_blur_region
from .detect import load_haar_face, detect_faces
from .core import Box as TrackableBox, assign_ids

# -----------------------------
# 설정 상수
# -----------------------------
MAX_PREVIEW_W = 1280             # 미리보기 최대 너비(성능)
DETECT_EVERY_N = 2               # 프레임 N개마다 검출(성능)
THUMB_SIZE = 84                  # 얼굴 썸네일 크기

# -----------------------------
# 보조 데이터
# -----------------------------
@dataclass
class UBox:
    x:int; y:int; w:int; h:int
    def as_tuple(self): return (self.x, self.y, self.w, self.h)

@dataclass
class TrackBox:
    frame:int; t_ms:int; id:int; x:int; y:int; w:int; h:int; source:str="auto"

# -----------------------------
# 백그라운드 프레임 리더 (버퍼링)
# -----------------------------
class VideoReader:
    def __init__(self, path:str, buffer_size:int=30):
        # 일부 코덱에서 직접 열다가 죽는 케이스 방지용 try
        self.cap = cv2.VideoCapture(path)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없음: {path}")
        self.path = path
        self.buffer: Deque[Tuple[int, any]] = collections.deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.stop = False

        # 일부 드라이버가 첫 프레임 접근에서 죽는 이슈 회피
        try:
            with self.lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass

        # 순방향 재생용 버퍼 (사용 안 해도 안정성 위해 try-protect)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop:
            try:
                with self.lock:
                    if len(self.buffer) < self.buffer.maxlen:
                        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        ok, frame = self.cap.read()
                        if not ok or frame is None:
                            break
                        self.buffer.append((int(pos), frame))
                time.sleep(0.001)
            except Exception:
                # 백그라운드 버퍼링 문제는 조용히 종료
                break

    # 안전가드: 실패시 None 반환
    def read_at(self, index:int) -> Optional[any]:
        """요청 프레임으로 직접 이동해서 읽기. 실패 시 None 반환(충돌 방지)"""
        with self.lock:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
        return frame if ok and frame is not None else None

    # 안전 종료
    def close(self):
        self.stop = True
        try:
            self.thread.join(timeout=0.5)
        except:
            pass
        try:
            with self.lock:
                if self.cap is not None:
                    self.cap.release()
        except:
            pass

# -----------------------------
# Tk 스튜디오
# -----------------------------
class Studio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mosaic Studio")
        self.geometry("1280x820")
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except:
            pass

        # 상태
        self.reader: Optional[VideoReader] = None
        self.video_path = None
        self.W = 0; self.H = 0; self.fps = 30.0; self.frame_count = 0
        self.current_idx = 0
        self.range_start = 0; self.range_end = 0

        # 파라미터
        self.mode = tk.StringVar(value="pixelate")
        self.strength = tk.IntVar(value=20)
        self.padx = tk.DoubleVar(value=0.15)
        self.pady = tk.DoubleVar(value=0.20)
        self.min_face = tk.IntVar(value=24)
        self.neighbors = tk.IntVar(value=5)
        self.blur_all = tk.BooleanVar(value=True)

        # 내부 상태
        self.manual_boxes: List[UBox] = []
        self.dragging_idx=None; self.resizing_idx=None
        self.drag_offset=(0,0); self.resize_anchor=None
        self._playing = False
        self._last_preview: Optional[any] = None
        self._scale_preview = 1.0
        self.detect_cache: Dict[int, List[Tuple[int,int,int,int]]] = {}  # frame->rects
        self._detect_dirty = True    # 파라미터 변화시 캐시 무시

        # UI
        self._build_ui()
        self._bind_shortcuts()

        # 초기 렌더 지연: 캔버스 크기 1x1일 때 PIL 변환에서 죽는 케이스 회피
        self.after(50, lambda: None)

    # ---------------------
    # UI
    # ---------------------
    def _build_ui(self):
        root = ttk.Frame(self); root.pack(fill="both", expand=True)
        left = ttk.Frame(root); left.pack(side="left", fill="y", padx=10, pady=10)
        right = ttk.Frame(root); right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # 좌측: 컨트롤
        ttk.Button(left, text="① 파일 열기", command=self.open_video).pack(fill="x", pady=4)

        sec = ttk.LabelFrame(left, text="② 구간 선택")
        sec.pack(fill="x", pady=8)
        row1 = ttk.Frame(sec); row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="시작(s)").pack(side="left")
        self.entry_start = ttk.Entry(row1, width=7); self.entry_start.insert(0,"0"); self.entry_start.pack(side="left", padx=4)
        ttk.Label(row1, text="끝(s)").pack(side="left")
        self.entry_end = ttk.Entry(row1, width=7); self.entry_end.insert(0,"0"); self.entry_end.pack(side="left", padx=4)
        ttk.Button(sec, text="구간적용", command=self.apply_range).pack(fill="x", pady=2)

        sec3 = ttk.LabelFrame(left, text="③ 얼굴 선택")
        sec3.pack(fill="x", pady=8)
        ttk.Button(sec3, text="현재 프레임에서 얼굴 탐지", command=self.detect_on_current).pack(fill="x", pady=2)
        ttk.Checkbutton(sec3, text="얼굴 전부 모자이크", variable=self.blur_all, command=self._mark_detect_dirty).pack(anchor="w")

        self.thumbs = ttk.LabelFrame(left, text="선택한 얼굴(체크된 것만)")
        self.thumbs.pack(fill="both", expand=True, pady=6)
        self.thumb_vars: List[Tuple[int, tk.BooleanVar]] = []  # (id, var)

        sec4 = ttk.LabelFrame(left, text="④ 모자이크 설정")
        sec4.pack(fill="x", pady=8)
        ttk.Label(sec4, text="모드").pack(anchor="w")
        ttk.Radiobutton(sec4, text="픽셀", variable=self.mode, value="pixelate", command=self._mark_detect_dirty).pack(anchor="w")
        ttk.Radiobutton(sec4, text="가우시안", variable=self.mode, value="gaussian", command=self._mark_detect_dirty).pack(anchor="w")
        ttk.Label(sec4, text="강도").pack(anchor="w")
        ttk.Scale(sec4, from_=1, to=60, variable=self.strength, orient="horizontal").pack(fill="x")
        ttk.Label(sec4, text="패딩 X/Y").pack(anchor="w")
        pad = ttk.Frame(sec4); pad.pack(fill="x")
        ttk.Scale(pad, from_=0, to=0.5, variable=self.padx, orient="horizontal").pack(side="left", fill="x", expand=True, padx=(0,4))
        ttk.Scale(pad, from_=0, to=0.5, variable=self.pady, orient="horizontal").pack(side="left", fill="x", expand=True)

        det = ttk.LabelFrame(left, text="얼굴 탐지 민감도")
        det.pack(fill="x", pady=8)
        ttk.Label(det, text="MinFace / Neigh").pack(anchor="w")
        rowd = ttk.Frame(det); rowd.pack(fill="x")
        ttk.Scale(rowd, from_=8, to=200, variable=self.min_face, orient="horizontal", command=lambda *_: self._mark_detect_dirty()).pack(side="left", fill="x", expand=True, padx=(0,4))
        ttk.Scale(rowd, from_=0, to=15, variable=self.neighbors, orient="horizontal", command=lambda *_: self._mark_detect_dirty()).pack(side="left", fill="x", expand=True)

        sec5 = ttk.LabelFrame(left, text="⑤ 수동 박스")
        sec5.pack(fill="x", pady=8)
        ttk.Label(sec5, text="캔버스 드래그=추가 / 박스 클릭=이동 / 모서리=리사이즈").pack(anchor="w")
        rowb = ttk.Frame(sec5); rowb.pack(fill="x")
        ttk.Button(rowb, text="마지막 박스 삭제", command=self._delete_last_box).pack(side="left", fill="x", expand=True, padx=(0,4))
        ttk.Button(rowb, text="전체 삭제", command=self._clear_boxes).pack(side="left", fill="x", expand=True)

        ttk.Button(left, text="⑥ 내보내기 (MP4)", command=self.export_video).pack(fill="x", pady=12)

        # 우측: 미리보기 + 타임라인
        pv = ttk.Frame(right); pv.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(pv, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        tl = ttk.Frame(right); tl.pack(fill="x", pady=8)
        self.btn_home = ttk.Button(tl, text="⏮ 처음", command=lambda: self.seek(self.range_start)); self.btn_home.pack(side="left")
        self.btn_play = ttk.Button(tl, text="▶ 재생", command=self.play); self.btn_play.pack(side="left", padx=6)
        ttk.Button(tl, text="⏸ 정지", command=self.stop).pack(side="left")
        self.scale = ttk.Scale(tl, from_=0, to=0, orient="horizontal", command=self._on_scale); self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.lbl_time = ttk.Label(tl, text="00:00 / 00:00"); self.lbl_time.pack(side="left", padx=(6,6))

        # 초로 이동 UI
        jump = ttk.Frame(right); jump.pack(fill="x", pady=(0,8))
        ttk.Label(jump, text="초로 이동").pack(side="left")
        self.entry_jump = ttk.Entry(jump, width=8); self.entry_jump.pack(side="left", padx=6)
        ttk.Button(jump, text="이동", command=self._jump_to_time).pack(side="left")

        self.status = ttk.Label(right, text="파일을 열어주세요."); self.status.pack(anchor="w", pady=(2,0))

    def _bind_shortcuts(self):
        self.bind("<space>", lambda e: (self.stop() if self._playing else self.play()))
        self.bind("<Left>", lambda e: self.seek(self.current_idx-1))
        self.bind("<Right>", lambda e: self.seek(self.current_idx+1))
        self.bind("<Shift-Left>", lambda e: self.seek(self.current_idx-10))
        self.bind("<Shift-Right>", lambda e: self.seek(self.current_idx+10))

    # ---------------------
    # 비디오 열기/정보
    # ---------------------
    def open_video(self):
        path = filedialog.askopenfilename(
            title="동영상 선택",
            filetypes=[("Video","*.mp4;*.mov;*.mkv;*.avi;*.webm;*.mpg;*.mpeg")]
        )
        if not path:
            return
        try:
            path = os.path.normpath(path)
            if self.reader:
                self.reader.close()
            self.reader = VideoReader(path)
            self.video_path = path

            cap = self.reader.cap
            self.W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 0
            self.H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            # 안전 보정
            if self.fps <= 0:
                self.fps = 30.0  # 안전 기본값
            if self.W <= 0 or self.H <= 0 or self.frame_count <= 0:
                raise RuntimeError(
                    "해당 코덱/포맷을 열 수 없습니다. (HEVC/H.265/가변프레임 가능)\n"
                    "ffmpeg로 H.264(yuv420p)로 변환 후 다시 시도하세요."
                )

            # 첫 프레임 확인
            if self.reader.read_at(0) is None:
                raise RuntimeError("첫 프레임을 읽을 수 없습니다. (코덱/권한/경로 확인)")

            # 타임라인/구간 설정
            self.range_start, self.range_end = 0, self.frame_count - 1
            self.scale.configure(to=self.frame_count - 1)
            dur = (self.frame_count - 1) / self.fps
            self.entry_start.delete(0,"end"); self.entry_start.insert(0,"0")
            self.entry_end.delete(0,"end");   self.entry_end.insert(0,f"{dur:.2f}")
            self._update_status()
            self.seek(0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "영상 열기 실패",
                f"{e}\n\n해결 팁:\n"
                "- ffmpeg 변환: ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -c:a aac -movflags +faststart output_h264.mp4\n"
                "- 다른 파일로 테스트\n- 관리자 권한/경로(한글/공백) 점검"
            )

    def _update_status(self):
        if not self.video_path:
            self.status.config(text="파일을 열어주세요."); return
        dur = (self.frame_count-1)/self.fps
        self.status.config(text=f"{os.path.basename(self.video_path)} | {self.W}x{self.H} | {self.fps:.2f}fps | {dur:.2f}s")

    # ---------------------
    # 탐지/캐시
    # ---------------------
    def _mark_detect_dirty(self, *_):
        self._detect_dirty = True

    def _detect_faces_cached(self, frame_idx:int, frame_bgr:any) -> List[Tuple[int,int,int,int]]:
        # 캐시 재사용 + N프레임마다 재검출
        if not self._detect_dirty and frame_idx in self.detect_cache:
            return self.detect_cache[frame_idx]

        do_detect = (frame_idx % DETECT_EVERY_N == 0) or self._detect_dirty or (frame_idx not in self.detect_cache)
        if not do_detect:
            # 직전 캐시 가까운 값 사용
            return self.detect_cache.get(frame_idx, self.detect_cache.get(frame_idx-1, []))

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        try:
            if not hasattr(self, "_cascade"):
                self._cascade = load_haar_face()
            rects = detect_faces(gray, self._cascade, 1.1, int(self.neighbors.get()), int(self.min_face.get()))
        except Exception as e:
            rects = []
            print("[WARN] detect failed:", e)
        self.detect_cache[frame_idx] = [(int(x),int(y),int(w),int(h)) for (x,y,w,h) in rects]
        self._detect_dirty = False
        return self.detect_cache[frame_idx]

    # ---------------------
    # 시킹/재생
    # ---------------------
    def _format_time(self, sec:float)->str:
        m = int(sec//60); s = int(sec%60)
        return f"{m:02d}:{s:02d}"

    def seek(self, idx:int):
        if not self.reader:
            return
        idx = clamp(idx, self.range_start, self.range_end)
        frame = self.reader.read_at(idx)
        if frame is None:
            return  # 문제 프레임 스킵
        self.current_idx = idx
        self.scale.set(idx)
        self._render_preview(frame)
        cur = self._format_time(idx/self.fps)
        dur = self._format_time(self.range_end/self.fps)
        self.lbl_time.config(text=f"{cur} / {dur}")

    def _on_scale(self, v):
        if not self.reader:
            return
        self.seek(int(float(v)))

    def apply_range(self):
        if not self.reader:
            return
        try:
            s = max(0.0, float(self.entry_start.get()))
            e = max(0.0, float(self.entry_end.get()))
        except:
            messagebox.showerror("오류","시작/끝을 초 단위로 입력하세요.")
            return
        s_idx, e_idx = int(s*self.fps), int(e*self.fps)
        s_idx = clamp(s_idx, 0, self.frame_count-1)
        e_idx = clamp(e_idx, 0, self.frame_count-1)
        if e_idx <= s_idx:
            messagebox.showerror("오류","끝이 시작보다 커야 합니다.")
            return
        self.range_start, self.range_end = s_idx, e_idx
        self.seek(s_idx)

    def play(self):
        if not self.reader:
            return
        self._playing = True
        self.btn_play.config(text="⏸ 일시정지")
        self._loop_play()

    def stop(self):
        self._playing = False
        self.btn_play.config(text="▶ 재생")

    def _loop_play(self):
        if not self._playing:
            return
        nxt = self.current_idx + 1
        if nxt > self.range_end:
            nxt = self.range_start
        self.seek(nxt)
        delay = int(1000/self.fps) if self.fps > 0 else 33
        self.after(delay, self._loop_play)

    # ---------------------
    # 미리보기 렌더
    # ---------------------
    def _fit_to_canvas(self, frame):
        cw = self.canvas.winfo_width() or MAX_PREVIEW_W
        ch = self.canvas.winfo_height() or int(MAX_PREVIEW_W*9/16)
        h, w = frame.shape[:2]
        # 0 division 회피
        if w <= 0 or h <= 0:
            w, h = 1, 1
        scale = min(cw/w, ch/h, MAX_PREVIEW_W/w)
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        offx = (cw - nw)//2; offy = (ch - nh)//2
        try:
            resized = cv2.resize(frame, (nw,nh))
        except Exception:
            resized = frame
        return resized, scale, offx, offy

    def _render_preview(self, frame_bgr:any):
        try:
            rects = self._detect_faces_cached(self.current_idx, frame_bgr)
            tracks = assign_ids([], rects, self.current_idx, int(self.current_idx*1000/self.fps))

            vis = frame_bgr.copy()
            # 자동 처리
            selected_ids: Set[int] = set()
            if not self.blur_all.get():
                for fid, var in self.thumb_vars:
                    if var.get():
                        selected_ids.add(fid)

            for tb in tracks:
                x,y,w,h = expand_box(tb.x,tb.y,tb.w,tb.h,
                                     float(self.padx.get()), float(self.pady.get()),
                                     self.W, self.H)
                if self.blur_all.get() or (tb.id in selected_ids):
                    if self.mode.get()=="pixelate":
                        pixelate_region(vis, x,y,w,h, int(self.strength.get()))
                    else:
                        gaussian_blur_region(vis, x,y,w,h, int(self.strength.get()))

            # 수동 박스
            for b in self.manual_boxes:
                if self.mode.get()=="pixelate":
                    pixelate_region(vis, b.x,b.y,b.w,b.h, int(self.strength.get()))
                else:
                    gaussian_blur_region(vis, b.x,b.y,b.w,b.h, int(self.strength.get()))

            img, s, ox, oy = self._fit_to_canvas(vis)
            self._scale_preview = s
            self._last_preview = frame_bgr  # 원본 보관(히트테스트용)
            self.canvas.delete("all")

            # PIL 변환 예외 방지
            try:
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(pil)
            except Exception:
                # 이미지가 손상되었거나 PIL이 실패하면 미리보기 생략
                return
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0,0, anchor="nw", image=imgtk)

            # 수동 박스 테두리/핸들
            for i,b in enumerate(self.manual_boxes):
                x1 = int(b.x*s)+ox; y1=int(b.y*s)+oy
                x2 = int((b.x+b.w)*s)+ox; y2 = int((b.y+b.h)*s)+oy
                self.canvas.create_rectangle(x1,y1,x2,y2, outline="#00ff88", width=2)
                r=5
                for hx,hy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    self.canvas.create_rectangle(hx-r,hy-r,hx+r,hy+r, fill="#00ff88", outline="")
        except Exception as e:
            import traceback
            traceback.print_exc()

    # ---------------------
    # 얼굴 탐지 → 썸네일 체크
    # ---------------------
    def detect_on_current(self):
        if not self.reader:
            return
        frame = self.reader.read_at(self.current_idx)
        if frame is None:
            return
        rects = self._detect_faces_cached(self.current_idx, frame)
        tracks = assign_ids([], rects, self.current_idx, int(self.current_idx*1000/self.fps))
        # 썸네일 다시 구성
        for w in self.thumbs.winfo_children():
            w.destroy()
        self.thumb_vars.clear()
        row=col=0
        for tb in tracks:
            x,y,w,h = expand_box(tb.x,tb.y,tb.w,tb.h, float(self.padx.get()), float(self.pady.get()), self.W, self.H)
            crop = frame[y:y+h, x:x+w]
            if crop.size==0:
                continue
            try:
                thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)))
            except Exception:
                continue
            var = tk.BooleanVar(value=True)
            cell = ttk.Frame(self.thumbs); cell.grid(row=row, column=col, padx=4, pady=4, sticky="w")
            ttk.Checkbutton(cell, text=f"ID {tb.id}", variable=var, command=self._mark_detect_dirty).pack(anchor="w")
            lbl = ttk.Label(cell, image=imgtk); lbl.image = imgtk; lbl.pack()
            self.thumb_vars.append((tb.id, var))
            col += 1
            if col >= 3:
                col=0; row+=1
        # 미리보기 갱신
        self._render_preview(frame)

    # ---------------------
    # 수동 박스(드래그/리사이즈)
    # ---------------------
    def _canvas_to_video(self, cx, cy):
        if self._last_preview is None:
            return 0,0
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        s = self._scale_preview
        nw, nh = int(self.W*s), int(self.H*s)
        ox = (cw-nw)//2; oy = (ch-nh)//2
        x = int((cx-ox)/s); y = int((cy-oy)/s)
        return clamp(x,0,self.W-1), clamp(y,0,self.H-1)

    def _hit_handle(self, cx, cy):
        s = self._scale_preview
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        nw, nh = int(self.W*s), int(self.H*s)
        ox = (cw-nw)//2; oy = (ch-nh)//2
        for i,b in enumerate(self.manual_boxes):
            x1 = int(b.x*s)+ox; y1=int(b.y*s)+oy
            x2 = int((b.x+b.w)*s)+ox; y2=int((b.y+b.h)*s)+oy
            for idx,(hx,hy) in enumerate([(x1,y1),(x2,y1),(x1,y2),(x2,y2)]):
                if abs(cx-hx)<=6 and abs(cy-hy)<=6:
                    return i, idx
        return None, None

    def _hit_box(self, cx, cy):
        s = self._scale_preview
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        nw, nh = int(self.W*s), int(self.H*s)
        ox = (cw-nw)//2; oy = (ch-nh)//2
        for i,b in enumerate(self.manual_boxes):
            x1 = int(b.x*s)+ox; y1=int(b.y*s)+oy
            x2 = int((b.x+b.w)*s)+ox; y2 = int((b.y+b.h)*s)+oy
            if x1<=cx<=x2 and y1<=cy<=y2:
                return i
        return None

    def _on_mouse_down(self, e):
        if not self.reader:
            return
        hi,hcorner = self._hit_handle(e.x, e.y)
        if hi is not None:
            self.resizing_idx = hi; self.resize_anchor = hcorner; return
        bi = self._hit_box(e.x, e.y)
        if bi is not None:
            self.dragging_idx = bi
            vx, vy = self._canvas_to_video(e.x, e.y)
            b = self.manual_boxes[bi]
            self.drag_offset = (vx - b.x, vy - b.y)
        else:
            self.dragging_idx=None; self.resizing_idx=None
            self._new_box_start = (e.x, e.y)

    def _on_mouse_drag(self, e):
        if not self.reader or self._last_preview is None:
            return
        if self.resizing_idx is not None:
            b = self.manual_boxes[self.resizing_idx]
            vx, vy = self._canvas_to_video(e.x, e.y)
            x2, y2 = b.x+b.w, b.y+b.h
            if self.resize_anchor == 0:
                nx,ny = vx,vy; nw,nh = x2-nx, y2-ny
                if nw>2 and nh>2: b.x,b.y,b.w,b.h = nx,ny,nw,nh
            elif self.resize_anchor == 1:
                nx,ny = b.x,vy; nw,nh = vx-b.x, y2-ny
                if nw>2 and nh>2: b.y,b.w,b.h = ny,nw,nh
            elif self.resize_anchor == 2:
                nx,ny = vx,b.y; nw,nh = x2-nx, vy-b.y
                if nw>2 and nh>2: b.x,b.w,b.h = nx,nw,nh
            else:
                nw,nh = vx-b.x, vy-b.y
                if nw>2 and nh>2: b.w,b.h = nw,nh
            self.seek(self.current_idx)   # 미리보기 갱신
        elif self.dragging_idx is not None:
            b = self.manual_boxes[self.dragging_idx]
            vx, vy = self._canvas_to_video(e.x, e.y)
            nx = clamp(vx - self.drag_offset[0], 0, self.W-b.w)
            ny = clamp(vy - self.drag_offset[1], 0, self.H-b.h)
            b.x, b.y = nx, ny
            self.seek(self.current_idx)
        else:
            pass

    def _on_mouse_up(self, e):
        if not self.reader:
            return
        if hasattr(self, "_new_box_start") and self._new_box_start:
            sx, sy = self._new_box_start
            vx1, vy1 = self._canvas_to_video(sx, sy)
            vx2, vy2 = self._canvas_to_video(e.x, e.y)
            x = min(vx1, vx2); y = min(vy1, vy2)
            w = abs(vx2 - vx1); h = abs(vy2 - vy1)
            if w > 8 and h > 8:
                self.manual_boxes.append(UBox(x,y,w,h))
                self.seek(self.current_idx)
        self.dragging_idx=None; self.resizing_idx=None
        self._new_box_start=None

    def _delete_last_box(self):
        if self.manual_boxes:
            self.manual_boxes.pop(); self.seek(self.current_idx)

    def _clear_boxes(self):
        self.manual_boxes.clear(); self.seek(self.current_idx)

    # ---------------------
    # “초로 이동” 기능
    # ---------------------
    def _jump_to_time(self):
        if not self.reader:
            return
        try:
            sec = float(self.entry_jump.get())
        except:
            messagebox.showerror("오류", "숫자(초)를 입력하세요.")
            return
        idx = int(sec * self.fps)
        self.seek(idx)

    # ---------------------
    # 내보내기
    # ---------------------
    def export_video(self):
        if not self.reader:
            messagebox.showinfo("안내","먼저 파일을 여세요."); return
        out = filedialog.asksaveasfilename(defaultextension=".mp4",
                 filetypes=[("MP4","*.mp4")], title="내보내기 파일")
        if not out: return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out, fourcc, self.fps, (self.W,self.H))
        if not writer.isOpened():
            messagebox.showerror("오류","출력 파일을 열 수 없습니다."); return

        # 선택된 얼굴 ID
        selected_ids: Set[int] = set()
        if not self.blur_all.get():
            for fid, var in self.thumb_vars:
                if var.get(): selected_ids.add(fid)

        # 진행창
        top = tk.Toplevel(self); top.title("내보내는 중…")
        pb = ttk.Progressbar(top, maximum=(self.range_end-self.range_start+1), length=360)
        pb.pack(padx=20, pady=16)
        msg = ttk.Label(top, text="0%"); msg.pack()
        self.update()

        prev: List[TrackableBox] = []
        try:
            for idx in range(self.range_start, self.range_end+1):
                frame = self.reader.read_at(idx)
                if frame is None:
                    continue  # 문제 프레임은 건너뜀

                rects = self._detect_faces_cached(idx, frame)
                tracks = assign_ids(prev, rects, idx, int(idx*1000/self.fps))
                prev = tracks

                # 자동 얼굴
                for tb in tracks:
                    if self.blur_all.get() or (tb.id in selected_ids):
                        x,y,w,h = expand_box(tb.x,tb.y,tb.w,tb.h, float(self.padx.get()), float(self.pady.get()), self.W, self.H)
                        if self.mode.get()=="pixelate":
                            pixelate_region(frame, x,y,w,h, int(self.strength.get()))
                        else:
                            gaussian_blur_region(frame, x,y,w,h, int(self.strength.get()))

                # 수동 박스
                for b in self.manual_boxes:
                    if self.mode.get()=="pixelate":
                        pixelate_region(frame, b.x,b.y,b.w,b.h, int(self.strength.get()))
                    else:
                        gaussian_blur_region(frame, b.x,b.y,b.w,b.h, int(self.strength.get()))

                writer.write(frame)
                if (idx - self.range_start) % 5 == 0:
                    v = (idx-self.range_start+1)
                    pb['value'] = v
                    pct = 100.0 * v / (self.range_end-self.range_start+1)
                    msg.config(text=f"{pct:.1f}%"); self.update()
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("오류", f"내보내기 중 오류: {e}")
        finally:
            try: writer.release()
            except: pass
            try: top.destroy()
            except: pass
        messagebox.showinfo("완료", f"저장됨: {out}")

def run():
    app = Studio()
    app.mainloop()

if __name__ == "__main__":
    run()
