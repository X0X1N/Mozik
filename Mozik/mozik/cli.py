import sys, argparse
from .pipelines import cmd_scan, cmd_render, cmd_preview, cmd_gui
from .studio_tk import run as cmd_studio


def build_parser():
    p = argparse.ArgumentParser(description="Face Mosaic Tool (scan & render & preview & gui)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_opts(sp):
        sp.add_argument("--scale-factor", type=float, default=1.1)
        sp.add_argument("--min-neighbors", type=int, default=5)
        sp.add_argument("--min-face", type=int, default=24)
        sp.add_argument("--pad-x", type=float, default=0.15)
        sp.add_argument("--pad-y", type=float, default=0.20)
        sp.add_argument("--preview", action="store_true")
        sp.add_argument("--preview-stride", type=int, default=15)

    s = sub.add_parser("scan", help="얼굴 자동 탐지 후 CSV로 내보내기")
    s.add_argument("-i","--input", required=True)
    s.add_argument("-o","--output-csv", default="faces_auto.csv")
    add_common_opts(s)

    r = sub.add_parser("render", help="CSV(선택) 기반 렌더")
    r.add_argument("-i","--input", required=True)
    r.add_argument("-o","--output", default="mosaic_output.mp4")
    r.add_argument("--csv")
    r.add_argument("--mode", choices=["pixelate","gaussian"], default="pixelate")
    r.add_argument("--strength", type=int, default=18)
    r.add_argument("--draw-box", action="store_true")
    add_common_opts(r)

    pvw = sub.add_parser("preview", help="실시간 프리뷰(키보드/슬라이더, r=저장)")
    pvw.add_argument("-i","--input", required=True)
    pvw.add_argument("-o","--output", default="preview_record.mp4")
    pvw.add_argument("--scale-factor", type=float, default=1.1)
    pvw.add_argument("--min-neighbors", type=int, default=5)
    pvw.add_argument("--min-face", type=int, default=24)
    pvw.add_argument("--pad-x", type=float, default=0.15)
    pvw.add_argument("--pad-y", type=float, default=0.20)

    gui = sub.add_parser("gui", help="GUI(재생/일시정지/저장/종료 버튼)")

    gui = sub.add_parser("gui", help="GUI(재생/일시정지/저장/종료 버튼)")
    gui.add_argument("-i","--input", required=True)
    gui.add_argument("-o","--output", default="gui_output.mp4")
    gui.add_argument("--scale-factor", type=float, default=1.1)
    gui.add_argument("--min-neighbors", type=int, default=5)
    gui.add_argument("--min-face", type=int, default=24)
    gui.add_argument("--pad-x", type=float, default=0.15)
    gui.add_argument("--pad-y", type=float, default=0.20)

    studio = sub.add_parser("studio", help="Tk 인터랙티브 스튜디오(업로드→구간선택→얼굴선택→수동박스→내보내기)")


    return p

def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help(); sys.exit(2)
    args = parser.parse_args()
    try:
        if args.cmd == "scan": cmd_scan(args)
        elif args.cmd == "render": cmd_render(args)
        elif args.cmd == "preview": cmd_preview(args)
        elif args.cmd == "gui": cmd_gui(args)
        elif args.cmd == "studio": cmd_studio()
        else: parser.print_help()
    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단. 안전 종료.")
