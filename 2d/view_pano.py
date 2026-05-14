"""
全景查看器 —— 在浏览器里以中心视角浏览 equirectangular 全景图

用法:
    # 单张
    python view_pano.py --input ../20260129135824/CAM/600.267317_IMG.jpg

    # 整个目录（用 ←/→ 翻图）
    python view_pano.py --input ../20260129135824/CAM

    # 浏览带框的检测结果
    python view_pano.py --input out_example_owlv2 --glob "*_annot.jpg"

操作:
    拖动 = 转视角；滚轮 = 缩放 (FoV)；←/→ = 前后图；R = 重置视角

实现:
    Python 起本地 http.server，浏览器加载 pano_viewer/index.html
    用 three.js 把全景贴在内表面 sphere 上，相机放球心。
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

# 工作区根目录（脚本所在 2d/ 的上级）
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
VIEWER_HTML_REL = "2d/pano_viewer/index.html"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """静音版 SimpleHTTPRequestHandler，只在出错时打印。"""
    # 由 main() 在启动前注入：访问 / 时跳转到的 viewer URL
    redirect_target: str | None = None

    def log_message(self, fmt, *args):
        if "200" not in (args[1] if len(args) > 1 else ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path in ("/", "") and self.redirect_target:
            self.send_response(302)
            self.send_header("Location", self.redirect_target)
            self.end_headers()
            return
        super().do_GET()


def main():
    ap = argparse.ArgumentParser(description="全景查看器")
    ap.add_argument("--input", required=True,
                    help="单张 jpg/png 文件，或包含全景图的目录")
    ap.add_argument("--glob", default="*.jpg", help="目录模式下匹配，默认 *.jpg")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    p = Path(args.input).expanduser().resolve()
    if p.is_file():
        imgs = [p]
    elif p.is_dir():
        imgs = sorted(p.rglob(args.glob))
    else:
        print(f"[error] input not found: {p}", file=sys.stderr); sys.exit(1)

    if not imgs:
        print(f"[error] no images matched in {p}", file=sys.stderr); sys.exit(1)

    # 让所有路径都相对工作区根，再统一通过 http server 提供
    rel_imgs: list[str] = []
    for img in imgs:
        try:
            rel_imgs.append(str(img.relative_to(WORKSPACE_ROOT)))
        except ValueError:
            print(f"[error] image {img} is outside workspace {WORKSPACE_ROOT}",
                  file=sys.stderr); sys.exit(1)

    cfg = {"images": rel_imgs}
    fragment = urllib.parse.quote(json.dumps(cfg))
    viewer_path = f"/{VIEWER_HTML_REL}#{fragment}"
    url = f"http://127.0.0.1:{args.port}{viewer_path}"

    os.chdir(WORKSPACE_ROOT)
    _QuietHandler.redirect_target = viewer_path
    try:
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", args.port),
                                                _QuietHandler)
    except OSError as e:
        print(f"[error] cannot bind port {args.port}: {e}\n"
              f"        加 --port 用别的端口", file=sys.stderr)
        sys.exit(1)

    print(f"[serve] root  = {WORKSPACE_ROOT}")
    print(f"[serve] http  = http://127.0.0.1:{args.port}/")
    print(f"[serve] open  = {url}")
    print(f"[serve] images={len(rel_imgs)}")
    print("Press Ctrl+C to stop.")

    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    if not args.no_open:
        webbrowser.open(url)

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n[bye]")
        httpd.shutdown()


if __name__ == "__main__":
    main()
