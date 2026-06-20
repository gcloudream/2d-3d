"""3d_viewer 入口。

用法:
    cd /Users/gengchen/Desktop/3dtiqu
    .venv/bin/python 3d_viewer/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 from core/render/ui 这种相对导入能用
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow

WORKSPACE = HERE.parent  # /Users/gengchen/Desktop/3dtiqu


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow(WORKSPACE, prompt_for_dataset=True)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
