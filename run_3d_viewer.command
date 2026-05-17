#!/usr/bin/env bash
# macOS 双击启动：Finder 会用 Terminal.app 打开 .command 文件。
# 首次运行需右键 → 打开，之后可直接双击。

set -euo pipefail

cd "$(dirname "$0")"

echo "=== 3D Viewer: Point Cloud + Panorama ==="

# 创建虚拟环境（若不存在）
if [ ! -x ".venv/bin/python" ]; then
  echo "→ 创建虚拟环境..."
  uv venv .venv
fi

# 安装/同步依赖
echo "→ 检查依赖..."
uv pip install --quiet --python .venv/bin/python -r 3d_viewer/requirements.txt

echo "→ 启动..."
.venv/bin/python 3d_viewer/main.py
