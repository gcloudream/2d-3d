# 2d — 门窗识别

基于开放词汇目标检测模型，对图像（含 360° 全景）做门 / 窗定位。

当前主力模型：**`google/owlv2-base-patch16-ensemble`**（OWLv2，Google）。

## 目录结构

```
2d/
├── detect_owlvit.py        ★ 主脚本：OWLv2 / OWL-ViT 门窗检测
├── detect_doorwindow.py     备选脚本：Florence-2 (描述类任务时仍可用)
├── requirements.txt
├── README.md
└── out_example_owlv2/       一次成功运行的输出示例
```

## 模型选择背景

试了三种路线，最后定在 OWLv2：

| 模型 | 结果 | 备注 |
|---|---|---|
| Florence-2-large | 大量幻觉框 + 门窗互锁套娃 | 不是检测器、无分数、必须每类各调一次 |
| OWL-ViT base/32 | 0 检测，分数全 < 0.03 | 模型偏弱，patch32 粒度粗 |
| **OWLv2 base/16-ens** | 4 个有意义的 window 框 | 默认选用 |

## 安装

仓库根目录的 `.venv` 中安装：

```bash
../.venv/bin/pip install -r requirements.txt
```

国内网络，OWLv2 权重需要直连 `huggingface.co`（hf-mirror 对该模型未代理）。脚本里默认设置了 `HF_ENDPOINT=https://hf-mirror.com`，遇到下载失败可以临时切回官方源：

```bash
HF_ENDPOINT=https://huggingface.co ../.venv/bin/python detect_owlvit.py ...
```

首次加载会从 Hugging Face 下载约 600 MB 权重，缓存在 `~/.cache/huggingface`。

## 快速开始

单张全景：

```bash
../.venv/bin/python detect_owlvit.py \
    --input ../20260129135824/CAM/592.692836_IMG.jpg \
    --auto-pano
```

批量（前 20 张）：

```bash
../.venv/bin/python detect_owlvit.py \
    --input ../20260129135824/CAM \
    --glob "*.jpg" --limit 20 \
    --auto-pano
```

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `google/owlv2-base-patch16-ensemble` | 也支持 `google/owlvit-base-patch32` 等 OWL-ViT v1 权重 |
| `--classes` | `door,window` | 类别列表。同一类别多个 prompt 用 ` \| ` 分隔 |
| `--score-thr` | `0.15` | OWLv2 输出分数阈值 |
| `--auto-pano` | off | 输入是 2:1 全景图时自动切 6 张透视图 |
| `--pano-split` | `0` | 显式指定切 N 张透视图（不依赖 auto-pano） |
| `--pano-fov` | `90` | 每张透视图 FoV 角度 |
| `--pano-out-size` | `1024` | 每张透视图分辨率 |
| `--nms-iou` | `0.4` | 跨视图同类别 NMS 阈值 |
| `--cross-label-iou` | `0.5` | 跨类别 NMS（IoU） |
| `--cross-label-containment` | `0.7` | 跨类别 NMS（包含度，避免门框里又冒出窗框） |
| `--min-area` | `0.002` | 最小框面积比例（过滤幻觉小框） |
| `--device` | `auto` | `auto` / `mps` / `cuda` / `cpu` |

## 多 prompt 用法

OWLv2 对 prompt 敏感，给同一类别多个表达通常能显著提升召回：

```bash
--classes 'door|a door|a wooden door|a glass door|a sliding door,window|a window|a glass window|a sliding window'
```

每个候选框会和所有 prompt 算相似度，最高分者胜出，最终框上的 label 仍是该类首项（`door` / `window`）。

## 输出

默认落到 `2d/out/`：

- `<stem>_annot.jpg` 带框 + 类别 + 分数的可视化图
- `<stem>.json` 单图结构化结果，包含 `detections: [{label, score, bbox, source_yaw}]`
- `detections.jsonl` 本次运行的汇总（每行一张图）

## 全景处理流程

输入若是 5760×2880 equirectangular 全景：

1. 按 `--pano-split` 切 N 张透视图（默认 N=6，FoV=90°，相邻视图带重叠）
2. 每张透视图独立做检测
3. 把 bbox 反投影到全景 (u, v) 坐标
4. 跨视图 NMS（环形 IoU，处理 ±180° 接缝）
5. 跨类别 NMS（IoU + 包含度），消除"门框套窗框"
6. 全景分辨率下做最终面积过滤

## 性能（M3 + 24 GB MPS）

| 配置 | 单张延迟 | 全套 753 张 |
|---|---|---|
| OWLv2 base + 6 视图 + 768 输入 | ~10 s | ~2 小时 |
| OWLv2 base + 6 视图 + 1024 输入（默认） | ~30 s | ~6 小时 |
| 单图（不切全景） | ~3 s | — |

加速建议：`--pano-split 4 --pano-out-size 768`

## 已知局限

- OWLv2 对 base 版仍有漏检，特别是被遮挡或姿态特殊的门
- 升级到 `google/owlv2-large-patch14-ensemble`（约 2.7 GB）能进一步提升召回，这台 M3 + 24 GB 跑得动
- 室内场景中"玻璃门 vs 落地窗""橱柜门 vs 真实通行门"在视觉上接近，开放词汇模型本身分得不彻底

## Florence-2（备用）

`detect_doorwindow.py` 仍保留。它在检测任务上不如 OWLv2，但做以下任务时仍可用：
- 区域/全图 caption（`<CAPTION>` / `<DENSE_REGION_CAPTION>`）
- OCR
- 短语定位（`<CAPTION_TO_PHRASE_GROUNDING>`）

用法见脚本顶部 docstring。
