"""
门窗识别脚本 —— 基于 microsoft/Florence-2-large

用法示例:
    # 单张图
    python detect_doorwindow.py --input ../20260129135824/CAM/600.267317_IMG.jpg

    # 整个目录（递归）
    python detect_doorwindow.py --input ../20260129135824/CAM --glob "*.jpg" --limit 20

    # 全景图模式：5760x2880 equirectangular 按水平方向切 6 张透视图
    python detect_doorwindow.py --input ../20260129135824/CAM --pano-split 6

    # 切到 CPU（默认自动 mps -> cpu）
    python detect_doorwindow.py --input xxx.jpg --device cpu

输出:
    - 在 --output 目录 (默认 2d/out) 下生成
        * <stem>_annot.jpg   带框 + 标签的可视化图（映射回原图坐标）
        * <stem>.json        结构化结果
    - detections.jsonl       汇总所有图片的检测结果 (每行一张图)

全景图说明:
    对 equirectangular (2:1 宽高比) 图像，启用 --pano-split N 后：
    * 按 FoV=90° (默认) 做 N 个等间隔透视投影，相邻视图带 20% 重叠
    * 每张子图独立做检测，然后把 bbox 映射回全景 (x,y)
    * 对跨子图重复框做 IoU 去重
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# 国内网络加载 Hugging Face 模型建议走镜像 (不影响国外网络)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 避免 tokenizers 多进程警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports as _orig_get_imports


def _patched_get_imports(filename):
    """Florence-2 的建模文件顶层 `import flash_attn` 会触发依赖检查报错。
    这里把它从依赖列表里剔除，实际并不会执行到（attn_implementation=eager）。"""
    imports = _orig_get_imports(filename)
    return [x for x in imports if x != "flash_attn"]


MODEL_ID_DEFAULT = "microsoft/Florence-2-large"

# 颜色表：不同类别用不同颜色
LABEL_COLORS = {
    "door": (255, 99, 71),     # tomato
    "window": (30, 144, 255),  # dodger blue
}
DEFAULT_COLOR = (50, 205, 50)  # limegreen


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, device: str):
    print(f"[load] model={model_id}  device={device}", flush=True)
    # MPS 在部分算子上还对 fp16 敏感，默认用 fp32；CUDA 用 fp16
    dtype = torch.float16 if device == "cuda" else torch.float32
    # 绕过 Florence-2 建模文件顶层 `import flash_attn` 的静态检查
    with patch("transformers.dynamic_module_utils.get_imports", _patched_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",  # 避免依赖 flash_attn
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, dtype


@torch.inference_mode()
def run_once(
    model,
    processor,
    image: Image.Image,
    device: str,
    dtype: torch.dtype,
    task_prompt: str,
    text_input: str,
) -> dict[str, Any]:
    prompt = task_prompt + text_input if text_input else task_prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # pixel_values 走浮点 dtype，input_ids 保持 long
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    return parsed


def extract_detections(parsed: dict, task_prompt: str) -> list[dict]:
    """
    Florence-2 结构示例 (OPEN_VOCABULARY_DETECTION):
        {"<OPEN_VOCABULARY_DETECTION>": {
            "bboxes": [[x1,y1,x2,y2], ...],
            "bboxes_labels": ["door", ...],
            "polygons": [...], "polygons_labels": [...]
        }}
    """
    payload = parsed.get(task_prompt, {})
    bboxes = payload.get("bboxes", []) or []
    labels = payload.get("bboxes_labels") or payload.get("labels") or []
    dets = []
    for i, bbox in enumerate(bboxes):
        label = labels[i] if i < len(labels) else "object"
        x1, y1, x2, y2 = [float(v) for v in bbox]
        dets.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "width": x2 - x1,
            "height": y2 - y1,
        })
    return dets


def detect_multi_class(
    model, processor, image, device, dtype,
    task_prompt: str, prompt: str,
) -> list[dict]:
    """
    对多类别 prompt 做检测。
    - OPEN_VOCABULARY_DETECTION: 每次只接受一个短语，按 '.' 拆分后对每类各调一次
    - CAPTION_TO_PHRASE_GROUNDING: 单次调用即可，从自然语言 caption 里 ground 所有名词短语
    """
    if task_prompt == "<OPEN_VOCABULARY_DETECTION>":
        classes = [c.strip() for c in prompt.split(".") if c.strip()]
        if not classes:
            classes = [prompt.strip()]
        all_dets: list[dict] = []
        for cls in classes:
            parsed = run_once(model, processor, image, device, dtype, task_prompt, cls)
            dets = extract_detections(parsed, task_prompt)
            # 某些版本 post_process 会把 label 留空/等于原查询，统一规范化
            for d in dets:
                if not d["label"] or d["label"].strip() == "":
                    d["label"] = cls
                else:
                    d["label"] = d["label"].strip().strip(".").lower() or cls
            all_dets.extend(dets)
        return all_dets

    # 短语定位：一次调用
    parsed = run_once(model, processor, image, device, dtype, task_prompt, prompt)
    dets = extract_detections(parsed, task_prompt)
    for d in dets:
        d["label"] = (d["label"] or "").strip().strip(".").lower() or "object"
    return dets


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    # macOS 上找一个常见中英字体作为标注字体
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_detections(image: Image.Image, dets: list[dict]) -> Image.Image:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    font = _load_font(max(14, vis.width // 80))
    for d in dets:
        label = d["label"].lower().strip(".")
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = label
        # 背景块 + 文字，便于阅读
        try:
            tb = draw.textbbox((x1, max(0, y1 - 20)), text, font=font)
        except Exception:
            tb = (x1, max(0, y1 - 20), x1 + 8 * len(text), max(0, y1 - 20) + 18)
        draw.rectangle(tb, fill=color)
        draw.text((tb[0] + 2, tb[1]), text, fill=(255, 255, 255), font=font)
    return vis


def iter_inputs(input_path: Path, glob_pat: str) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    if input_path.is_dir():
        for p in sorted(input_path.rglob(glob_pat)):
            if p.is_file():
                yield p
        return
    raise FileNotFoundError(f"input not found: {input_path}")


# --------------------- 全景图相关 ---------------------

def equirect_to_perspective(
    pano: Image.Image,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    out_size: int = 1024,
) -> tuple[Image.Image, dict]:
    """
    把 equirectangular 全景图中的一个视窗投影成透视图。
    返回 (透视图, 元数据)。元数据里含把透视图坐标反投影回全景坐标的工具信息。
    """
    W, H = pano.size
    pano_arr = np.asarray(pano)

    f = 0.5 * out_size / np.tan(np.deg2rad(fov_deg) / 2.0)
    # 透视图像素坐标
    i = np.arange(out_size, dtype=np.float32)
    j = np.arange(out_size, dtype=np.float32)
    ii, jj = np.meshgrid(i, j)  # x, y
    x = (ii - out_size / 2.0)
    y = (jj - out_size / 2.0)
    z = np.full_like(x, f)
    # 归一化到单位球
    norm = np.sqrt(x * x + y * y + z * z)
    xn = x / norm
    yn = y / norm
    zn = z / norm
    # 旋转：先绕 Y 轴 (yaw)，再绕 X 轴 (pitch)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # pitch then yaw
    xr = xn
    yr = cp * yn - sp * zn
    zr = sp * yn + cp * zn
    xw = cy * xr + sy * zr
    yw = yr
    zw = -sy * xr + cy * zr
    # 球坐标
    lon = np.arctan2(xw, zw)            # [-pi, pi]
    lat = np.arcsin(np.clip(yw, -1, 1))  # [-pi/2, pi/2]
    # equirectangular 采样坐标
    u = (lon / (2 * np.pi) + 0.5) * W
    v = (0.5 - lat / np.pi) * H
    u = np.clip(u, 0, W - 1).astype(np.int32)
    v = np.clip(v, 0, H - 1).astype(np.int32)
    persp_arr = pano_arr[v, u]
    persp = Image.fromarray(persp_arr)
    meta = {
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "fov_deg": fov_deg,
        "out_size": out_size,
        "pano_w": W,
        "pano_h": H,
        "focal": float(f),
    }
    return persp, meta


def _persp_pixel_to_latlon(x: float, y: float, meta: dict) -> tuple[float, float]:
    """把透视图像素 (x,y) 反投影到全景的 (lon, lat) 弧度。"""
    s = meta["out_size"]
    f = meta["focal"]
    xd = x - s / 2.0
    yd = y - s / 2.0
    zd = f
    n = np.sqrt(xd * xd + yd * yd + zd * zd)
    xn, yn, zn = xd / n, yd / n, zd / n
    yaw = np.deg2rad(meta["yaw_deg"])
    pitch = np.deg2rad(meta["pitch_deg"])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    xr = xn
    yr = cp * yn - sp * zn
    zr = sp * yn + cp * zn
    xw = cy * xr + sy * zr
    yw = yr
    zw = -sy * xr + cy * zr
    lon = np.arctan2(xw, zw)
    lat = np.arcsin(np.clip(yw, -1, 1))
    return float(lon), float(lat)


def persp_bbox_to_pano(bbox: list[float], meta: dict) -> list[float]:
    """
    透视图 bbox 映射回 equirectangular 坐标（矩形近似）。
    采样 bbox 四角和中点得到全景 u 的 min/max、v 的 min/max。
    处理跨左右接缝（lon ~ ±π）的情况：如果跨缝，返回两个框会复杂，这里保留主体框。
    """
    x1, y1, x2, y2 = bbox
    W, H = meta["pano_w"], meta["pano_h"]
    sample_xy = [
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
        ((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2),
        (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
    ]
    lons, lats = [], []
    for px, py in sample_xy:
        lon, lat = _persp_pixel_to_latlon(px, py, meta)
        lons.append(lon)
        lats.append(lat)
    lons = np.array(lons)
    lats = np.array(lats)
    # 处理跨接缝：把 lon 平移到同一段
    if lons.max() - lons.min() > np.pi:
        lons = np.where(lons < 0, lons + 2 * np.pi, lons)
    u = (lons / (2 * np.pi) + 0.5) * W
    v = (0.5 - lats / np.pi) * H
    u_min = float(np.clip(u.min(), 0, W - 1))
    u_max = float(np.clip(u.max(), 0, W - 1))
    v_min = float(np.clip(v.min(), 0, H - 1))
    v_max = float(np.clip(v.max(), 0, H - 1))
    # 如果平移过，u 可能 > W，折回
    if u_max >= W:
        u_min = u_min % W
        u_max = u_max % W
    return [u_min, v_min, u_max, v_max]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    uni = area_a + area_b - inter
    return inter / uni if uni > 0 else 0.0


def bbox_containment(a: list[float], b: list[float]) -> float:
    """min-IoU / 包含度：返回 inter / min(area_a, area_b)。
    1.0 表示其中一个框完全在另一个框里面。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    m = min(area_a, area_b)
    return inter / m if m > 0 else 0.0


def bbox_iou_wrap(a: list[float], b: list[float], pano_w: int) -> float:
    """考虑左右接缝的 IoU：把其中一个框整体 +pano_w 再算一次，取最大。"""
    base = bbox_iou(a, b)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    shifted_a = [ax1 + pano_w, ay1, ax2 + pano_w, ay2]
    shifted_b = [bx1 + pano_w, by1, bx2 + pano_w, by2]
    return max(base, bbox_iou(shifted_a, b), bbox_iou(a, shifted_b),
               bbox_iou(shifted_a, shifted_b))


def nms_by_label(dets: list[dict], iou_thr: float = 0.4,
                 pano_w: int | None = None) -> list[dict]:
    kept: list[dict] = []
    # 按面积从大到小保留
    dets_sorted = sorted(
        dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        reverse=True,
    )
    iou_fn = (lambda x, y: bbox_iou_wrap(x, y, pano_w)) if pano_w else bbox_iou
    for d in dets_sorted:
        dup = False
        for k in kept:
            if k["label"] != d["label"]:
                continue
            if iou_fn(d["bbox"], k["bbox"]) > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def nms_cross_label(dets: list[dict], iou_thr: float = 0.7,
                    pano_w: int | None = None,
                    containment_thr: float = 0.7) -> list[dict]:
    """跨类别抑制：同一块区域若被打上多个类别标签，保留面积占优势的那个。

    使用两个判据，任一触发即抑制：
    - IoU > iou_thr：两框基本重合
    - containment > containment_thr：小框基本被包含在大框里（解决"door 框里又给个 window 框"）
    """
    dets_sorted = sorted(
        dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        reverse=True,
    )
    iou_fn = (lambda x, y: bbox_iou_wrap(x, y, pano_w)) if pano_w else bbox_iou
    kept: list[dict] = []
    for d in dets_sorted:
        drop = False
        for k in kept:
            if iou_fn(d["bbox"], k["bbox"]) > iou_thr:
                drop = True
                break
            if bbox_containment(d["bbox"], k["bbox"]) > containment_thr:
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


def filter_detections(
    dets: list[dict], img_w: int, img_h: int,
    min_area_frac: float = 0.003, max_area_frac: float = 0.8,
    edge_thickness_frac: float = 0.02,
) -> list[dict]:
    """过滤明显是幻觉的框：
       - 面积 < min_area_frac 或 > max_area_frac 的整图面积
       - 紧贴图像边界的极薄条（整框 <edge_thickness 宽或高）
    """
    kept = []
    img_area = img_w * img_h
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        frac = (w * h) / img_area
        if frac < min_area_frac or frac > max_area_frac:
            continue
        # 极薄条且贴边
        thin = (w < img_w * edge_thickness_frac) or (h < img_h * edge_thickness_frac)
        hugs_edge = (x1 < img_w * edge_thickness_frac) or (x2 > img_w * (1 - edge_thickness_frac)) \
                    or (y1 < img_h * edge_thickness_frac) or (y2 > img_h * (1 - edge_thickness_frac))
        if thin and hugs_edge:
            continue
        kept.append(d)
    return kept


def main():
    ap = argparse.ArgumentParser(description="Florence-2 门窗识别")
    ap.add_argument("--input", required=True, help="图片文件或目录")
    ap.add_argument("--output", default=str(Path(__file__).parent / "out"),
                    help="输出目录，默认 2d/out")
    ap.add_argument("--glob", default="*.jpg", help="目录模式下的匹配，默认 *.jpg")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少张（目录模式）")
    ap.add_argument("--model", default=MODEL_ID_DEFAULT)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--prompt", default="door. window.",
                    help="开放词汇检测提示词，用 '.' 分隔多个类别")
    ap.add_argument("--task", default="<OPEN_VOCABULARY_DETECTION>",
                    choices=[
                        "<OPEN_VOCABULARY_DETECTION>",
                        "<CAPTION_TO_PHRASE_GROUNDING>",
                    ])
    ap.add_argument("--pano-split", type=int, default=0,
                    help="把 equirectangular 全景图水平切 N 张透视图分别识别 (建议 4~8)。"
                         "0 表示不切分。")
    ap.add_argument("--pano-fov", type=float, default=90.0,
                    help="--pano-split 的每张透视图 FoV (度)，默认 90")
    ap.add_argument("--pano-out-size", type=int, default=1024,
                    help="--pano-split 每张透视图的输出分辨率 (像素)，默认 1024")
    ap.add_argument("--auto-pano", action="store_true",
                    help="输入图若为 2:1 宽高比则自动启用 --pano-split (默认 6)")
    ap.add_argument("--nms-iou", type=float, default=0.4,
                    help="跨视图合并时的 NMS IoU 阈值")
    ap.add_argument("--cross-label-iou", type=float, default=0.5,
                    help="跨类别抑制的 IoU 阈值，超过则保留面积较大的那个")
    ap.add_argument("--cross-label-containment", type=float, default=0.7,
                    help="跨类别抑制的包含度阈值。小框被大框包含超过该比例则抑制小框")
    ap.add_argument("--min-area", type=float, default=0.003,
                    help="过滤掉 bbox 面积小于该比例(相对全图)的检测，默认 0.003")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model, processor, dtype = load_model(args.model, device)

    summary_path = out_dir / "detections.jsonl"
    summary_f = summary_path.open("w", encoding="utf-8")

    files = list(iter_inputs(input_path, args.glob))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"[warn] no files matched under {input_path}")
        return

    print(f"[run] {len(files)} images, task={args.task}, prompt={args.prompt!r}")
    t_all0 = time.time()
    total_hits = 0
    for i, img_path in enumerate(files, 1):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[skip] {img_path.name}: open failed -> {e}")
            continue

        t0 = time.time()
        # 决定是否走全景切分
        do_pano = args.pano_split > 0
        if args.auto_pano and not do_pano and abs(image.width - 2 * image.height) < 4:
            do_pano = True
            n_split = 6
        else:
            n_split = args.pano_split

        if do_pano:
            dets: list[dict] = []
            yaw_step = 360.0 / n_split
            for k in range(n_split):
                yaw = -180.0 + k * yaw_step + yaw_step / 2.0  # 中心化
                persp, meta = equirect_to_perspective(
                    image, yaw_deg=yaw, pitch_deg=0.0,
                    fov_deg=args.pano_fov, out_size=args.pano_out_size,
                )
                sub_dets = detect_multi_class(model, processor, persp,
                                              device, dtype, args.task, args.prompt)
                sub_dets = filter_detections(
                    sub_dets, persp.width, persp.height,
                    min_area_frac=args.min_area,
                )
                # 映射回全景坐标
                for d in sub_dets:
                    pano_bbox = persp_bbox_to_pano(d["bbox"], meta)
                    x1, y1, x2, y2 = pano_bbox
                    d["bbox"] = pano_bbox
                    d["width"] = x2 - x1
                    d["height"] = y2 - y1
                    d["source_yaw"] = round(yaw, 1)
                dets.extend(sub_dets)
            # 跨视图 NMS 去重（考虑全景左右接缝）
            dets = nms_by_label(dets, iou_thr=args.nms_iou, pano_w=image.width)
            # 同一块区域被多个类别同时标注的情况，跨类别再抑制
            dets = nms_cross_label(dets, iou_thr=args.cross_label_iou,
                                   pano_w=image.width,
                                   containment_thr=args.cross_label_containment)
            # 映射回全景后，再用全景分辨率做一次面积过滤（剔除投影后变得极小的框）
            dets = filter_detections(
                dets, image.width, image.height,
                min_area_frac=args.min_area,
            )
        else:
            dets = detect_multi_class(model, processor, image, device, dtype,
                                      args.task, args.prompt)
            dets = filter_detections(
                dets, image.width, image.height, min_area_frac=args.min_area,
            )
        dt = time.time() - t0
        total_hits += len(dets)

        # 保存可视化图
        vis = draw_detections(image, dets)
        vis_path = out_dir / f"{img_path.stem}_annot.jpg"
        vis.save(vis_path, quality=92)

        # 保存单图 json
        per_json = {
            "image": str(img_path),
            "width": image.width,
            "height": image.height,
            "task": args.task,
            "prompt": args.prompt,
            "pano_split": n_split if do_pano else 0,
            "detections": dets,
            "latency_sec": round(dt, 3),
        }
        (out_dir / f"{img_path.stem}.json").write_text(
            json.dumps(per_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_f.write(json.dumps(per_json, ensure_ascii=False) + "\n")

        counts = {}
        for d in dets:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        tag = ", ".join(f"{k}:{v}" for k, v in counts.items()) or "none"
        print(f"[{i}/{len(files)}] {img_path.name}  dets={len(dets)} ({tag})  {dt:.2f}s")

    summary_f.close()
    dt_all = time.time() - t_all0
    print(f"\n[done] {len(files)} images, total detections={total_hits}, "
          f"elapsed={dt_all:.1f}s, avg={dt_all/max(1,len(files)):.2f}s/img")
    print(f"[out]  {out_dir}")
    print(f"[out]  summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] interrupted", file=sys.stderr)
        sys.exit(130)
