"""
门窗识别脚本 v2 —— 基于 google/owlv2-base-patch16-ensemble (默认)

OWLv2 / OWL-ViT 都是真正的开放词汇检测器：
  * 输出 confidence score，可设阈值
  * 单次前向同时处理多个类别，不会"门窗互锁"
  * 没目标就返 0 框

用法示例:
    # 全景图 (5760x2880 equirectangular)
    python detect_owlvit.py --input ../20260129135824/CAM/592.692836_IMG.jpg --auto-pano

    # 批量
    python detect_owlvit.py --input ../20260129135824/CAM --limit 20 --auto-pano

    # 调阈值
    python detect_owlvit.py --input xxx.jpg --auto-pano --score-thr 0.10

    # 多 prompt 同义词提升召回 (用 | 分隔同义词)
    python detect_owlvit.py --input xxx.jpg --auto-pano \\
        --classes 'door|a door|a wooden door|a glass door,window|a window|a glass window'

    # 用 OWL-ViT v1 (更快，但精度差很多)
    python detect_owlvit.py --model google/owlvit-base-patch32 --input xxx.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    OwlViTProcessor, OwlViTForObjectDetection,
    Owlv2Processor, Owlv2ForObjectDetection,
)

# 复用 Florence 脚本里写好的全景 / NMS / 可视化工具
from detect_doorwindow import (
    equirect_to_perspective, persp_bbox_to_pano,
    bbox_iou, bbox_iou_wrap, bbox_containment,
    nms_by_label, nms_cross_label, filter_detections,
    draw_detections, iter_inputs, pick_device,
)

MODEL_ID_DEFAULT = "google/owlv2-base-patch16-ensemble"


def load_model(model_id: str, device: str):
    print(f"[load] model={model_id}  device={device}", flush=True)
    if "owlv2" in model_id.lower():
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    else:
        processor = OwlViTProcessor.from_pretrained(model_id)
        model = OwlViTForObjectDetection.from_pretrained(model_id).to(device).eval()
    return model, processor


@torch.inference_mode()
def detect_owlvit(
    model, processor, image: Image.Image, device: str,
    classes: list[str], score_thr: float = 0.15,
    class_prompts: list[list[str]] | None = None,
) -> list[dict]:
    """单张图、多类别一次推理。
    class_prompts: 每类对应的多个等价 prompt（如 ['door', 'a door', 'a wooden door']），
    检测时把它们都喂给模型，回收时按所属类别合并。
    """
    if class_prompts is None:
        class_prompts = [[c] for c in classes]
    flat_prompts: list[str] = []
    prompt_to_class: list[int] = []
    for ci, prompts in enumerate(class_prompts):
        for p in prompts:
            flat_prompts.append(p)
            prompt_to_class.append(ci)
    queries = [flat_prompts]  # 一个 batch
    inputs = processor(text=queries, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    target_sizes = torch.tensor([(image.height, image.width)], device=device)
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=score_thr,
    )[0]
    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["labels"].cpu().tolist()
    dets = []
    for b, s, lid in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in b]
        ci = prompt_to_class[lid]
        dets.append({
            "label": classes[ci],
            "matched_prompt": flat_prompts[lid],
            "score": round(float(s), 4),
            "bbox": [x1, y1, x2, y2],
            "width": x2 - x1,
            "height": y2 - y1,
        })
    return dets


def nms_by_label_with_score(dets: list[dict], iou_thr: float, pano_w: int | None) -> list[dict]:
    """按 label 分组，按 score 降序保留"""
    dets_sorted = sorted(dets, key=lambda d: d.get("score", 0.0), reverse=True)
    iou_fn = (lambda x, y: bbox_iou_wrap(x, y, pano_w)) if pano_w else bbox_iou
    kept: list[dict] = []
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


def main():
    ap = argparse.ArgumentParser(description="OWL-ViT 门窗识别")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID_DEFAULT)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--classes", default="door,window",
                    help="逗号分隔的类别列表。可在每类后用 |分隔写多个 prompt 同义词，"
                         "如 'door|a door|a wooden door,window|a window|a glass window'")
    ap.add_argument("--exclude", default="",
                    help="干扰类别（不输出，仅用于跟正类竞争分数，避免误判）。"
                         "用 ',' 分隔类别，'|' 分隔同义词。"
                         "例：'bookshelf|a bookshelf,monitor|a computer monitor'")
    ap.add_argument("--score-thr", type=float, default=0.15,
                    help="OWL-ViT 输出分数阈值，默认 0.15")
    ap.add_argument("--pano-split", type=int, default=0)
    ap.add_argument("--pano-fov", type=float, default=90.0)
    ap.add_argument("--pano-out-size", type=int, default=1024)
    ap.add_argument("--auto-pano", action="store_true",
                    help="2:1 全景图自动切 6 张透视图")
    ap.add_argument("--nms-iou", type=float, default=0.4)
    ap.add_argument("--cross-label-iou", type=float, default=0.5)
    ap.add_argument("--cross-label-containment", type=float, default=0.7)
    ap.add_argument("--min-area", type=float, default=0.002)
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    classes_raw = [c.strip() for c in args.classes.split(",") if c.strip()]
    classes: list[str] = []
    class_prompts: list[list[str]] = []
    for c in classes_raw:
        parts = [p.strip() for p in c.split("|") if p.strip()]
        classes.append(parts[0])
        class_prompts.append(parts)

    # 干扰类别：和正类一起打分，但最终结果里过滤掉
    exclude_raw = [c.strip() for c in args.exclude.split(",") if c.strip()]
    exclude_classes: list[str] = []
    for c in exclude_raw:
        parts = [p.strip() for p in c.split("|") if p.strip()]
        exclude_classes.append(parts[0])
        classes.append(parts[0])
        class_prompts.append(parts)
    exclude_set = set(exclude_classes)
    if exclude_classes:
        print(f"[exclude] {exclude_classes}")
    device = pick_device(args.device)
    model, processor = load_model(args.model, device)

    files = list(iter_inputs(input_path, args.glob))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"[warn] no files matched under {input_path}")
        return

    print(f"[run] {len(files)} images, classes={classes}, score_thr={args.score_thr}")
    summary_path = out_dir / "detections.jsonl"
    summary_f = summary_path.open("w", encoding="utf-8")

    t_all0 = time.time()
    total_hits = 0
    for i, img_path in enumerate(files, 1):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[skip] {img_path.name}: {e}")
            continue

        t0 = time.time()
        do_pano = args.pano_split > 0
        n_split = args.pano_split
        if args.auto_pano and not do_pano and abs(image.width - 2 * image.height) < 4:
            do_pano = True
            n_split = 6

        if do_pano:
            dets: list[dict] = []
            yaw_step = 360.0 / n_split
            for k in range(n_split):
                yaw = -180.0 + k * yaw_step + yaw_step / 2.0
                persp, meta = equirect_to_perspective(
                    image, yaw_deg=yaw, pitch_deg=0.0,
                    fov_deg=args.pano_fov, out_size=args.pano_out_size,
                )
                sub = detect_owlvit(model, processor, persp, device,
                                    classes, score_thr=args.score_thr,
                                    class_prompts=class_prompts)
                sub = filter_detections(sub, persp.width, persp.height,
                                        min_area_frac=args.min_area)
                for d in sub:
                    pb = persp_bbox_to_pano(d["bbox"], meta)
                    d["bbox"] = pb
                    d["width"] = pb[2] - pb[0]
                    d["height"] = pb[3] - pb[1]
                    d["source_yaw"] = round(yaw, 1)
                dets.extend(sub)
            dets = nms_by_label_with_score(dets, iou_thr=args.nms_iou,
                                           pano_w=image.width)
            dets = nms_cross_label(dets, iou_thr=args.cross_label_iou,
                                   pano_w=image.width,
                                   containment_thr=args.cross_label_containment)
            dets = filter_detections(dets, image.width, image.height,
                                     min_area_frac=args.min_area)
        else:
            dets = detect_owlvit(model, processor, image, device,
                                 classes, score_thr=args.score_thr,
                                 class_prompts=class_prompts)
            dets = filter_detections(dets, image.width, image.height,
                                     min_area_frac=args.min_area)
            dets = nms_by_label_with_score(dets, iou_thr=args.nms_iou, pano_w=None)
            dets = nms_cross_label(dets, iou_thr=args.cross_label_iou,
                                   containment_thr=args.cross_label_containment)
        # 过滤干扰类别（只保留 --classes 里声明的目标类）
        if exclude_set:
            dets = [d for d in dets if d["label"] not in exclude_set]
        dt = time.time() - t0
        total_hits += len(dets)

        vis = draw_detections(image, dets)
        vis.save(out_dir / f"{img_path.stem}_annot.jpg", quality=92)
        per_json = {
            "image": str(img_path),
            "width": image.width, "height": image.height,
            "model": args.model, "classes": classes,
            "score_thr": args.score_thr,
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort]", file=sys.stderr); sys.exit(130)
