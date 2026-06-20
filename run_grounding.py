"""
run_grounding.py
----------------
CLI script to run the CLIP + SAM zero-shot grounding pipeline.

Usage:
    python run_grounding.py --image data/your_image.jpg --prompt "a car"

Arguments:
    --image         : path to input image
    --prompt        : text prompt to ground (e.g. "a dog")
    --checkpoint    : path to SAM checkpoint (default: checkpoints/sam_vit_b_01ec64.pth)
    --output        : output image path (default: outputs/result.png)
    --points        : SAM points_per_side (default: 20)
    --top3          : save top candidate matches instead of single best
                       (low-confidence/low-margin candidates are filtered out
                       automatically -- see README "Handling Ambiguous Prompts")
"""

import argparse
import os
import sys

import cv2
import numpy as np

from src.clip_utils import load_clip_model
from src.sam_utils import load_sam_model, generate_masks, filter_masks
from src.grounding import find_best_mask
from src.visualization import save_visualization, create_topk_visualization
from PIL import Image


LOW_CONFIDENCE_THRESHOLD = 0.10


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP + SAM Zero-Shot Grounding"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/54cae422a6fc6_-_american-muscle-facts-03-0312-xln.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="car",
        help="Text prompt to ground (e.g. 'a car')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam_vit_b_01ec64.pth",
        help="Path to SAM model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.png",
        help="Output image path",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=20,
        help="SAM points_per_side (higher = more masks, slower)",
    )
    parser.add_argument(
        "--top3",
        action="store_true",
        help="Save top candidate matches instead of just the single best match "
             "(useful for ambiguous prompts, e.g. 'white car' with multiple white cars). "
             "Weak/low-margin candidates are automatically excluded.",
    )
    return parser.parse_args()


def load_image_rgb(path):
    image_bgr = cv2.imread(path)

    if image_bgr is None:
        print(f"[ERROR] Could not load image: {path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]
    if w > 800:
        scale = 800 / w
        new_w = 800
        new_h = int(h * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h))

    return image_rgb


def ensure_output_dir(output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def main():
    args = parse_args()

    print("\n" + "=" * 50)
    print("  CLIP + SAM Zero-Shot Grounding")
    print("=" * 50)
    print(f"  Image    : {args.image}")
    print(f"  Prompt   : {args.prompt}")
    print(f"  Points   : {args.points}")
    print(f"  Output   : {args.output}")
    print("=" * 50 + "\n")

    print("[1/5] Loading CLIP...")
    model, processor = load_clip_model(device="cpu")
    print("      CLIP loaded ✓")

    print("[2/5] Loading SAM...")
    sam = load_sam_model(
        checkpoint_path=args.checkpoint,
        model_type="vit_b",
        device="cpu",
    )
    print("      SAM loaded ✓")

    print("[3/5] Loading image...")
    image = load_image_rgb(args.image)
    print(f"      Shape: {image.shape} ✓")

    print("[4/5] Generating masks...")
    masks = generate_masks(image=image, sam=sam, points_per_side=args.points)

    image_area = image.shape[0] * image.shape[1]
    filtered = filter_masks(masks=masks, image_area=image_area)

    if len(filtered) < 5:
        print("Few masks detected — retrying with more sensitive thresholds...")
        masks = generate_masks(
            image=image,
            sam=sam,
            points_per_side=32,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.85,
        )
        filtered = filter_masks(masks=masks, image_area=image_area)

    masks = filtered

    if len(masks) == 0:
        print("[ERROR] No masks survived filtering. Try adjusting min/max area ratios.")
        sys.exit(1)

    print(f"[5/5] Grounding prompt: '{args.prompt}'...")
    best_idx, best_score, all_scores = find_best_mask(
        image=image,
        masks=masks,
        prompt=args.prompt,
        model=model,
        processor=processor,
        device="cpu",
    )

    print("\n" + "─" * 50)
    print("  RESULTS")
    print("─" * 50)
    print(f"  Best mask index : {best_idx}")
    print(f"  Confidence      : {best_score:.1%}")
    print(f"  Mask area       : {masks[best_idx]['area']} px")
    print(f"  Area ratio      : {masks[best_idx]['area'] / image_area:.2%}")

    if best_score < LOW_CONFIDENCE_THRESHOLD:
        print(
            f"\n  ⚠️  LOW CONFIDENCE: best match is only {best_score:.1%}. "
            f"The model did not find a strong match for '{args.prompt}' "
            f"in this image. Consider a different or more specific prompt."
        )

    print("\n  Top 10 Masks:")
    print(f"  {'Rank':<6} {'Idx':<6} {'Confidence':<12} {'Area':>8}")
    print("  " + "-" * 40)

    for rank, (idx, score) in enumerate(all_scores[:10], 1):
        area = masks[idx]["area"]
        print(f"  {rank:<6} {idx:<6} {score:<12.1%} {area:>8}")

    ensure_output_dir(args.output)

    if args.top3:
        result, is_low_confidence, num_shown = create_topk_visualization(
            image=image,
            masks=masks,
            all_scores=all_scores,
            prompt=args.prompt,
            k=3,
        )
        Image.fromarray(result).save(args.output)
        print(f"\nSaved top-{num_shown} visualization → {args.output}")
        if is_low_confidence:
            print("  (flagged low-confidence in the figure title)")
    else:
        save_visualization(
            image=image,
            masks=masks,
            best_idx=best_idx,
            output_path=args.output,
            prompt=args.prompt,
            best_score=best_score,
        )

    print("\n  Done. ✓")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()