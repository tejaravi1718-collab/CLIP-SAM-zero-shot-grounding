import time

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)


def load_sam_model(
    checkpoint_path,
    model_type="vit_b",
    device="cpu",
):
    """Load SAM model from checkpoint."""

    sam = sam_model_registry[model_type](
        checkpoint=checkpoint_path
    )
    sam.to(device)

    return sam


def generate_masks(
    image,
    sam,
    points_per_side=20,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
):
    """
    Generate all masks for an image using SAM's automatic mask generator.

    Defaults match SAM's own recommended thresholds (strict) to keep
    mask boundaries clean and precise -- looser thresholds let "leaky"
    masks through that bleed into surrounding background, which can
    fool CLIP into scoring them confidently (the crop still contains
    the object, just with messy boundaries).

    For low-contrast images (e.g. a pet on a plain studio background)
    where strict thresholds return too few masks, callers should retry
    with a denser point grid AND relaxed thresholds as a fallback --
    see app.py / run_grounding.py for the fallback pattern. Loosening
    thresholds alone does not help in this case, because thresholds
    only filter already-proposed masks -- they don't create new
    proposals. A denser grid (more points_per_side) is what generates
    more distinct region hypotheses in the first place.

    Args:
        image                   : numpy array (H, W, 3) in RGB
        sam                     : loaded SAM model
        points_per_side         : grid density for mask generation
        pred_iou_thresh         : min predicted IoU to keep a mask (lower = more masks kept)
        stability_score_thresh  : min stability score to keep a mask (lower = more masks kept)

    Returns:
        list[dict]: raw SAM mask list
    """

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )

    print(
        f"Generating masks (points_per_side={points_per_side}, "
        f"iou_thresh={pred_iou_thresh}, stability_thresh={stability_score_thresh})..."
    )
    print(f"  [DEBUG] Input image shape to SAM: {image.shape}")
    print(f"  [DEBUG] Input image dtype: {image.dtype}")

    start = time.time()
    masks = mask_generator.generate(image)
    elapsed = time.time() - start

    print(f"Generated {len(masks)} masks in {elapsed:.2f}s")

    return masks


def filter_masks(
    masks,
    image_area,
    min_area_ratio=0.005,
    max_area_ratio=0.90,
):
    """
    Filter masks by relative area to remove noise (too small)
    and full-image masks (too large).

    Args:
        masks           : list of SAM mask dicts
        image_area      : total image pixel count (H * W)
        min_area_ratio  : minimum mask area as fraction of image (default 0.5%)
        max_area_ratio  : maximum mask area as fraction of image (default 90%)

    Returns:
        list[dict]: filtered mask list
    """

    min_area = min_area_ratio * image_area
    max_area = max_area_ratio * image_area

    filtered = [
        m for m in masks
        if min_area <= m["area"] <= max_area
    ]

    print(
        f"Filtered masks: {len(masks)} → {len(filtered)} "
        f"(removed {len(masks) - len(filtered)} out-of-range masks)"
    )

    return filtered