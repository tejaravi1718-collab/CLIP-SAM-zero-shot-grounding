from PIL import Image
import torch
import numpy as np

from transformers import (
    CLIPModel,
    CLIPProcessor,
)


def load_clip_model(device="cpu"):
    """Load CLIP model and processor."""

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    model.to(device)
    model.eval()

    return model, processor


def crop_with_padding(image, bbox, padding_ratio=0.10):
    """
    Crop an image region with extra padding around the bbox so the
    object isn't touching the crop edges. CLIP performs better on
    natural-looking crops with breathing room.

    Args:
        image         : numpy array (H, W, 3)
        bbox          : (x, y, w, h) from SAM mask_info["bbox"]
        padding_ratio : fraction of bbox size to pad on each side

    Returns:
        numpy array: padded crop
    """

    img_h, img_w = image.shape[:2]
    x, y, w, h = [int(v) for v in bbox]

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return image[y1:y2, x1:x2]


def score_mask_contrastive(
    image,
    mask_info,
    prompt,
    model,
    processor,
    device="cpu",
):
    """
    Score a mask using contrastive CLIP zero-shot classification.

    Instead of measuring raw similarity to the target prompt alone
    (which biases toward tight, clean crops of recognizable parts --
    e.g. a car door scoring higher than the full car), this computes
    a softmax probability against explicit distractor prompts:

        - "a part of a {object}"      -> penalizes partial crops
        - "a photo of the background" -> penalizes background/scenery
        - "a blurry, cropped photo"   -> penalizes low-quality crops

    This forces CLIP to make a relative judgment: "is this crop MORE
    like the whole object than a part of it?" -- which is a much more
    natural way to disambiguate full-object masks from partial ones.

    Args:
        image      : numpy array (H, W, 3), RGB
        mask_info  : single SAM mask dict
        prompt     : text prompt string (e.g. "car")
        model      : CLIP model
        processor  : CLIP processor
        device     : torch device string

    Returns:
        float: softmax probability (0-1) that this crop is the full target object
    """

    base = prompt.strip().lower()

    candidate_prompts = [
        f"a photo of a whole {base}",   # target: complete object
        f"a part of a {base}",           # distractor: partial object
        "a photo of the background",     # distractor: background/scenery
        "a blurry, cropped photo",       # distractor: low-quality crop
    ]

    cropped = crop_with_padding(
        image=image,
        bbox=mask_info["bbox"],
        padding_ratio=0.10,
    )

    if cropped.size == 0:
        return 0.0

    crop_pil = Image.fromarray(cropped)

    inputs = processor(
        text=candidate_prompts,
        images=crop_pil,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Softmax across the 4 candidate prompts -- index 0 is our target
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    target_prob = probs[0].item()

    return target_prob


def score_mask_bbox(
    image,
    mask_info,
    prompts,
    model,
    processor,
    device="cpu",
):
    """
    Simple bbox-crop scoring against one or more prompts (no distractors).
    Kept for comparison/debugging -- score_mask_contrastive is preferred
    for actual ranking.

    Args:
        image      : numpy array (H, W, 3)
        mask_info  : single SAM mask dict
        prompts    : str or list[str]
        model      : CLIP model
        processor  : CLIP processor
        device     : torch device string

    Returns:
        float: averaged raw CLIP similarity score across all prompts
    """

    if isinstance(prompts, str):
        prompts = [prompts]

    cropped = crop_with_padding(
        image=image,
        bbox=mask_info["bbox"],
        padding_ratio=0.10,
    )

    if cropped.size == 0:
        return -float("inf")

    crop_pil = Image.fromarray(cropped)

    inputs = processor(
        text=prompts,
        images=crop_pil,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits_per_image[0]
    avg_score = scores.mean().item()

    return avg_score