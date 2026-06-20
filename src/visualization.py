import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for both CLI and Gradio

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


def draw_mask_overlay(
    image,
    mask,
    color=(0, 255, 100),
    alpha=0.45,
):
    """
    Blend a colored mask overlay onto the image.
    """

    overlay = image.copy().astype(np.float32)

    for c, val in enumerate(color):
        overlay[:, :, c] = np.where(
            mask,
            overlay[:, :, c] * (1 - alpha) + val * alpha,
            overlay[:, :, c],
        )

    return overlay.astype(np.uint8)


def create_visualization(
    image,
    masks,
    best_idx,
    prompt=None,
    best_score=None,
):
    """
    Create a side-by-side visualization:
    left = original image, right = best mask overlay.
    """

    best_mask = masks[best_idx]["segmentation"]
    overlaid = draw_mask_overlay(image, best_mask)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    axes[0].imshow(image)
    axes[0].set_title("Original Image", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    axes[1].imshow(overlaid)

    title = "Best Matching Mask"
    if prompt:
        title += f'  |  prompt: "{prompt}"'
    if best_score is not None:
        title += f"  |  confidence: {best_score:.1%}"

    axes[1].set_title(title, color="white", fontsize=12, pad=10)
    axes[1].axis("off")

    plt.tight_layout(pad=1.5)

    fig.canvas.draw()
    buf = fig.canvas.renderer.buffer_rgba()
    result = np.asarray(buf)[:, :, :3]

    plt.close(fig)

    return result


def create_topk_visualization(
    image,
    masks,
    all_scores,
    prompt=None,
    k=3,
    min_confidence=0.10,
    min_gap_ratio=0.25,
):
    """
    Create a visualization showing the original image plus the top
    candidate masks side by side, each labeled with its confidence.

    Unlike a naive top-k that always shows k results regardless of how
    weak they are, this applies two quality filters before including a
    candidate beyond the #1 match:

        1. min_confidence  : a candidate's own score must clear this
                              floor (default 10%) -- a 3% match is noise,
                              not a genuine alternative interpretation.
        2. min_gap_ratio    : a candidate's score must be at least this
                              fraction of the #1 score (default 25%) --
                              if the top match is decisively ahead of
                              everything else, showing "close competitors"
                              that are actually nowhere close is misleading.

    If the #1 match itself is weak (below min_confidence), the result
    is still shown, but flagged so the caller can warn the user that
    the model found no strong match for this prompt in this image at all
    -- this is a more honest outcome than silently presenting a
    low-confidence guess as if it were reliable.

    Returns:
        tuple: (result_image, is_low_confidence, num_shown)
    """

    if not all_scores:
        return image, True, 0

    top_score = all_scores[0][1]
    is_low_confidence = top_score < min_confidence

    qualified = [all_scores[0]]  # always show the #1 match

    for mask_idx, score in all_scores[1:k]:
        if score < min_confidence:
            continue
        if score < top_score * min_gap_ratio:
            continue
        qualified.append((mask_idx, score))

    n_panels = 1 + len(qualified)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    fig.patch.set_facecolor("#1a1a2e")

    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(image)
    axes[0].set_title("Original Image", color="white", fontsize=12, pad=10)
    axes[0].axis("off")

    for panel_idx, (mask_idx, score) in enumerate(qualified, start=1):
        mask = masks[mask_idx]["segmentation"]
        overlaid = draw_mask_overlay(image, mask)

        axes[panel_idx].imshow(overlaid)

        rank_label = f"#{panel_idx} match"
        title = f"{rank_label}  |  {score:.1%}"
        axes[panel_idx].set_title(title, color="white", fontsize=12, pad=10)
        axes[panel_idx].axis("off")

    suptitle = "Top Matches"
    if prompt:
        suptitle += f'  for prompt: "{prompt}"'
    if is_low_confidence:
        suptitle += "  (low confidence overall)"
    fig.suptitle(suptitle, color="white", fontsize=13, y=1.02)

    plt.tight_layout(pad=1.5)

    fig.canvas.draw()
    buf = fig.canvas.renderer.buffer_rgba()
    result = np.asarray(buf)[:, :, :3]

    plt.close(fig)

    return result, is_low_confidence, len(qualified)


def save_visualization(
    image,
    masks,
    best_idx,
    output_path,
    prompt=None,
    best_score=None,
):
    """
    Create and save visualization to disk as PNG.
    """

    result = create_visualization(
        image=image,
        masks=masks,
        best_idx=best_idx,
        prompt=prompt,
        best_score=best_score,
    )

    Image.fromarray(result).save(output_path)
    print(f"Saved visualization → {output_path}")