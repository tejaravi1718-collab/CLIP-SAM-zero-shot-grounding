from src.clip_utils import (
    score_mask_contrastive,
)


def find_best_mask(
    image,
    masks,
    prompt,
    model,
    processor,
    device="cpu",
):
    """
    Score all masks against a text prompt using contrastive CLIP scoring
    and return the best matching mask.

    Each mask is scored as a softmax probability against distractor
    prompts ("a part of X", "background", "blurry crop"), which lets
    CLIP itself disambiguate full objects from partial crops -- no
    artificial area-based weighting is applied on top.

    Args:
        image      : numpy array (H, W, 3) in RGB
        masks      : list of SAM mask dicts (already filtered)
        prompt     : text prompt string
        model      : CLIP model
        processor  : CLIP processor
        device     : torch device string

    Returns:
        tuple: (best_idx, best_score, all_scores)
            best_idx   : index into masks list of the best mask
            best_score : contrastive probability score of best mask (0-1)
            all_scores : list of (idx, score) for all masks, sorted descending
    """

    best_score = -float("inf")
    best_idx = None
    all_scores = []

    for idx, mask_info in enumerate(masks):

        score = score_mask_contrastive(
            image=image,
            mask_info=mask_info,
            prompt=prompt,
            model=model,
            processor=processor,
            device=device,
        )

        all_scores.append((idx, score))

        if score > best_score:
            best_score = score
            best_idx = idx

    all_scores_sorted = sorted(
        all_scores,
        key=lambda x: x[1],
        reverse=True,
    )

    return best_idx, best_score, all_scores_sorted