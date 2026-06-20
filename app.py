"""
app.py
------
Gradio web app for CLIP + SAM Zero-Shot Grounding.

Run with:
    python app.py
"""

import os
import sys

import cv2
import numpy as np
import gradio as gr

from src.clip_utils import load_clip_model
from src.sam_utils import load_sam_model, generate_masks, filter_masks
from src.grounding import find_best_mask
from src.visualization import create_visualization, create_topk_visualization


SAM_CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
DEVICE = "cpu"
MAX_IMAGE_WIDTH = 800
POINTS_PER_SIDE = 20
LOW_CONFIDENCE_THRESHOLD = 0.10


print("Loading CLIP model...")
clip_model, clip_processor = load_clip_model(device=DEVICE)
print("CLIP loaded OK")

print("Loading SAM model...")
sam = load_sam_model(
    checkpoint_path=SAM_CHECKPOINT,
    model_type=SAM_MODEL_TYPE,
    device=DEVICE,
)
print("SAM loaded OK")
print("Models ready. Starting app...\n")


def resize_preserve_aspect(image, max_width=MAX_IMAGE_WIDTH):
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))


def ground_object(image, prompt, show_top3=False, progress=gr.Progress()):
    if image is None:
        return None, "Please upload an image."
    if not prompt or not prompt.strip():
        return None, "Please enter a text prompt."

    prompt = prompt.strip()

    try:
        progress(0.05, desc="Preparing image...")
        image = resize_preserve_aspect(image, max_width=MAX_IMAGE_WIDTH)
        image_area = image.shape[0] * image.shape[1]
        print(f"[DEBUG] Image shape after resize: {image.shape}")

        progress(0.15, desc="Segmenting image with SAM (this is the slow step)...")
        masks = generate_masks(
            image=image,
            sam=sam,
            points_per_side=POINTS_PER_SIDE,
        )

        progress(0.7, desc="Filtering candidate regions...")
        filtered = filter_masks(
            masks=masks,
            image_area=image_area,
            min_area_ratio=0.005,
            max_area_ratio=0.90,
        )

        if len(filtered) < 5:
            print("Few masks detected -- retrying with more sensitive thresholds...")
            progress(0.72, desc="Few regions found -- re-segmenting with finer detail...")
            masks = generate_masks(
                image=image,
                sam=sam,
                points_per_side=32,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.85,
            )
            filtered = filter_masks(
                masks=masks,
                image_area=image_area,
                min_area_ratio=0.005,
                max_area_ratio=0.90,
            )

        masks = filtered

        if len(masks) == 0:
            return None, "No valid masks found. Try a different image."

        progress(0.85, desc=f"Matching '{prompt}' against {len(masks)} regions with CLIP...")
        best_idx, best_score, all_scores = find_best_mask(
            image=image,
            masks=masks,
            prompt=prompt,
            model=clip_model,
            processor=clip_processor,
            device=DEVICE,
        )

        progress(0.97, desc="Rendering result...")

        low_conf_note = ""
        if best_score < LOW_CONFIDENCE_THRESHOLD:
            low_conf_note = (
                f"\n⚠️ Low confidence overall ({best_score:.1%}) — the model "
                f"didn't find a strong match for '{prompt}' in this image. "
                f"Try a more specific or different prompt."
            )

        if show_top3:
            result, is_low_confidence, num_shown = create_topk_visualization(
                image=image,
                masks=masks,
                all_scores=all_scores,
                prompt=prompt,
                k=3,
            )
            top3_lines = "\n".join(
                f"  #{i+1}: {score:.1%}"
                for i, (idx, score) in enumerate(all_scores[:num_shown])
            )
            status = (
                f"Showing {num_shown} candidate(s) for: '{prompt}'\n"
                f"Masks generated: {len(masks)}\n"
                f"Scores:\n{top3_lines}"
                f"{low_conf_note}"
            )
        else:
            result = create_visualization(
                image=image,
                masks=masks,
                best_idx=best_idx,
                prompt=prompt,
                best_score=best_score,
            )
            area_ratio = masks[best_idx]["area"] / image_area
            status = (
                f"Found: '{prompt}'\n"
                f"Masks generated: {len(masks)}\n"
                f"Confidence: {best_score:.1%}\n"
                f"Mask area: {area_ratio:.1%} of image"
                f"{low_conf_note}"
            )

        return result, status

    except Exception as e:
        return None, f"Error: {str(e)}"


with gr.Blocks(
    title="CLIP + SAM Zero-Shot Grounding",
    theme=gr.themes.Soft(),
    css="""
        .title-text { text-align: center; margin-bottom: 8px; }
        .subtitle-text { text-align: center; color: #666; margin-bottom: 20px; }
        .status-box { font-family: monospace; font-size: 13px; }
    """
) as demo:

    gr.HTML("""
        <div class="title-text">
            <h1>CLIP + SAM Zero-Shot Grounding</h1>
        </div>
        <div class="subtitle-text">
            <p>Upload an image and type what you're looking for.
            SAM segments everything -- CLIP finds the best match.</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
            )
            prompt_box = gr.Textbox(
                label="What are you looking for?",
                placeholder='e.g. "a car", "the dog", "red chair"',
                lines=1,
            )
            show_top3_checkbox = gr.Checkbox(
                label="Show top matches (useful when there are multiple similar objects, e.g. several white cars)",
                value=False,
            )
            run_btn = gr.Button(
                "Find It",
                variant="primary",
                size="lg",
            )
            gr.HTML(
                '<p style="text-align:center; color:#888; font-size:12px; margin-top:4px;">'
                'Runs on CPU -- typically takes 1-4 minutes depending on image complexity</p>'
            )

            gr.Examples(
                examples=[
                    ["data/54cae422a6fc6_-_american-muscle-facts-03-0312-xln.jpg", "car"],
                    ["data/obedient-golden-retriever-sitting-in-the-gr...", "dog"],
                ],
                inputs=[input_image, prompt_box],
                label="Try these examples",
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Result",
                type="numpy",
            )
            status_box = gr.Textbox(
                label="Info",
                lines=6,
                interactive=False,
                elem_classes=["status-box"],
            )

    run_btn.click(
        fn=ground_object,
        inputs=[input_image, prompt_box, show_top3_checkbox],
        outputs=[output_image, status_box],
    )

    prompt_box.submit(
        fn=ground_object,
        inputs=[input_image, prompt_box, show_top3_checkbox],
        outputs=[output_image, status_box],
    )

    gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:#888; font-size:12px;">
            Built with Transformers, Meta SAM, OpenAI CLIP
        </div>
    """)


if __name__ == "__main__":
    demo.launch(share=False)