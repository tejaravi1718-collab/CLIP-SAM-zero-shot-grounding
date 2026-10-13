# CLIP + SAM Zero-Shot Visual Grounding

Ground any object in an image using natural language — no training, no labeled dataset, no fine-tuning.

> Upload an image, type what you're looking for (e.g. `"a car"`, `"the dog"`, `"red chair"`), and the pipeline highlights exactly that object.

---

## How It Works

This project combines two pretrained vision models with no additional training:

- **SAM (Segment Anything Model)** — Meta AI's model that proposes every plausible object mask in an image, with no knowledge of what each object actually is.
- **CLIP (Contrastive Language-Image Pretraining)** — OpenAI's model that scores how well a text prompt matches an image region.

**Pipeline:** image → SAM proposes all candidate masks → CLIP scores each mask against the text prompt → highest-scoring mask is returned.

This is "zero-shot grounding": the system can localize objects it was never explicitly trained to recognize, as long as CLIP has seen the concept during its own pretraining.

---

## The Core Problem (and Why Naive CLIP+SAM Fails)

A naive implementation — crop each SAM mask, score it against the prompt with plain CLIP similarity, take the highest score — fails in a specific, predictable way: **it tends to pick a recognizable *part* of an object instead of the whole object.**

Example: prompt `"a car"` on a photo of a car returns a mask of just the door, or just a wheel, instead of the full car — even though a full-car mask exists among SAM's candidates.

**Why this happens:** CLIP was trained on natural photographs, and tight, clean crops of recognizable object *parts* (a car door panel, a wheel) often score very highly against simple prompts like `"a car"`, because the crop is unambiguous and high-contrast. A full-object crop, by contrast, often includes more background, shadow, and visual complexity — which can dilute its raw similarity score relative to a clean partial crop. Plain similarity scoring has no concept of "whole object vs. part of object" — it just measures resemblance.

### The Fix: Contrastive Scoring

Instead of scoring a mask against the target prompt alone, each candidate mask is scored against **the target prompt plus three explicit distractor prompts**, and CLIP computes a softmax probability across all four:

```python
candidate_prompts = [
    f"a photo of a whole {object}",   # target: complete object
    f"a part of a {object}",           # distractor: partial object
    "a photo of the background",       # distractor: background/scenery
    "a blurry, cropped photo",         # distractor: low-quality crop
]
```

This reframes the question CLIP is answering. Instead of "does this look like a car?" (which a door panel can satisfy), it becomes "is this crop **more** like a whole car than it is like a part of a car, the background, or a bad crop?" — a relative judgment that naturally favors complete, well-formed object masks over partial ones.

This single change was the difference between the pipeline consistently picking a car door and consistently picking the full car.

---

## Known Limitation: Low-Texture Images

SAM's automatic mask generator works by sampling a grid of points across the image and asking "what object is here?" at each point. On images with **plenty of internal texture and contrast** (a car with window trim, door seams, wheel wells), this produces many distinct, well-separated mask candidates.

On **low-texture, plain-background images** (e.g. a pet photographed against a smooth studio backdrop), the grid can produce very few distinct masks — sometimes only 2-3 — because there are few sharp internal edges for SAM to latch onto. In this situation, the correct full-object mask may simply never get proposed.

**Mitigation implemented:** if the initial segmentation pass produces fewer than 5 candidate masks after filtering, the pipeline automatically retries with a denser sampling grid (`points_per_side=32`) and relaxed acceptance thresholds. This generates more region hypotheses for low-texture images at the cost of extra processing time, only in this specific case.

This is a genuine characteristic of grid-based automatic segmentation, not a bug — and the fallback is the documented mitigation, not a complete guarantee. Extremely flat, low-contrast images may still occasionally underperform.

---

## Performance Characteristics (CPU-Only)

This project runs entirely on CPU (no GPU/CUDA required), using SAM's smallest checkpoint (`vit_b`) specifically chosen for CPU feasibility.

**Measured on a mid-range CPU (Intel i5 / Ryzen 5 class, 8 threads, no GPU):**

| Stage | Time |
|---|---|
| SAM image encoder (one-time per image, ViT-B forward pass) | ~16s |
| SAM mask decoder (per grid point) | ~0.07s/point |
| Default grid (`points_per_side=20` → 400 points) | ~45s decode |
| Fallback grid (`points_per_side=32` → 1024 points, low-texture images only) | ~75s decode |
| CLIP contrastive scoring (per mask, 4-prompt softmax) | ~0.1–0.3s/mask |

**Typical end-to-end time per query: 45 seconds to 4 minutes**, depending on image complexity and whether the low-texture fallback triggers. This was confirmed by direct profiling (see `diagnose_sam_speed.py`), which isolates the SAM image encoder cost from the per-point decoder cost — both came back within the expected range for `vit_b` on CPU.

This is a real hardware constraint of running a ViT-based segmentation model without GPU acceleration, not a bug in the pipeline. A live progress bar in the Gradio app communicates pipeline stage during the wait so it doesn't appear frozen.

**If GPU were available:** moving `device="cuda"` for both SAM and CLIP would realistically bring this down to 2-5 seconds end-to-end, since the ViT encoder is highly parallelizable and CPU is the binding constraint.

---

## Project Structure

```
clip_sam_project/
├── checkpoints/
│   └── sam_vit_b_01ec64.pth      # SAM model weights (download separately)
├── data/                          # Sample input images
├── notebooks/
│   ├── clip_intro.ipynb           # CLIP fundamentals walkthrough
│   ├── sam_intro.ipynb            # SAM + CLIP integration walkthrough
│   └── demo.ipynb                 # Full pipeline demo with score tables
├── outputs/                       # Saved results
├── src/
│   ├── clip_utils.py              # CLIP loading + contrastive scoring
│   ├── sam_utils.py               # SAM loading + mask generation + filtering
│   ├── grounding.py                # Mask ranking logic
│   └── visualization.py            # Mask overlay + side-by-side rendering
├── app.py                          # Gradio web app (with live progress bar)
├── run_grounding.py                # CLI script
├── diagnose_sam_speed.py           # Standalone SAM performance profiler
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/tejaravi1718-collab/CLIP-SAM-zero-shot-grounding.git
cd CLIP-SAM-zero-shot-grounding
pip install -r requirements.txt
```

### 2. Download SAM checkpoint

```bash
mkdir checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

---

## Usage

### Gradio Web App

```bash
python app.py
```

Opens at `http://localhost:7860`. Upload any image, type a prompt, click **Find It**, and watch the progress bar through each pipeline stage.

### CLI Script

```bash
python run_grounding.py --image data/your_image.jpg --prompt "a car"
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--image` | sample car image | Path to input image |
| `--prompt` | `"car"` | Text prompt to ground |
| `--checkpoint` | `checkpoints/sam_vit_b_01ec64.pth` | SAM weights path |
| `--output` | `outputs/result.png` | Output image path |
| `--points` | `20` | SAM `points_per_side` (higher = more masks, slower) |

**Example:**

```bash
python run_grounding.py \
    --image data/street.jpg \
    --prompt "traffic light" \
    --points 20 \
    --output outputs/traffic_light.png
```

### Performance Diagnostic

```bash
python diagnose_sam_speed.py
```

Isolates SAM's image-encoder time from its per-point decoder time, useful for understanding where processing time goes on your specific hardware.

---

## Engineering Notes

A few decisions worth calling out for anyone reviewing this codebase:

- **Mask filtering by area** (`min_area_ratio=0.005`, `max_area_ratio=0.90`) removes noise-level masks and near-full-image background masks before scoring, reducing wasted CLIP calls and eliminating obviously-wrong candidates early.
- **Padded crops** (10% padding around each mask's bounding box) give CLIP natural-looking image crops rather than tight, edge-touching crops, which it handles more reliably — this matches how CLIP's own training data was typically composed.
- **Strict-then-relaxed SAM thresholds**: defaults match SAM's own recommended strict settings to avoid "leaky" masks that bleed across object boundaries (loosening thresholds globally was tried and reverted after it was found to let exactly this kind of leaky mask through on a textured image). Relaxed thresholds are applied only as a fallback when too few masks are returned.
- **Non-determinism**: `SamAutomaticMaskGenerator` does not expose a fixed random seed, so mask boundaries can vary slightly between runs on the same image. This is a documented characteristic of the library, not a defect introduced here.

---

## Models Used

| Model | Source | Size |
|---|---|---|
| `openai/clip-vit-base-patch32` | HuggingFace Transformers | ~340MB |
| `sam_vit_b` | Meta AI (Segment Anything) | ~375MB |

---

## References

- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything)
- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- [HuggingFace Transformers — CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
