"""
diagnose_sam_speed.py
----------------------
Standalone diagnostic to isolate WHERE SAM spends its time:
the one-time image encoder pass, or the per-point mask decoder loop.

Run with:
    python diagnose_sam_speed.py
"""

import time
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor

print("=" * 60)
print("SAM SPEED DIAGNOSTIC")
print("=" * 60)

print(f"\nTorch threads: {torch.get_num_threads()}")
print(f"Torch version: {torch.__version__}")

# ── Load SAM ──────────────────────────────────
print("\n[1] Loading SAM model...")
t0 = time.time()
sam = sam_model_registry["vit_b"](
    checkpoint="checkpoints/sam_vit_b_01ec64.pth"
)
sam.to("cpu")
sam.eval()
print(f"    Model loaded in {time.time() - t0:.2f}s")

# ── Create a dummy image ──
print("\n[2] Creating dummy test image (512x512 random)...")
dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# ── Time the IMAGE ENCODER (one-time per image) ──
print("\n[3] Running image encoder (set_image) — this should be the slow one-time cost...")
predictor = SamPredictor(sam)

t0 = time.time()
predictor.set_image(dummy_image)
encoder_time = time.time() - t0
print(f"    >>> Image encoder took: {encoder_time:.2f}s <<<")

# ── Time a SINGLE point prediction (decoder) ──
print("\n[4] Running single-point mask decoder...")
input_point = np.array([[256, 256]])
input_label = np.array([1])

t0 = time.time()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
decoder_time_single = time.time() - t0
print(f"    >>> Single-point decoder took: {decoder_time_single:.4f}s <<<")

# ── Time 10 more point predictions ──
print("\n[5] Running 10 more point predictions (encoder already cached)...")
t0 = time.time()
for i in range(10):
    px = np.random.randint(50, 462)
    py = np.random.randint(50, 462)
    predictor.predict(
        point_coords=np.array([[px, py]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
decoder_time_10 = time.time() - t0
print(f"    >>> 10 more point predictions took: {decoder_time_10:.2f}s "
      f"({decoder_time_10/10:.3f}s per point) <<<")

# ── Summary ──────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Image encoder (one-time)  : {encoder_time:.2f}s")
print(f"  Per-point decoder (avg)   : {decoder_time_10/10:.3f}s")
print(f"  Projected cost for 1024 points (32x32 grid):")
print(f"    Encoder: {encoder_time:.2f}s (paid once)")
print(f"    Decoder: {(decoder_time_10/10) * 1024:.2f}s (1024 points)")
print(f"    TOTAL ESTIMATE: {encoder_time + (decoder_time_10/10) * 1024:.2f}s")
print("=" * 60)

print("\nIf the encoder time alone is 60s+, the bottleneck is the ViT")
print("image encoder itself -- likely a CPU/MKL threading issue.")
print("\nIf per-point decoder time is 0.05s+ each, 1024 points will")
print("naturally take 50s+ just in the decode loop -- this is")
print("expected and points_per_side should be reduced.")