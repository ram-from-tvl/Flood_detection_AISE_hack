"""
Ensemble Inference — iter8 + iter9 model averaging

Loads both trained checkpoints, runs inference on prediction patches,
averages softmax probabilities, and writes the submission CSV.

iter8: original channels + SelectIndices + trainval + two-phase LR
iter9: same as iter8 + boundary-aware loss + TTA

Ensembling averages the two probability distributions before argmax.
Models trained with different loss functions make different errors —
averaging cancels individual mistakes and improves Flood IoU.
"""

import os, glob, zipfile
import numpy as np
import pandas as pd
import torch
import rasterio
import terratorch
from terratorch.tasks import SemanticSegmentationTask

# ---------------------------------------------------------------------------
# Paths — update checkpoint paths if filenames differ
# ---------------------------------------------------------------------------
BASE_DIR  = "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data"
SPLIT_DIR = os.path.join(BASE_DIR, "split")
OUT_DIR   = "/kaggle/working/ensemble"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_ITER8 = next(iter(sorted(glob.glob(
    "/kaggle/working/iter8/checkpoints/best-*.ckpt"))), None)
CKPT_ITER9 = next(iter(sorted(glob.glob(
    "/kaggle/working/iter9/checkpoints/best-*.ckpt"))), None)

print(f"iter8 ckpt: {CKPT_ITER8}")
print(f"iter9 ckpt: {CKPT_ITER9}")

# Both models use original 6-band norm stats
MEANS = [841.1257, 371.6175, 1734.1789, 1588.3142, 1742.8452, 1218.5551]
STDS  = [473.7090, 170.3611,  623.0474,  612.8465,  564.5835,  528.0894]

# ---------------------------------------------------------------------------
# Model loader — same architecture for both iter8 and iter9
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    model = SemanticSegmentationTask(
        model_args=dict(
            backbone_pretrained=False,   # weights come from checkpoint
            backbone="prithvi_eo_v2_300_tl",
            backbone_bands=[1, 2, 3, 4, 5, 6],
            backbone_num_frames=1,
            decoder="UperNetDecoder",
            decoder_channels=256,
            decoder_scale_modules=True,
            num_classes=3,
            head_dropout=0.1,
            necks=[
                dict(name="SelectIndices", indices=[5, 11, 17, 23]),
                dict(name="ReshapeTokensToImage", effective_time_dim=1),
            ],
            rescale=True,
        ),
        plot_on_val=False,
        class_weights=[0.3, 5.0, 1.0],
        loss="ce",
        lr=2e-5,
        optimizer="AdamW",
        optimizer_hparams=dict(weight_decay=0.05),
        ignore_index=-1,
        freeze_backbone=False,
        freeze_decoder=False,
        model_factory="EncoderDecoderFactory",
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    return model.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_8 = load_model(CKPT_ITER8, device)
model_9 = load_model(CKPT_ITER9, device)
print("Both models loaded")

# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------
pred_files = sorted(glob.glob(os.path.join(BASE_DIR, "prediction", "image", "*.tif")))
print(f"Predicting {len(pred_files)} patches...")

pred_map = {}
with torch.no_grad():
    for fp in pred_files:
        pid = os.path.basename(fp).replace("_image.tif", "").replace(".tif", "")
        with rasterio.open(fp) as src:
            img = src.read().astype(np.float32)
        for b in range(6):
            img[b] = (img[b] - MEANS[b]) / STDS[b]
        img_t = torch.tensor(img).unsqueeze(0).to(device)

        probs_8 = torch.softmax(model_8.model(img_t).output, dim=1)
        probs_9 = torch.softmax(model_9.model(img_t).output, dim=1)

        avg = (probs_8 + probs_9) / 2.0
        pred_map[pid] = avg.argmax(dim=1).squeeze(0).cpu().numpy().astype("int16")

print(f"Done: {len(pred_map)} patches")

# ---------------------------------------------------------------------------
# RLE encoding
# ---------------------------------------------------------------------------
def mask_to_rle(mask):
    flat = mask.flatten(order="F").astype(np.uint8)
    if flat.sum() == 0:
        return "0 0"
    padded  = np.concatenate([[0], flat, [0]])
    diffs   = np.diff(padded.astype(np.int8))
    starts  = np.where(diffs ==  1)[0] + 1
    lengths = np.where(diffs == -1)[0] - starts + 1
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))

with open(os.path.join(SPLIT_DIR, "pred.txt")) as f:
    submit_ids = [l.strip() for l in f if l.strip()]

rows = []
for tid in submit_ids:
    arr = pred_map.get(tid)
    if arr is None:
        rows.append({"id": tid, "rle_mask": "0 0"})
        print(f"WARNING: missing {tid}")
        continue
    rows.append({"id": tid, "rle_mask": mask_to_rle((arr == 1).astype(np.uint8))})

sub_df  = pd.DataFrame(rows)
sub_csv = os.path.join(OUT_DIR, "submission_ensemble.csv")
sub_df.to_csv(sub_csv, index=False)
print(f"Submission: {len(sub_df)} rows | empty: {(sub_df['rle_mask']=='0 0').sum()}")

zip_path = "/kaggle/working/submission_ensemble.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub_csv, "submission_ensemble.csv")
print(f"Submit: {zip_path}")
