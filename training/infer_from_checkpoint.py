# ============================================================
# STANDALONE INFERENCE — run after cell1 of any iteration
# Finds the latest checkpoint in iter4, runs prediction,
# writes submission CSV and ZIP. No training needed.
# ============================================================

import os, glob, zipfile
import numpy as np
import pandas as pd
import torch
import rasterio

# ---------------------------------------------------------------------------
# Find checkpoint — picks last.ckpt if exists, else best-*.ckpt
# ---------------------------------------------------------------------------
ckpt_dir  = "/kaggle/working/iter4/checkpoints"
ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
if not os.path.exists(ckpt_path):
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "best-*.ckpt")))
    ckpt_path  = candidates[-1] if candidates else None

if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

sys.__stdout__.write(f"Using checkpoint: {ckpt_path}\n")

# ---------------------------------------------------------------------------
# Load model from checkpoint
# Requires FloodSegmentationTask and CFG to be defined (run cell1 first)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FloodSegmentationTask(
    model_args = dict(
        backbone              = CFG["BACKBONE"],
        backbone_pretrained   = False,   # weights come from ckpt, skip HF download
        backbone_bands        = CFG["BANDS"],
        backbone_num_frames   = 1,
        decoder               = CFG["DECODER"],
        decoder_channels      = CFG["DECODER_CH"],
        decoder_scale_modules = True,
        necks = [
            dict(name="SelectIndices", indices=CFG["NECK_INDICES"]),
            dict(name="ReshapeTokensToImage", effective_time_dim=1),
        ],
        num_classes  = CFG["NUM_CLASSES"],
        head_dropout = CFG["HEAD_DROPOUT"],
        rescale      = True,
    ),
    model_factory     = "EncoderDecoderFactory",
    loss              = "ce",
    lr                = CFG["LR_DECODER"],
    optimizer         = "AdamW",
    optimizer_hparams = dict(weight_decay=CFG["WEIGHT_DECAY"]),
    ignore_index      = -1,
    freeze_backbone   = False,
    freeze_decoder    = False,
    plot_on_val       = False,
    class_weights     = CFG["CLASS_WEIGHTS"],
    dice_weight       = CFG["DICE_WEIGHT"],
)

state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["state_dict"], strict=False)
model.eval()
model = model.to(device)
sys.__stdout__.write(f"Model loaded — {sum(p.numel() for p in model.parameters())/1e6:.0f}M params\n")

# ---------------------------------------------------------------------------
# Prediction files — use engineered dir if iter4, else raw
# ---------------------------------------------------------------------------
pred_dir = CFG.get("ENG_PRED_DIR", CFG.get("PRED_DIR"))
pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.tif")))
pred_file_map = {
    os.path.basename(p).replace("_image.tif", "").replace(".tif", ""): p
    for p in pred_files
}
sys.__stdout__.write(f"Predicting {len(pred_files)} patches from {pred_dir}\n")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
all_preds = {}
with torch.no_grad():
    for pid, fp in pred_file_map.items():
        with rasterio.open(fp) as src:
            img_np = src.read().astype(np.float32)
        for b in range(CFG["NUM_BANDS"]):
            img_np[b] = (img_np[b] - MEANS[b]) / STDS[b]
        img_t = torch.tensor(img_np).unsqueeze(0).to(device)
        pred  = model.model(img_t).output.argmax(dim=1).squeeze(0).cpu().numpy().astype("int16")
        all_preds[pid] = pred

sys.__stdout__.write(f"Done: {len(all_preds)} predictions\n")

# ---------------------------------------------------------------------------
# RLE + CSV
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

with open(os.path.join(CFG["SPLIT_DIR"], "pred.txt")) as f:
    submit_ids = [l.strip() for l in f if l.strip()]

rows, issues = [], []
for tid in submit_ids:
    pred_arr = all_preds.get(tid)
    if pred_arr is None:
        sys.__stdout__.write(f"  WARNING: missing {tid} — using '0 0'\n")
        rows.append({"id": tid, "rle_mask": "0 0"})
        issues.append(tid)
        continue
    flood_bin = (pred_arr == 1).astype(np.uint8)
    rows.append({"id": tid, "rle_mask": mask_to_rle(flood_bin)})

sub_df  = pd.DataFrame(rows)
out_dir = CFG.get("OUT_DIR", "/kaggle/working/iter4")
sub_csv = os.path.join(out_dir, "submission_infer.csv")
sub_df.to_csv(sub_csv, index=False)
sys.__stdout__.write(
    f"Submission: {len(sub_df)} rows | "
    f"empty: {(sub_df['rle_mask']=='0 0').sum()} | issues: {len(issues)}\n"
)

zip_path = "/kaggle/working/submission_infer.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub_csv, "submission_infer.csv")

sys.__stdout__.write(f"ZIP: {zip_path} ({os.path.getsize(zip_path)/1024:.1f} KB)\n")
sys.__stdout__.write(f"Checkpoint used: {os.path.basename(ckpt_path)}\n")
