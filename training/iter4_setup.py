# ============================================================
# ITER 4 — CELL 1: Setup
# ============================================================
#
# Workaround: use Prithvi with engineered 6-channel input.
#
# EDA finding:
#   Optical (Green, Red, NIR, SWIR): KS ≈ 0.00, cloud-covered, useless
#   SAR raw backscatter KS (Flood vs WaterBody): 0.40
#   SAR local texture KS (Flood vs WaterBody):  0.55  ← strongest signal
#
# Strategy: keep Prithvi's 6-channel input but replace the 4 dead
# optical channels with the 4 SAR texture channels:
#   [SAR_HH, SAR_HV, HH_std_3x3, HH_std_7x7, HV_std_3x3, HV_std_7x7]
#
# Prithvi's Transformer layers (300M params) are pretrained and valid.
# Only the patch embedding (~1M params) adapts to new channel stats.
# The Transformer body is the valuable pretrained asset — it remains.
#
# Implementation: precompute and save new 6-channel TIFs once to
# /kaggle/working/iter4_data/, then train exactly as iter2/3.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "terratorch", "albumentations", "rasterio", "-q"])

import os, glob, zipfile, json, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask

warnings.filterwarnings("ignore")
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG = {
    "BASE_DIR"      : "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data",
    "OUT_DIR"       : "/kaggle/working/iter4",
    "DATA_DIR"      : "/kaggle/working/iter4_data",   # precomputed 6-ch TIFs
    "NUM_CLASSES"   : 3,
    "NUM_BANDS"     : 6,
    "BAND_NAMES"    : ["SAR_HH", "SAR_HV", "HH_std3", "HH_std7", "HV_std3", "HV_std7"],
    "BANDS"         : [1, 2, 3, 4, 5, 6],
    "BACKBONE"      : "prithvi_eo_v2_300_tl",
    "DECODER"       : "UperNetDecoder",
    "DECODER_CH"    : 256,
    "HEAD_DROPOUT"  : 0.1,
    "NECK_INDICES"  : [5, 11, 17, 23],
    "EPOCHS"        : 50,
    "BATCH_SIZE"    : 2,
    "LR_DECODER"    : 8e-6,
    "LR_BACKBONE"   : 8e-7,
    "WEIGHT_DECAY"  : 0.05,
    "NUM_WORKERS"   : 2,
    "SEED"          : 42,
    "CLASS_WEIGHTS" : [0.5, 3.0, 1.5],
    "DICE_WEIGHT"   : 0.5,
    "ES_PATIENCE"   : 15,
}

CFG["IMG_DIR"]   = os.path.join(CFG["BASE_DIR"], "image")
CFG["LABEL_DIR"] = os.path.join(CFG["BASE_DIR"], "label")
CFG["PRED_DIR"]  = os.path.join(CFG["BASE_DIR"], "prediction", "image")
CFG["SPLIT_DIR"] = os.path.join(CFG["BASE_DIR"], "split")
CFG["CKPT_DIR"]  = os.path.join(CFG["OUT_DIR"], "checkpoints")

for d in [CFG["OUT_DIR"], CFG["CKPT_DIR"],
          os.path.join(CFG["DATA_DIR"], "image"),
          os.path.join(CFG["DATA_DIR"], "label"),
          os.path.join(CFG["DATA_DIR"], "prediction", "image")]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Texture computation
# local_std via uniform_filter: std(x)=sqrt(E[x^2]-E[x]^2), no pixel loops
# ---------------------------------------------------------------------------
def local_std(band, radius):
    b     = band.astype(np.float64)
    mean  = uniform_filter(b, size=radius)
    mean2 = uniform_filter(b ** 2, size=radius)
    return np.sqrt(np.maximum(mean2 - mean ** 2, 0.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Precompute 6-channel TIFs
# Input: [SAR_HH, SAR_HV, Green, Red, NIR, SWIR]  (original 6-band)
# Output: [SAR_HH, SAR_HV, HH_std3, HH_std7, HV_std3, HV_std7] (new 6-band)
#
# Labels are copied unchanged — identical 3-class masks.
# Only runs if files don't already exist (safe to rerun).
# ---------------------------------------------------------------------------
def precompute_engineered_tifs(src_dir, dst_dir, file_pattern="*_image.tif"):
    src_files = sorted(glob.glob(os.path.join(src_dir, file_pattern)))
    dst_img_dir = os.path.join(dst_dir, "image")
    built = 0

    for src_path in src_files:
        dst_path = os.path.join(dst_img_dir, os.path.basename(src_path))
        if os.path.exists(dst_path):
            continue

        with rasterio.open(src_path) as src:
            arr  = src.read().astype(np.float32)  # (6, H, W)
            meta = src.meta.copy()

        hh     = arr[0]
        hv     = arr[1]
        hh_s3  = local_std(hh, 3)
        hh_s7  = local_std(hh, 7)
        hv_s3  = local_std(hv, 3)
        hv_s7  = local_std(hv, 7)

        new_arr = np.stack([hh, hv, hh_s3, hh_s7, hv_s3, hv_s7], axis=0)

        meta.update({"count": 6, "dtype": "float32"})
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(new_arr)
        built += 1

    return built


print("Precomputing engineered TIFs (labelled patches)...")
n = precompute_engineered_tifs(CFG["IMG_DIR"],
                                CFG["DATA_DIR"])
print(f"  Built {n} new image TIFs in {CFG['DATA_DIR']}/image/")

# Copy prediction images too
print("Precomputing engineered TIFs (prediction patches)...")
n_pred = precompute_engineered_tifs(
    CFG["PRED_DIR"],
    os.path.join(CFG["DATA_DIR"], "prediction")
)
print(f"  Built {n_pred} new prediction TIFs")

# Symlink / copy label files (unchanged)
label_dst_dir = os.path.join(CFG["DATA_DIR"], "label")
src_labels = glob.glob(os.path.join(CFG["LABEL_DIR"], "*.tif"))
copied_labels = 0
for lp in src_labels:
    dst_lp = os.path.join(label_dst_dir, os.path.basename(lp))
    if not os.path.exists(dst_lp):
        import shutil
        shutil.copy2(lp, dst_lp)
        copied_labels += 1
print(f"  Copied {copied_labels} label TIFs")

# Verify counts
n_img  = len(glob.glob(os.path.join(CFG["DATA_DIR"], "image", "*.tif")))
n_lbl  = len(glob.glob(os.path.join(CFG["DATA_DIR"], "label", "*.tif")))
n_pred = len(glob.glob(os.path.join(CFG["DATA_DIR"], "prediction", "image", "*.tif")))
print(f"\nData ready: {n_img} images | {n_lbl} labels | {n_pred} prediction patches")

# Update CFG to point at engineered data
CFG["ENG_IMG_DIR"]  = os.path.join(CFG["DATA_DIR"], "image")
CFG["ENG_LBL_DIR"]  = os.path.join(CFG["DATA_DIR"], "label")
CFG["ENG_PRED_DIR"] = os.path.join(CFG["DATA_DIR"], "prediction", "image")


# ---------------------------------------------------------------------------
# Build trainval.txt (all 69 labelled patches)
# ---------------------------------------------------------------------------
trainval_path = os.path.join(CFG["OUT_DIR"], "trainval.txt")
if not os.path.exists(trainval_path):
    ids = []
    for fname in ["train.txt", "val.txt"]:
        with open(os.path.join(CFG["SPLIT_DIR"], fname)) as f:
            ids += [l.strip() for l in f if l.strip()]
    with open(trainval_path, "w") as f:
        f.write("\n".join(ids))
    print(f"Created trainval.txt: {len(ids)} patches")
else:
    with open(trainval_path) as f:
        n = sum(1 for l in f if l.strip())
    print(f"trainval.txt exists: {n} patches")

CFG["TRAINVAL_SPLIT"] = trainval_path


# ---------------------------------------------------------------------------
# Normalisation — computed from engineered training TIFs
# These are different from iter1-3 because channels have changed.
# ---------------------------------------------------------------------------
def compute_norm_stats(image_dir, split_file, num_bands, band_names):
    with open(split_file) as f:
        ids = [l.strip() for l in f if l.strip()]
    id_to_path = {
        os.path.basename(p).replace("_image.tif","").replace(".tif",""): p
        for p in glob.glob(os.path.join(image_dir, "*.tif"))
    }
    count = np.zeros(num_bands, dtype=np.float64)
    mean  = np.zeros(num_bands, dtype=np.float64)
    M2    = np.zeros(num_bands, dtype=np.float64)

    for pid in ids:
        path = next((v for k,v in id_to_path.items() if pid in k or k in pid), None)
        if path is None: continue
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float64)
        for b in range(min(num_bands, arr.shape[0])):
            vals = arr[b].ravel()
            vals = vals[vals > 0] if b < 2 else vals  # exclude nodata for raw SAR
            for x in vals:
                count[b] += 1
                d = x - mean[b]
                mean[b] += d / count[b]
                M2[b]   += d * (x - mean[b])

    std = np.maximum(np.sqrt(M2 / np.maximum(count - 1, 1)), 1e-6)
    print("Normalisation (engineered channels):")
    for b, name in enumerate(band_names):
        print(f"  {name:<12} mean={mean[b]:.4f}  std={std[b]:.4f}")
    return mean.tolist(), std.tolist()


MEANS, STDS = compute_norm_stats(
    CFG["ENG_IMG_DIR"],
    CFG["TRAINVAL_SPLIT"],
    CFG["NUM_BANDS"],
    CFG["BAND_NAMES"],
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index).float()
        loss  = 0.0
        for c in range(self.num_classes):
            p = probs[:, c] * valid
            t = (targets == c).float() * valid
            loss += 1.0 - (2.0*(p*t).sum() + self.smooth) / \
                          (p.sum() + t.sum() + self.smooth)
        return loss / self.num_classes


# ---------------------------------------------------------------------------
# Task — identical to iter3 structure
# ---------------------------------------------------------------------------
class FloodSegmentationTask(SemanticSegmentationTask):

    def __init__(self, dice_weight=0.5, lr_backbone=8e-7,
                 total_steps=2000, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight     = dice_weight
        self.lr_backbone     = lr_backbone
        self.total_steps     = total_steps
        self.val_step_outputs = []
        w = torch.tensor(CFG["CLASS_WEIGHTS"], dtype=torch.float32)
        self.ce_loss   = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
        self.dice_loss = MulticlassDiceLoss(CFG["NUM_CLASSES"], ignore_index=-1)

    def _loss(self, logits, y):
        self.ce_loss.weight = self.ce_loss.weight.to(logits.device)
        ce   = self.ce_loss(logits, y)
        dice = self.dice_loss(logits, y)
        return (1 - self.dice_weight)*ce + self.dice_weight*dice

    def training_step(self, batch, batch_idx):
        y = batch["mask"].long()
        if y.ndim == 4: y = y.squeeze(1)
        logits = self.model(batch["image"]).output
        loss   = self._loss(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["mask"].long()
        if y.ndim == 4: y = y.squeeze(1)
        logits = self.model(batch["image"]).output
        loss   = self._loss(logits, y)
        preds  = logits.argmax(dim=1)
        self.val_step_outputs.append({"preds": preds.cpu(), "targets": y.cpu()})
        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return
        all_preds   = torch.cat([x["preds"]   for x in self.val_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.val_step_outputs])
        self.val_step_outputs.clear()

        valid   = (all_targets != -1)
        iou_sum = 0.0
        parts   = []
        for c, name in enumerate(["NoFlood", "Flood", "WaterBody"]):
            tp  = ((all_preds==c) & (all_targets==c) & valid).sum().float()
            fp  = ((all_preds==c) & (all_targets!=c) & valid).sum().float()
            fn  = ((all_preds!=c) & (all_targets==c) & valid).sum().float()
            iou = (tp / (tp + fp + fn + 1e-6)).item()
            iou_sum += iou
            self.log(f"val/IoU_{name}", iou, prog_bar=(name=="Flood"))
            parts.append(f"{name}={iou:.3f}")
        miou = iou_sum / CFG["NUM_CLASSES"]
        self.log("val/mIoU", miou, prog_bar=True)
        sys.__stdout__.write(
            f"\n  Epoch {self.current_epoch}: mIoU={miou:.4f}  |  "
            + "  ".join(parts) + "\n"
        )
        sys.__stdout__.flush()

    def configure_optimizers(self):
        backbone_params = list(self.model.encoder.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        other_params    = [p for p in self.parameters() if id(p) not in backbone_ids]
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.lr_backbone},
            {"params": other_params,    "lr": CFG["LR_DECODER"]},
        ], weight_decay=CFG["WEIGHT_DECAY"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.total_steps, eta_min=1e-8
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


print("\nCell 1 done.")