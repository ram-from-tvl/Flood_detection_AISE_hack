# ============================================================
# CELL 1 — Run once per session (or after kernel restart)
# Installs, imports, config, norm stats, loss + task classes
# ============================================================

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "terratorch", "albumentations", "rasterio",
                "numpy==2.2.0",
                "-q"])

import os, glob, zipfile, json, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask

warnings.filterwarnings("ignore")
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = {
    "BASE_DIR"      : "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data",
    "OUT_DIR"       : "/kaggle/working/iter1",
    "BAND_NAMES"    : ["SAR_HH", "SAR_HV", "Green", "Red", "NIR", "SWIR"],
    "BANDS"         : [1, 2, 3, 4, 5, 6],
    "NUM_CLASSES"   : 3,
    "NUM_BANDS"     : 6,
    "BACKBONE"      : "prithvi_eo_v2_300_tl",
    "DECODER"       : "UperNetDecoder",
    "DECODER_CH"    : 256,
    "HEAD_DROPOUT"  : 0.1,
    "NECK_INDICES"  : [5, 11, 17, 23],
    "EPOCHS"        : 20,
    "BATCH_SIZE"    : 2,
    "LR"            : 2e-5,
    "WEIGHT_DECAY"  : 0.05,
    "NUM_WORKERS"   : 2,
    "SEED"          : 42,
    "CLASS_WEIGHTS" : [0.5, 3.0, 1.5],
    "DICE_WEIGHT"   : 0.5,
    "ES_PATIENCE"   : 10,
}

CFG["IMG_DIR"]   = os.path.join(CFG["BASE_DIR"], "image")
CFG["LABEL_DIR"] = os.path.join(CFG["BASE_DIR"], "label")
CFG["PRED_DIR"]  = os.path.join(CFG["BASE_DIR"], "prediction", "image")
CFG["SPLIT_DIR"] = os.path.join(CFG["BASE_DIR"], "split")
CFG["CKPT_DIR"]  = os.path.join(CFG["OUT_DIR"], "checkpoints")

os.makedirs(CFG["OUT_DIR"],  exist_ok=True)
os.makedirs(CFG["CKPT_DIR"], exist_ok=True)


# ---------------------------------------------------------------------------
# Norm stats — slow (scans all train TIFs). Cached in MEANS/STDS after first run.
# ---------------------------------------------------------------------------
def compute_norm_stats(image_dir, split_file, num_bands, band_names):
    with open(split_file) as f:
        ids = [l.strip() for l in f if l.strip()]
    id_to_path = {
        os.path.basename(p).replace("_image.tif", "").replace(".tif", ""): p
        for p in glob.glob(os.path.join(image_dir, "*.tif"))
    }
    count = np.zeros(num_bands, dtype=np.float64)
    mean  = np.zeros(num_bands, dtype=np.float64)
    M2    = np.zeros(num_bands, dtype=np.float64)
    for pid in ids:
        path = next((v for k, v in id_to_path.items() if pid in k or k in pid), None)
        if path is None:
            continue
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float64)
        for b in range(min(num_bands, arr.shape[0])):
            vals = arr[b].ravel()
            vals = vals[vals > 0] if b < 2 else vals
            for x in vals:
                count[b] += 1
                d = x - mean[b]
                mean[b] += d / count[b]
                M2[b]   += d * (x - mean[b])
    std = np.maximum(np.sqrt(M2 / np.maximum(count - 1, 1)), 1e-6)
    print("Normalisation:")
    for b, name in enumerate(band_names):
        print(f"  {name:<10} mean={mean[b]:.2f}  std={std[b]:.2f}")
    return mean.tolist(), std.tolist()

MEANS, STDS = compute_norm_stats(
    CFG["IMG_DIR"], os.path.join(CFG["SPLIT_DIR"], "train.txt"),
    CFG["NUM_BANDS"], CFG["BAND_NAMES"])


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
            loss += 1.0 - (2.0 * (p * t).sum() + self.smooth) / \
                          (p.sum() + t.sum() + self.smooth)
        return loss / self.num_classes


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class FloodSegmentationTask(SemanticSegmentationTask):

    def __init__(self, dice_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight      = dice_weight
        self.val_step_outputs = []
        w = torch.tensor(CFG["CLASS_WEIGHTS"], dtype=torch.float32)
        self.ce_loss   = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
        self.dice_loss = MulticlassDiceLoss(CFG["NUM_CLASSES"], ignore_index=-1)

    def _loss(self, logits, y):
        self.ce_loss.weight = self.ce_loss.weight.to(logits.device)
        ce   = self.ce_loss(logits, y)
        dice = self.dice_loss(logits, y)
        return (1 - self.dice_weight) * ce + self.dice_weight * dice

    def training_step(self, batch, batch_idx):
        y = batch["mask"].long()
        if y.ndim == 4:
            y = y.squeeze(1)
        logits = self.model(batch["image"]).output
        loss   = self._loss(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["mask"].long()
        if y.ndim == 4:
            y = y.squeeze(1)
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
            tp  = ((all_preds == c) & (all_targets == c) & valid).sum().float()
            fp  = ((all_preds == c) & (all_targets != c) & valid).sum().float()
            fn  = ((all_preds != c) & (all_targets == c) & valid).sum().float()
            iou = (tp / (tp + fp + fn + 1e-6)).item()
            iou_sum += iou
            self.log(f"val/IoU_{name}", iou, prog_bar=(name == "Flood"))
            parts.append(f"{name}={iou:.3f}")
        miou = iou_sum / CFG["NUM_CLASSES"]
        self.log("val/mIoU", miou, prog_bar=True)
        sys.__stdout__.write(
            f"\n  Epoch {self.current_epoch}: mIoU={miou:.4f}  |  " + "  ".join(parts) + "\n"
        )
        sys.__stdout__.flush()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"]
        )


print("\nCell 1 done — MEANS, STDS, CFG, classes all ready.")
