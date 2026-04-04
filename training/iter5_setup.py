# ============================================================
# ITER 5 — CELL 1: Setup
# Strategy: fix the train/LB gap (overfit) from iter4
#
# iter4 result: val IoU_Flood=0.31 (on trainval), LB=0.12
# Gap cause: only 69 patches, model memorises them
#
# Fixes:
#   1. Heavy augmentation — primary regularizer for tiny datasets
#   2. Label smoothing (eps=0.1) — stops overconfident predictions
#   3. Head dropout 0.1 → 0.3 — more regularization at output
#   4. Proper train/val split — use train.txt only for training,
#      val.txt for monitoring, so we can see real overfitting
#   5. Warm-start from iter4 best ckpt (0.31 val IoU)
#   6. Lower LR (warm-start needs gentler updates)
# ============================================================

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "terratorch", "albumentations", "rasterio",
                "numpy==2.2.0", "-q"])

import os, glob, zipfile, json, warnings, shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from scipy.ndimage import uniform_filter
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
# Config
# ---------------------------------------------------------------------------
CFG = {
    "BASE_DIR"      : "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data",
    "OUT_DIR"       : "/kaggle/working/iter5",
    "DATA_DIR"      : "/kaggle/working/iter4_data",   # reuse iter4 engineered TIFs
    "NUM_CLASSES"   : 3,
    "NUM_BANDS"     : 6,
    "BAND_NAMES"    : ["SAR_HH", "SAR_HV", "HH_std3", "HH_std7", "HV_std3", "HV_std7"],
    "BANDS"         : [1, 2, 3, 4, 5, 6],
    "BACKBONE"      : "prithvi_eo_v2_300_tl",
    "DECODER"       : "UperNetDecoder",
    "DECODER_CH"    : 256,
    "HEAD_DROPOUT"  : 0.3,        # iter4=0.1 → 0.3, more regularization
    "NECK_INDICES"  : [5, 11, 17, 23],
    "EPOCHS"        : 10,
    "BATCH_SIZE"    : 2,
    "LR_DECODER"    : 4e-6,       # iter4=8e-6 → half, warm-start needs gentle LR
    "LR_BACKBONE"   : 4e-7,       # iter4=8e-7 → half
    "WEIGHT_DECAY"  : 0.05,
    "NUM_WORKERS"   : 2,
    "SEED"          : 42,
    "CLASS_WEIGHTS" : [0.5, 3.0, 1.5],
    "DICE_WEIGHT"   : 0.5,
    "LABEL_SMOOTH"  : 0.1,        # new: label smoothing on CE
    "ES_PATIENCE"   : 15,
    "ITER4_CKPT"    : "/kaggle/working/iter4/checkpoints/best-epoch=06-floodval/IoU_Flood=0.3106.ckpt",
}

CFG["IMG_DIR"]   = os.path.join(CFG["BASE_DIR"], "image")
CFG["LABEL_DIR"] = os.path.join(CFG["BASE_DIR"], "label")
CFG["PRED_DIR"]  = os.path.join(CFG["BASE_DIR"], "prediction", "image")
CFG["SPLIT_DIR"] = os.path.join(CFG["BASE_DIR"], "split")
CFG["CKPT_DIR"]  = os.path.join(CFG["OUT_DIR"], "checkpoints")
CFG["ENG_IMG_DIR"]  = os.path.join(CFG["DATA_DIR"], "image")
CFG["ENG_LBL_DIR"]  = os.path.join(CFG["DATA_DIR"], "label")
CFG["ENG_PRED_DIR"] = os.path.join(CFG["DATA_DIR"], "prediction", "image")

os.makedirs(CFG["OUT_DIR"],  exist_ok=True)
os.makedirs(CFG["CKPT_DIR"], exist_ok=True)

# Verify engineered data exists
n_img = len(glob.glob(os.path.join(CFG["ENG_IMG_DIR"], "*.tif")))
n_lbl = len(glob.glob(os.path.join(CFG["ENG_LBL_DIR"], "*.tif")))
n_prd = len(glob.glob(os.path.join(CFG["ENG_PRED_DIR"], "*.tif")))
print(f"Engineered data: {n_img} images | {n_lbl} labels | {n_prd} pred patches")
if n_img == 0:
    raise RuntimeError("Run iter4 cell1 first to build engineered TIFs")


# ---------------------------------------------------------------------------
# Norm stats — reuse iter4 values (same engineered channels)
# ---------------------------------------------------------------------------
MEANS = [786.8264, 357.9923, 47.2103, 62.8541, 21.4892, 28.6317]
STDS  = [404.9680, 163.5374, 38.1205, 49.7823, 17.3641, 22.9104]

# Recompute if iter4 norm values aren't available — reads from engineered TIFs
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
            vals = vals[vals > 0] if b < 2 else vals
            for x in vals:
                count[b] += 1
                d = x - mean[b]
                mean[b] += d / count[b]
                M2[b]   += d * (x - mean[b])
    std = np.maximum(np.sqrt(M2 / np.maximum(count - 1, 1)), 1e-6)
    return mean.tolist(), std.tolist()

# Recompute from actual data to be safe
MEANS, STDS = compute_norm_stats(
    CFG["ENG_IMG_DIR"],
    os.path.join(CFG["SPLIT_DIR"], "train.txt"),
    CFG["NUM_BANDS"], CFG["BAND_NAMES"]
)
print("Norm stats (engineered channels):")
for name, m, s in zip(CFG["BAND_NAMES"], MEANS, STDS):
    print(f"  {name:<12} mean={m:.4f}  std={s:.4f}")


# ---------------------------------------------------------------------------
# Loss — CE with label smoothing + Dice
# Label smoothing: instead of hard 0/1 targets, use 0.05/0.95
# This prevents the model from being overconfident on 69 patches
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
# Task
# ---------------------------------------------------------------------------
class FloodSegmentationTask(SemanticSegmentationTask):

    def __init__(self, dice_weight=0.5, lr_backbone=4e-7,
                 label_smooth=0.1, total_steps=2000, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight      = dice_weight
        self.lr_backbone      = lr_backbone
        self.total_steps      = total_steps
        self.val_step_outputs = []
        w = torch.tensor(CFG["CLASS_WEIGHTS"], dtype=torch.float32)
        # label_smoothing built into CrossEntropyLoss since PyTorch 1.10
        self.ce_loss   = nn.CrossEntropyLoss(weight=w, ignore_index=-1,
                                             label_smoothing=label_smooth)
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
