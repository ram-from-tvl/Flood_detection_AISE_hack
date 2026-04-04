# ============================================================
# ITER 3 — CELL 1: Run once per session
# ============================================================

import subprocess, sys

# Restore numpy 2.x — a previous run may have downgraded it to 1.26.x
subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=2.0", "-q"],
               capture_output=True)

subprocess.run([sys.executable, "-m", "pip", "install",
                "terratorch", "albumentations", "rasterio",
                "numpy==2.2.0",
                "-q"])

# Hard-verify numpy version before any imports
import importlib, numpy as _np_check
if tuple(int(x) for x in _np_check.__version__.split(".")[:2]) < (2, 0):
    raise RuntimeError(
        f"numpy {_np_check.__version__} is too old — restart the kernel and rerun"
    )

import os, glob, zipfile, json, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
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
# Config
# Iter3 changes vs iter2:
#   LR_DECODER   2e-5  → 8e-6   (reduce oscillation)
#   LR_BACKBONE  2e-6  → 8e-7   (reduce oscillation)
#   EPOCHS       40    → 50     (still improving at epoch 21)
#   Scheduler    ReduceLROnPlateau → CosineAnnealing
#                (plateau scheduler unreliable with only 10 noisy val patches)
#   TRAIN_SPLIT  train.txt → trainval.txt (all 69 labelled patches)
#                10 val patches gives a ±0.05 noisy IoU estimate; leaderboard
#                is the true validation. More data is more valuable.
#   ITER2_CKPT   warm-start from iter2 best checkpoint
# ---------------------------------------------------------------------------
CFG = {
    "BASE_DIR"      : "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data",
    "OUT_DIR"       : "/kaggle/working/iter3",
    "BAND_NAMES"    : ["SAR_HH", "SAR_HV", "Green", "Red", "NIR", "SWIR"],
    "BANDS"         : [1, 2, 3, 4, 5, 6],
    "NUM_CLASSES"   : 3,
    "NUM_BANDS"     : 6,
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
    # Auto-discover iter2 best checkpoint — finds whatever filename was saved
    "ITER2_CKPT"    : next(iter(sorted(glob.glob(
        "/kaggle/working/iter2/checkpoints/best-*.ckpt"))), None),
}

CFG["IMG_DIR"]    = os.path.join(CFG["BASE_DIR"], "image")
CFG["LABEL_DIR"]  = os.path.join(CFG["BASE_DIR"], "label")
CFG["PRED_DIR"]   = os.path.join(CFG["BASE_DIR"], "prediction", "image")
CFG["SPLIT_DIR"]  = os.path.join(CFG["BASE_DIR"], "split")
CFG["CKPT_DIR"]   = os.path.join(CFG["OUT_DIR"], "checkpoints")

os.makedirs(CFG["OUT_DIR"],  exist_ok=True)
os.makedirs(CFG["CKPT_DIR"], exist_ok=True)


# ---------------------------------------------------------------------------
# Write trainval.txt — all 69 labelled patches combined
# This is created once at setup time from train.txt + val.txt
# ---------------------------------------------------------------------------
trainval_path = os.path.join(CFG["OUT_DIR"], "trainval.txt")
if not os.path.exists(trainval_path):
    train_ids, val_ids = [], []
    with open(os.path.join(CFG["SPLIT_DIR"], "train.txt")) as f:
        train_ids = [l.strip() for l in f if l.strip()]
    with open(os.path.join(CFG["SPLIT_DIR"], "val.txt")) as f:
        val_ids = [l.strip() for l in f if l.strip()]
    all_ids = train_ids + val_ids
    with open(trainval_path, "w") as f:
        f.write("\n".join(all_ids))
    print(f"Created trainval.txt with {len(all_ids)} patches "
          f"({len(train_ids)} train + {len(val_ids)} val)")
else:
    with open(trainval_path) as f:
        n = sum(1 for l in f if l.strip())
    print(f"trainval.txt exists: {n} patches")

CFG["TRAINVAL_SPLIT"] = trainval_path


# ---------------------------------------------------------------------------
# Norm stats — reused from iter1 (same dataset)
# ---------------------------------------------------------------------------
MEANS = [786.8264, 357.9923, 1980.9010, 1828.1134, 1940.4946, 1392.8589]
STDS  = [404.9680, 163.5374, 634.3979,  616.9742,  594.2206,  526.2242]

print("Normalisation (reused from iter1):")
for name, m, s in zip(CFG["BAND_NAMES"], MEANS, STDS):
    print(f"  {name:<10} mean={m:.2f}  std={s:.2f}")


# ---------------------------------------------------------------------------
# Dice loss
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
# Training on all 69 patches means we have no val split, so Lightning will
# run validation on the same data as training (we point val_split at trainval.txt
# too). This is acceptable — the leaderboard is our real validation.
# IoU is still computed correctly in on_validation_epoch_end for monitoring.
# ---------------------------------------------------------------------------
class FloodSegmentationTask(SemanticSegmentationTask):

    def __init__(self, dice_weight=0.5, lr_backbone=8e-7, total_steps=1500, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight      = dice_weight
        self.lr_backbone      = lr_backbone
        self.total_steps      = total_steps   # for cosine scheduler
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
        backbone_params = list(self.model.encoder.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        other_params    = [p for p in self.parameters() if id(p) not in backbone_ids]

        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.lr_backbone},
            {"params": other_params,    "lr": CFG["LR_DECODER"]},
        ], weight_decay=CFG["WEIGHT_DECAY"])

        # CosineAnnealingLR: smooth predictable decay over total_steps.
        # Does not depend on a noisy val metric unlike ReduceLROnPlateau.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.total_steps, eta_min=1e-8
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


print("\nCell 1 done.")