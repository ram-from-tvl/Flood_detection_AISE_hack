# ============================================================
# ITER 2 — CELL 1: Run once per session
# Same as iter1 cell1 but OUT_DIR updated to iter2
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
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
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
# Config — iter2 changes marked with  # <-- iter2
# ---------------------------------------------------------------------------
CFG = {
    "BASE_DIR"      : "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data",
    "OUT_DIR"       : "/kaggle/working/iter2",                  # <-- iter2
    "BAND_NAMES"    : ["SAR_HH", "SAR_HV", "Green", "Red", "NIR", "SWIR"],
    "BANDS"         : [1, 2, 3, 4, 5, 6],
    "NUM_CLASSES"   : 3,
    "NUM_BANDS"     : 6,
    "BACKBONE"      : "prithvi_eo_v2_300_tl",
    "DECODER"       : "UperNetDecoder",
    "DECODER_CH"    : 256,
    "HEAD_DROPOUT"  : 0.1,
    "NECK_INDICES"  : [5, 11, 17, 23],
    "EPOCHS"        : 40,                                       # <-- iter2: more epochs
    "BATCH_SIZE"    : 2,
    "LR"            : 2e-5,                                     # decoder/head LR
    "LR_BACKBONE"   : 2e-6,                                     # <-- iter2: 10x lower for backbone
    "WEIGHT_DECAY"  : 0.05,
    "NUM_WORKERS"   : 2,
    "SEED"          : 42,
    "CLASS_WEIGHTS" : [0.5, 3.0, 1.5],
    "DICE_WEIGHT"   : 0.5,
    "ES_PATIENCE"   : 12,                                       # <-- iter2: slightly more patience
    "LR_PATIENCE"   : 5,                                        # <-- iter2: ReduceLROnPlateau
    "LR_FACTOR"     : 0.5,
    "LR_MIN"        : 1e-7,
    "ITER1_CKPT"    : "/kaggle/working/iter1/checkpoints/best-epoch=07-floodval/IoU_Flood=0.1738.ckpt",
}

CFG["IMG_DIR"]   = os.path.join(CFG["BASE_DIR"], "image")
CFG["LABEL_DIR"] = os.path.join(CFG["BASE_DIR"], "label")
CFG["PRED_DIR"]  = os.path.join(CFG["BASE_DIR"], "prediction", "image")
CFG["SPLIT_DIR"] = os.path.join(CFG["BASE_DIR"], "split")
CFG["CKPT_DIR"]  = os.path.join(CFG["OUT_DIR"], "checkpoints")

os.makedirs(CFG["OUT_DIR"],  exist_ok=True)
os.makedirs(CFG["CKPT_DIR"], exist_ok=True)


# ---------------------------------------------------------------------------
# Norm stats — reuse iter1 values directly (same dataset, no need to recompute)
# ---------------------------------------------------------------------------
MEANS = [786.8264219845304, 357.9923484652695, 1980.9010253751899,
         1828.1133501658953, 1940.4945955290766, 1392.8588503320093]
STDS  = [404.9680257956397, 163.53740120393311, 634.3979490962306,
         616.9742354094914, 594.2206139109147, 526.2242082238249]

print("Normalisation: reused from iter1")
for name, m, s in zip(CFG["BAND_NAMES"], MEANS, STDS):
    print(f"  {name:<10} mean={m:.2f}  std={s:.2f}")


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
# Task — iter2: two-phase LR groups + ReduceLROnPlateau
# ---------------------------------------------------------------------------
class FloodSegmentationTask(SemanticSegmentationTask):

    def __init__(self, dice_weight=0.5, lr_backbone=2e-6,
                 lr_patience=5, lr_factor=0.5, lr_min=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight      = dice_weight
        self.lr_backbone      = lr_backbone
        self.lr_patience      = lr_patience
        self.lr_factor        = lr_factor
        self.lr_min           = lr_min
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
        # Separate param groups: backbone gets 10x lower LR
        backbone_params = list(self.model.encoder.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        other_params    = [p for p in self.parameters() if id(p) not in backbone_ids]

        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.lr_backbone},
            {"params": other_params,    "lr": CFG["LR"]},
        ], weight_decay=CFG["WEIGHT_DECAY"])

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=self.lr_patience,
            factor=self.lr_factor, min_lr=self.lr_min,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor":   "val/mIoU",
                "interval":  "epoch",
                "frequency": 1,
            },
        }


print("\nCell 1 done — ready for iter2 cell 2.")
