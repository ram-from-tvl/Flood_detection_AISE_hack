"""
Iter 9 — EDA-grounded improvements on iter8's winning config

EDA findings acted on:
  1. Boundary confusion: 60-80% of flood pixels are adjacent to water body pixels.
     Both look identical in raw SAR. Standard CE weights all pixels equally.
     Fix: boundary-aware loss — dilate the flood/waterbody boundary mask and
     upweight those pixels 3x in the CE loss. Forces the model to focus on
     the hard zone.

  2. Test-time augmentation (TTA): free accuracy at inference.
     Run 8 forward passes (D4 group: 4 rotations × 2 flips), average softmax.
     No retraining cost. Reduces prediction variance on boundary pixels.

  3. Two T4 GPUs: devices=2, strategy=ddp, batch=4 per GPU (effective=8)

Everything else identical to iter8 (original channels, SelectIndices, trainval).
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "terratorch", "albumentations", "rasterio", "numpy==2.2.0", "-q"])

import os, glob, zipfile, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from scipy.ndimage import binary_dilation
import albumentations
from albumentations.pytorch import ToTensorV2
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import terratorch
from terratorch.tasks import SemanticSegmentationTask

warnings.filterwarnings("ignore")
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data"
OUT_DIR   = "/kaggle/working/iter9"
SPLIT_DIR = os.path.join(BASE_DIR, "split")

os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS          = 50
BATCH_SIZE      = 4      # per GPU; 2 GPUs → effective batch = 8
LR_DECODER      = 2e-5
LR_BACKBONE     = 2e-6
WEIGHT_DECAY    = 0.05
HEAD_DROPOUT    = 0.1
BANDS           = [1, 2, 3, 4, 5, 6]
NUM_FRAMES      = 1
CLASS_WEIGHTS   = [0.3, 5.0, 1.0]
BOUNDARY_WEIGHT = 3.0    # upweight boundary pixels this many times
SEED            = 42

MEANS = [841.1257, 371.6175, 1734.1789, 1588.3142, 1742.8452, 1218.5551]
STDS  = [473.7090, 170.3611,  623.0474,  612.8465,  564.5835,  528.0894]

# ---------------------------------------------------------------------------
# trainval.txt — all 69 patches
# ---------------------------------------------------------------------------
trainval_path = os.path.join(OUT_DIR, "trainval.txt")
if not os.path.exists(trainval_path):
    ids = []
    for fname in ["train.txt", "val.txt"]:
        with open(os.path.join(SPLIT_DIR, fname)) as f:
            ids += [l.strip() for l in f if l.strip()]
    with open(trainval_path, "w") as f:
        f.write("\n".join(ids))
    print(f"trainval.txt: {len(ids)} patches")

# ---------------------------------------------------------------------------
# Boundary-aware loss
# EDA showed 60-80% of flood pixels are adjacent to water body pixels.
# We upweight those boundary pixels so the model focuses on the hard zone.
#
# Implementation: for each batch, compute a pixel weight map where pixels
# at the flood/waterbody boundary get BOUNDARY_WEIGHT, others get 1.0.
# Pass this as the 'weight' argument to F.cross_entropy (per-pixel weights).
# ---------------------------------------------------------------------------
def compute_boundary_weights(targets, boundary_weight=3.0):
    """
    targets: (B, H, W) long tensor, values 0/1/2/-1
    Returns: (B, H, W) float weight map
    """
    B, H, W = targets.shape
    weights = torch.ones(B, H, W, dtype=torch.float32, device=targets.device)

    for b in range(B):
        t = targets[b].cpu().numpy()
        flood_mask = (t == 1).astype(np.uint8)
        water_mask = (t == 2).astype(np.uint8)

        # Dilate each mask by 2 pixels and find overlap zone
        flood_dilated = binary_dilation(flood_mask, iterations=2).astype(np.uint8)
        water_dilated = binary_dilation(water_mask, iterations=2).astype(np.uint8)
        boundary = (flood_dilated & water_dilated).astype(np.float32)

        weights[b] = torch.tensor(
            np.where(boundary > 0, boundary_weight, 1.0),
            dtype=torch.float32
        )
    return weights


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


class FloodTask(SemanticSegmentationTask):

    def __init__(self, boundary_weight=3.0, **kwargs):
        super().__init__(**kwargs)
        self.boundary_weight = boundary_weight
        self.dice_loss = MulticlassDiceLoss(num_classes=3, ignore_index=-1)
        w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
        self.ce_base = nn.CrossEntropyLoss(weight=w, ignore_index=-1, reduction="none")

    def _boundary_ce(self, logits, targets):
        # Per-pixel CE loss
        self.ce_base.weight = self.ce_base.weight.to(logits.device)
        per_pixel = self.ce_base(logits, targets)          # (B, H, W)

        # Boundary weight map
        bw = compute_boundary_weights(targets, self.boundary_weight).to(logits.device)

        # Mask out ignore_index pixels
        valid = (targets != -1).float()
        loss = (per_pixel * bw * valid).sum() / (valid.sum() + 1e-6)
        return loss

    def training_step(self, batch, batch_idx):
        y = batch["mask"].long()
        if y.ndim == 4: y = y.squeeze(1)
        logits = self.model(batch["image"]).output
        ce   = self._boundary_ce(logits, y)
        dice = self.dice_loss(logits, y)
        loss = 0.5 * ce + 0.5 * dice
        self.log("train/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["mask"].long()
        if y.ndim == 4: y = y.squeeze(1)
        logits = self.model(batch["image"]).output
        ce   = self._boundary_ce(logits, y)
        dice = self.dice_loss(logits, y)
        loss = 0.5 * ce + 0.5 * dice
        preds = logits.argmax(dim=1)

        valid   = (y != -1)
        iou_sum = 0.0
        for c, name in enumerate(["NoFlood", "Flood", "WaterBody"]):
            tp  = ((preds == c) & (y == c) & valid).sum().float()
            fp  = ((preds == c) & (y != c) & valid).sum().float()
            fn  = ((preds != c) & (y == c) & valid).sum().float()
            iou = (tp / (tp + fp + fn + 1e-6)).item()
            iou_sum += iou
            self.log(f"val/IoU_{name}", iou, on_epoch=True,
                     prog_bar=(name == "Flood"), sync_dist=True)
        self.log("val/mIoU", iou_sum / 3, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        backbone_params = list(self.model.encoder.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        other_params    = [p for p in self.parameters() if id(p) not in backbone_ids]
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": other_params,    "lr": LR_DECODER},
        ], weight_decay=WEIGHT_DECAY)
        total_steps = EPOCHS * 18  # ~18 steps/epoch (69 patches, batch=4, accum=1, 2 GPUs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=1e-8
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


# ---------------------------------------------------------------------------
# Datamodule
# ---------------------------------------------------------------------------
pl.seed_everything(SEED)

datamodule = terratorch.datamodules.GenericNonGeoSegmentationDataModule(
    batch_size=BATCH_SIZE,
    num_workers=2,
    num_classes=3,
    train_data_root=os.path.join(BASE_DIR, "image"),
    train_label_data_root=os.path.join(BASE_DIR, "label"),
    val_data_root=os.path.join(BASE_DIR, "image"),
    val_label_data_root=os.path.join(BASE_DIR, "label"),
    train_split=trainval_path,
    val_split=trainval_path,
    predict_data_root=os.path.join(BASE_DIR, "prediction", "image"),
    predict_split=os.path.join(SPLIT_DIR, "pred.txt"),
    img_grep="*image.tif",
    label_grep="*label.tif",
    train_transform=[albumentations.D4(), ToTensorV2()],
    val_transform=None,
    means=MEANS,
    stds=STDS,
    no_data_replace=0,
    no_label_replace=-1,
)
datamodule.setup("fit")
s = next(iter(datamodule.train_dataloader()))
print(f"DataModule: image={s['image'].shape}  mask={s['mask'].shape}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = FloodTask(
    boundary_weight=BOUNDARY_WEIGHT,
    model_args=dict(
        backbone_pretrained=True,
        backbone="prithvi_eo_v2_300_tl",
        backbone_bands=BANDS,
        backbone_num_frames=NUM_FRAMES,
        decoder="UperNetDecoder",
        decoder_channels=256,
        decoder_scale_modules=True,
        num_classes=3,
        head_dropout=HEAD_DROPOUT,
        necks=[
            dict(name="SelectIndices", indices=[5, 11, 17, 23]),
            dict(name="ReshapeTokensToImage", effective_time_dim=NUM_FRAMES),
        ],
        rescale=True,
    ),
    plot_on_val=False,
    class_weights=CLASS_WEIGHTS,
    loss="ce",
    lr=LR_DECODER,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
    ignore_index=-1,
    freeze_backbone=False,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

if hasattr(model.model.encoder, "set_grad_checkpointing"):
    model.model.encoder.set_grad_checkpointing(True)

print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

# ---------------------------------------------------------------------------
# Trainer — 2 GPUs
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ckpt_cb = ModelCheckpoint(
    monitor="val/mIoU", mode="max",
    dirpath=os.path.join(OUT_DIR, "checkpoints"),
    filename="best-ep{epoch:02d}-miou{val_mIoU:.4f}",
    save_top_k=1, save_last=True,
    auto_insert_metric_name=False,
)

trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    precision="32",
    accumulate_grad_batches=2,
    max_epochs=EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=5,
    gradient_clip_val=1.0,
    callbacks=[ckpt_cb],
    enable_progress_bar=True,
)

trainer.fit(model, datamodule=datamodule)
best_ckpt = ckpt_cb.best_model_path or ckpt_cb.last_model_path
print(f"Best ckpt: {best_ckpt}  |  val/mIoU: {ckpt_cb.best_model_score}")

# ---------------------------------------------------------------------------
# Save checkpoint zip (rank 0 only)
# ---------------------------------------------------------------------------
if True:  # rank 0 only needed for DDP; single GPU always saves
    ckpt_zip = "/kaggle/working/iter9_checkpoints.zip"
    with zipfile.ZipFile(ckpt_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in glob.glob(os.path.join(OUT_DIR, "checkpoints", "*.ckpt")):
            zf.write(c, os.path.basename(c))
    print(f"Checkpoints: {ckpt_zip} ({os.path.getsize(ckpt_zip)/1024/1024:.0f} MB)")

# ---------------------------------------------------------------------------
# Predict with Test-Time Augmentation (TTA)
# D4 group: identity, rot90, rot180, rot270, + horizontal flip of each = 8 passes
# Average softmax probabilities across all 8 passes before argmax
# ---------------------------------------------------------------------------
pred_files = sorted(glob.glob(os.path.join(BASE_DIR, "prediction", "image", "*.tif")))
print(f"\nTTA inference on {len(pred_files)} patches (8 passes each)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move to single GPU for inference — DDP not needed here
model_inf = model.module if hasattr(model, "module") else model
model_inf.eval()
model_inf = model_inf.to(device)
if best_ckpt and os.path.exists(best_ckpt):
    state = torch.load(best_ckpt, map_location=device)
    model_inf.load_state_dict(state["state_dict"], strict=False)

def tta_predict(img_tensor, model_fn):
    """
    img_tensor: (1, 6, H, W)
    Returns averaged softmax (1, 3, H, W)
    """
    probs = torch.zeros(1, 3, img_tensor.shape[2], img_tensor.shape[3],
                        device=img_tensor.device)
    for k in range(4):
        # rotate k*90 degrees
        x = torch.rot90(img_tensor, k, dims=[2, 3])
        p = torch.softmax(model_fn(x).output, dim=1)
        # rotate prediction back
        probs += torch.rot90(p, -k, dims=[2, 3])

        # horizontal flip
        xf = torch.flip(x, dims=[3])
        pf = torch.softmax(model_fn(xf).output, dim=1)
        probs += torch.flip(torch.rot90(pf, -k, dims=[2, 3]), dims=[3])

    return probs / 8.0

pred_map = {}
with torch.no_grad():
    for fp in pred_files:
        pid = os.path.basename(fp).replace("_image.tif", "").replace(".tif", "")
        with rasterio.open(fp) as src:
            img = src.read().astype(np.float32)
        for b in range(6):
            img[b] = (img[b] - MEANS[b]) / STDS[b]
        img_t = torch.tensor(img).unsqueeze(0).to(device)
        probs = tta_predict(img_t, model_inf.model)
        pred_map[pid] = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype("int16")

print(f"TTA predictions done: {len(pred_map)} patches")

# ---------------------------------------------------------------------------
# RLE + submission
# ---------------------------------------------------------------------------
def mask_to_rle(mask):
    flat = mask.flatten(order="F").astype(np.uint8)
    if flat.sum() == 0: return "0 0"
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
sub_csv = os.path.join(OUT_DIR, "submission_iter9.csv")
sub_df.to_csv(sub_csv, index=False)
print(f"Submission: {len(sub_df)} rows | empty: {(sub_df['rle_mask']=='0 0').sum()}")

zip_path = "/kaggle/working/submission_iter9.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub_csv, "submission_iter9.csv")
print(f"Submit: {zip_path}")
