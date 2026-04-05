"""
Iter 8 — Back to what worked: original channels + SelectIndices + trainval

Analysis of what happened:
  iter5: texture channels, no SelectIndices → improved LB
  iter6: original channels + SelectIndices → improved LB further
  iter7: texture channels + SelectIndices + trainval → WORSE

Conclusion: texture channel engineering hurts when combined with SelectIndices.
Prithvi's patch embedding needs to relearn from scratch for texture channels,
and 69 patches is not enough data for that. Original channels let the pretrained
patch embedding work as intended.

Iter 8 = iter6's winning formula + two additions:
  1. Train on all 69 patches (trainval) — iter6 used only 59
  2. Two-phase LR: backbone 10x lower than decoder — protects pretrained features
  3. More epochs (50) with cosine decay
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
OUT_DIR   = "/kaggle/working/iter8"
SPLIT_DIR = os.path.join(BASE_DIR, "split")

os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters — original channels, same as iter6 but with trainval + 2-phase LR
# ---------------------------------------------------------------------------
EPOCHS        = 50
BATCH_SIZE    = 4    # 4 per GPU × 2 GPUs = 8 effective batch
LR_DECODER    = 2e-5
LR_BACKBONE   = 2e-6    # 10x lower — protects pretrained Prithvi features
WEIGHT_DECAY  = 0.05
HEAD_DROPOUT  = 0.1
BANDS         = [1, 2, 3, 4, 5, 6]
NUM_FRAMES    = 1
CLASS_WEIGHTS = [0.3, 5.0, 1.0]
SEED          = 42

# Helper's norm stats — original 6 bands, same as iter6
MEANS = [841.1257, 371.6175, 1734.1789, 1588.3142, 1742.8452, 1218.5551]
STDS  = [473.7090, 170.3611,  623.0474,  612.8465,  564.5835,  528.0894]

# ---------------------------------------------------------------------------
# trainval.txt — all 69 labelled patches
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
# Custom task with two-phase LR + CE+Dice loss
# Subclassing SemanticSegmentationTask to override configure_optimizers only
# ---------------------------------------------------------------------------
class FloodTask(SemanticSegmentationTask):

    def configure_optimizers(self):
        backbone_params = list(self.model.encoder.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        other_params    = [p for p in self.parameters() if id(p) not in backbone_ids]

        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": other_params,    "lr": LR_DECODER},
        ], weight_decay=WEIGHT_DECAY)

        # Cosine decay over total steps — smooth, no plateau dependency
        total_steps = EPOCHS * 35  # ~35 steps/epoch for 69 patches, batch=2, accum=2
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=1e-8
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

# ---------------------------------------------------------------------------
# Datamodule — original channels, train on all 69 patches
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
# Model — original channels + SelectIndices neck (iter6's winning config)
# ---------------------------------------------------------------------------
model = FloodTask(
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
# Trainer
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
# Save checkpoint zip immediately after training
# ---------------------------------------------------------------------------
# Only rank 0 saves the zip (DDP runs this code on both GPUs)
if trainer.global_rank == 0:
    ckpt_zip = "/kaggle/working/iter8_checkpoints.zip"
    with zipfile.ZipFile(ckpt_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in glob.glob(os.path.join(OUT_DIR, "checkpoints", "*.ckpt")):
            zf.write(c, os.path.basename(c))
    print(f"Checkpoints zipped: {ckpt_zip} ({os.path.getsize(ckpt_zip)/1024/1024:.0f} MB)")

# ---------------------------------------------------------------------------
# Predict — direct inference
# ---------------------------------------------------------------------------
pred_files = sorted(glob.glob(os.path.join(BASE_DIR, "prediction", "image", "*.tif")))
print(f"\nInference on {len(pred_files)} patches...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model = model.to(device)
if best_ckpt and os.path.exists(best_ckpt):
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["state_dict"], strict=False)

pred_map = {}
with torch.no_grad():
    for fp in pred_files:
        pid = os.path.basename(fp).replace("_image.tif", "").replace(".tif", "")
        with rasterio.open(fp) as src:
            img = src.read().astype(np.float32)
        for b in range(6):
            img[b] = (img[b] - MEANS[b]) / STDS[b]
        out = model.model(torch.tensor(img).unsqueeze(0).to(device)).output
        pred_map[pid] = out.argmax(dim=1).squeeze(0).cpu().numpy().astype("int16")

print(f"Predicted {len(pred_map)} patches")

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
sub_csv = os.path.join(OUT_DIR, "submission_iter8.csv")
sub_df.to_csv(sub_csv, index=False)
print(f"Submission: {len(sub_df)} rows | empty: {(sub_df['rle_mask']=='0 0').sum()}")

zip_path = "/kaggle/working/submission_iter8.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub_csv, "submission_iter8.csv")
print(f"Submit: {zip_path}")
