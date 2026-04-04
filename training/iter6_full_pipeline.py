import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "terratorch", "albumentations", "rasterio", "numpy==2.2.0", "-q"])

import os, glob, zipfile, warnings
import numpy as np
import pandas as pd
import torch
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
# Paths — exactly as helper
# ---------------------------------------------------------------------------
BASE_DIR  = "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data"
OUT_DIR   = "/kaggle/working/iter6"
SPLIT_DIR = os.path.join(BASE_DIR, "split")

os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# Two changes from helper based on EDA:
#   1. CLASS_WEIGHTS: helper had [0.1,0.1,0.74] — WaterBody weighted highest,
#      but Flood is the eval metric and harder class. Fixed to weight Flood most.
#   2. SelectIndices neck added — UperNet needs multi-scale features from
#      4 transformer depths, not just the final layer.
# ---------------------------------------------------------------------------
EPOCHS        = 40
BATCH_SIZE    = 2
LR            = 2e-5
WEIGHT_DECAY  = 0.05
HEAD_DROPOUT  = 0.1
FREEZE_BACKBONE = False
BANDS         = [1, 2, 3, 4, 5, 6]
NUM_FRAMES    = 1
CLASS_WEIGHTS = [0.3, 5.0, 1.0]   # NoFlood, Flood, WaterBody
SEED          = 42

# Helper's norm stats (original 6 bands)
MEANS = [841.1257, 371.6175, 1734.1789, 1588.3142, 1742.8452, 1218.5551]
STDS  = [473.7090, 170.3611,  623.0474,  612.8465,  564.5835,  528.0894]

# ---------------------------------------------------------------------------
# Datamodule — identical to helper
# ---------------------------------------------------------------------------
datamodule = terratorch.datamodules.GenericNonGeoSegmentationDataModule(
    batch_size=BATCH_SIZE,
    num_workers=2,
    num_classes=3,
    train_data_root=os.path.join(BASE_DIR, "image"),
    train_label_data_root=os.path.join(BASE_DIR, "label"),
    val_data_root=os.path.join(BASE_DIR, "image"),
    val_label_data_root=os.path.join(BASE_DIR, "label"),
    test_data_root=os.path.join(BASE_DIR, "image"),
    test_label_data_root=os.path.join(BASE_DIR, "label"),
    train_split=os.path.join(SPLIT_DIR, "train.txt"),
    val_split=os.path.join(SPLIT_DIR, "val.txt"),
    test_split=os.path.join(SPLIT_DIR, "test.txt"),
    predict_data_root=os.path.join(BASE_DIR, "prediction", "image"),
    predict_split=os.path.join(SPLIT_DIR, "pred.txt"),
    img_grep="*image.tif",
    label_grep="*label.tif",
    train_transform=[albumentations.D4(), ToTensorV2()],
    val_transform=None,
    test_transform=None,
    means=MEANS,
    stds=STDS,
    no_data_replace=0,
    no_label_replace=-1,
)
datamodule.setup("fit")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
pl.seed_everything(SEED)

model = SemanticSegmentationTask(
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
    lr=LR,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
    ignore_index=-1,
    freeze_backbone=FREEZE_BACKBONE,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

if hasattr(model.model.encoder, "set_grad_checkpointing"):
    model.model.encoder.set_grad_checkpointing(True)

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
# Predict — direct inference, no trainer.predict tuple issues
# ---------------------------------------------------------------------------
pred_files = sorted(glob.glob(
    os.path.join(BASE_DIR, "prediction", "image", "*.tif")))
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
        pid = os.path.basename(fp).replace("_image.tif","").replace(".tif","")
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
    return " ".join(f"{s} {l}" for s,l in zip(starts, lengths))

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
sub_csv = os.path.join(OUT_DIR, "submission_iter6.csv")
sub_df.to_csv(sub_csv, index=False)
print(f"Submission: {len(sub_df)} rows | empty: {(sub_df['rle_mask']=='0 0').sum()}")

zip_path = "/kaggle/working/submission_iter6.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub_csv, "submission_iter6.csv")
print(f"Submit: {zip_path}")
