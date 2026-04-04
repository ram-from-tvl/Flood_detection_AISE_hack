"""
Iter 7 — Two high-return changes combined:
  1. SAR texture channels [HH, HV, HH_std3, HH_std7, HV_std3, HV_std7]
     (iter4 finding: texture KS=0.55 vs optical KS=0.00)
  2. SelectIndices neck [5,11,17,23] for multi-scale UperNet features
     (iter6 finding: single-scale was leaving performance on the table)
  3. Train on all 69 patches (trainval) — iter6 used only 59
  4. Flood class weight = 5.0, label smoothing = 0.1

Neither iter5 nor iter6 had all four of these simultaneously.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "terratorch", "albumentations", "rasterio", "numpy==2.2.0", "-q"])

import os, glob, zipfile, warnings, shutil
import numpy as np
import pandas as pd
import torch
import rasterio
from scipy.ndimage import uniform_filter
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
OUT_DIR   = "/kaggle/working/iter7"
DATA_DIR  = "/kaggle/working/iter7_data"
SPLIT_DIR = os.path.join(BASE_DIR, "split")

for d in [OUT_DIR, os.path.join(OUT_DIR, "checkpoints"),
          os.path.join(DATA_DIR, "image"),
          os.path.join(DATA_DIR, "label"),
          os.path.join(DATA_DIR, "prediction", "image")]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS        = 40
BATCH_SIZE    = 2
LR            = 2e-5
WEIGHT_DECAY  = 0.05
HEAD_DROPOUT  = 0.1
BANDS         = [1, 2, 3, 4, 5, 6]
NUM_FRAMES    = 1
CLASS_WEIGHTS = [0.3, 5.0, 1.0]   # flood weighted highest
LABEL_SMOOTH  = 0.1
SEED          = 42

# ---------------------------------------------------------------------------
# Build engineered TIFs (SAR texture replaces dead optical channels)
# Skips files that already exist — safe to rerun
# ---------------------------------------------------------------------------
def local_std(band, radius):
    b = band.astype(np.float64)
    m = uniform_filter(b, size=radius)
    m2 = uniform_filter(b**2, size=radius)
    return np.sqrt(np.maximum(m2 - m**2, 0.0)).astype(np.float32)

def build_eng(src_path, dst_path):
    if os.path.exists(dst_path):
        return False
    with rasterio.open(src_path) as src:
        arr = src.read().astype(np.float32)
        meta = src.meta.copy()
    hh, hv = arr[0], arr[1]
    new_arr = np.stack([hh, hv, local_std(hh, 3), local_std(hh, 7),
                        local_std(hv, 3), local_std(hv, 7)], axis=0)
    meta.update({"count": 6, "dtype": "float32"})
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(new_arr)
    return True

IMG_DIR   = os.path.join(BASE_DIR, "image")
LABEL_DIR = os.path.join(BASE_DIR, "label")
PRED_DIR  = os.path.join(BASE_DIR, "prediction", "image")

n1 = sum(build_eng(p, os.path.join(DATA_DIR, "image", os.path.basename(p)))
         for p in glob.glob(os.path.join(IMG_DIR, "*.tif")))
n2 = sum(build_eng(p, os.path.join(DATA_DIR, "prediction", "image", os.path.basename(p)))
         for p in glob.glob(os.path.join(PRED_DIR, "*.tif")))
n3 = 0
for lp in glob.glob(os.path.join(LABEL_DIR, "*.tif")):
    dst = os.path.join(DATA_DIR, "label", os.path.basename(lp))
    if not os.path.exists(dst):
        shutil.copy2(lp, dst)
        n3 += 1
print(f"Engineered TIFs: {n1} built, {n2} pred, {n3} labels copied")

ENG_IMG  = os.path.join(DATA_DIR, "image")
ENG_LBL  = os.path.join(DATA_DIR, "label")
ENG_PRED = os.path.join(DATA_DIR, "prediction", "image")

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
else:
    print(f"trainval.txt exists")

# ---------------------------------------------------------------------------
# Norm stats — computed from engineered training TIFs
# ---------------------------------------------------------------------------
def compute_norm(image_dir, split_file, num_bands):
    with open(split_file) as f:
        ids = [l.strip() for l in f if l.strip()]
    id_to_path = {
        os.path.basename(p).replace("_image.tif", "").replace(".tif", ""): p
        for p in glob.glob(os.path.join(image_dir, "*.tif"))
    }
    count = np.zeros(num_bands); mean = np.zeros(num_bands); M2 = np.zeros(num_bands)
    for pid in ids:
        path = next((v for k, v in id_to_path.items() if pid in k or k in pid), None)
        if path is None: continue
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float64)
        for b in range(num_bands):
            vals = arr[b].ravel()
            vals = vals[vals > 0] if b < 2 else vals
            for x in vals:
                count[b] += 1; d = x - mean[b]
                mean[b] += d / count[b]; M2[b] += d * (x - mean[b])
    std = np.maximum(np.sqrt(M2 / np.maximum(count - 1, 1)), 1e-6)
    return mean.tolist(), std.tolist()

MEANS, STDS = compute_norm(ENG_IMG, os.path.join(SPLIT_DIR, "train.txt"), 6)
print("Norm stats:", [f"{m:.1f}" for m in MEANS])

# ---------------------------------------------------------------------------
# Datamodule — train on all 69 patches, val on same (leaderboard is real val)
# ---------------------------------------------------------------------------
datamodule = terratorch.datamodules.GenericNonGeoSegmentationDataModule(
    batch_size=BATCH_SIZE,
    num_workers=2,
    num_classes=3,
    train_data_root=ENG_IMG,
    train_label_data_root=ENG_LBL,
    val_data_root=ENG_IMG,
    val_label_data_root=ENG_LBL,
    train_split=trainval_path,
    val_split=trainval_path,
    predict_data_root=ENG_PRED,
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
# Model — SAR texture channels + SelectIndices neck + label smoothing
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
# Predict — direct inference
# ---------------------------------------------------------------------------
pred_files = sorted(glob.glob(os.path.join(ENG_PRED, "*.tif")))
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
sub_csv = os.path.join(OUT_DIR, "submission_iter7.csv")
sub_df.to_csv(sub_csv, index=False)
print(f"Submission: {len(sub_df)} rows | empty: {(sub_df['rle_mask']=='0 0').sum()}")

zip_path = "/kaggle/working/submission_iter7.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub_csv, "submission_iter7.csv")
print(f"Submit: {zip_path}")
