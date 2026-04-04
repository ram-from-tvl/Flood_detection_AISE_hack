# ============================================================
# ITER 4 — CELL 2: Train + Predict
# Requires Cell 1 to have been run first.
# ============================================================

# ---------------------------------------------------------------------------
# Datamodule — points at engineered TIFs, not the original data
# ---------------------------------------------------------------------------
datamodule = GenericNonGeoSegmentationDataModule(
    train_data_root       = CFG["ENG_IMG_DIR"],
    train_label_data_root = CFG["ENG_LBL_DIR"],
    val_data_root         = CFG["ENG_IMG_DIR"],
    val_label_data_root   = CFG["ENG_LBL_DIR"],
    predict_data_root     = CFG["ENG_PRED_DIR"],
    train_split    = CFG["TRAINVAL_SPLIT"],
    val_split      = CFG["TRAINVAL_SPLIT"],
    predict_split  = os.path.join(CFG["SPLIT_DIR"], "pred.txt"),
    img_grep       = "*_image.tif",
    label_grep     = "*_label.tif",
    batch_size     = CFG["BATCH_SIZE"],
    num_workers    = CFG["NUM_WORKERS"],
    num_classes    = CFG["NUM_CLASSES"],
    means          = MEANS,
    stds           = STDS,
    train_transform= [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ],
    val_transform  = None,
    test_transform = None,
    no_data_replace  = 0,
    no_label_replace = -1,
)

datamodule.setup("fit")
s = next(iter(datamodule.train_dataloader()))
steps_per_epoch = len(datamodule.train_dataloader())
total_steps     = CFG["EPOCHS"] * steps_per_epoch
sys.__stdout__.write(
    f"DataModule: image={s['image'].shape}  mask={s['mask'].shape}\n"
    f"Steps/epoch={steps_per_epoch}  Total steps={total_steps}\n"
)


# ---------------------------------------------------------------------------
# Model — Prithvi with 6-channel engineered input
# backbone_pretrained=True: Transformer layers load pretrained weights fully.
# Patch embedding adapts to new channel stats via fine-tuning.
# backbone_bands=[1,2,3,4,5,6]: tells Prithvi to expect 6 input channels.
# ---------------------------------------------------------------------------
pl.seed_everything(CFG["SEED"])

model = FloodSegmentationTask(
    model_args = dict(
        backbone              = CFG["BACKBONE"],
        backbone_pretrained   = True,
        backbone_bands        = CFG["BANDS"],   # 6 channels, same count as before
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
    lr_backbone       = CFG["LR_BACKBONE"],
    total_steps       = total_steps,
)

if hasattr(model.model.encoder, "set_grad_checkpointing"):
    model.model.encoder.set_grad_checkpointing(True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
sys.__stdout__.write(f"Model: {trainable/1e6:.0f}M trainable\n")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

ckpt = ModelCheckpoint(
    monitor="val/IoU_Flood", mode="max",
    dirpath=CFG["CKPT_DIR"],
    filename="best-{epoch:02d}-flood{val/IoU_Flood:.4f}",
    save_top_k=1, save_last=False, verbose=True,
)

trainer = pl.Trainer(
    accelerator             = "auto",
    devices                 = 1,
    precision               = "32",
    accumulate_grad_batches = 2,
    max_epochs              = CFG["EPOCHS"],
    logger                  = CSVLogger(CFG["OUT_DIR"], name="logs"),
    callbacks               = [ckpt, LearningRateMonitor(logging_interval="epoch")],
    check_val_every_n_epoch = 1,
    log_every_n_steps       = 5,
    gradient_clip_val       = 1.0,
    enable_progress_bar     = True,
)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
sys.__stdout__.write(
    f"\nTraining iter4: {CFG['EPOCHS']} epochs | "
    f"lr_decoder={CFG['LR_DECODER']} | lr_backbone={CFG['LR_BACKBONE']}\n"
    f"Input channels: {CFG['BAND_NAMES']}\n"
)
t0 = datetime.now()
trainer.fit(model, datamodule=datamodule)
duration  = datetime.now() - t0
best_ckpt = ckpt.best_model_path
sys.__stdout__.write(f"\nDone in {duration}\n")
sys.__stdout__.write(f"  Best val/IoU_Flood: {ckpt.best_model_score}\n")
if best_ckpt and os.path.exists(best_ckpt):
    sys.__stdout__.write(f"  Disk: {os.path.getsize(best_ckpt)/1024/1024:.0f} MB\n")


# ---------------------------------------------------------------------------
# Predict — in-memory
# ---------------------------------------------------------------------------
datamodule.setup("predict")
pred_loader   = datamodule.predict_dataloader()
pred_files    = sorted(glob.glob(os.path.join(CFG["ENG_PRED_DIR"], "*.tif")))
pred_file_map = {
    os.path.basename(p).replace("_image.tif","").replace(".tif",""): p
    for p in pred_files
}
sys.__stdout__.write(f"\nInference on {len(pred_files)} patches...\n")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

if best_ckpt and os.path.exists(best_ckpt):
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["state_dict"], strict=False)
    sys.__stdout__.write(f"  Loaded: {os.path.basename(best_ckpt)}\n")

all_preds = {}
with torch.no_grad():
    for batch in pred_loader:
        images     = batch["image"].to(device)
        out        = model.model(images).output
        preds      = out.argmax(dim=1).cpu().numpy().astype("int16")
        file_paths = batch.get("filename", batch.get("filepath", []))
        if not file_paths:
            continue
        for i, fp in enumerate(file_paths):
            pid = os.path.basename(fp).replace("_image.tif","").replace(".tif","")
            all_preds[pid] = preds[i]

sys.__stdout__.write(f"  Predictions in memory: {len(all_preds)}\n")

# Fallback
if len(all_preds) == 0:
    sys.__stdout__.write("  Fallback: ordered inference\n")
    with torch.no_grad():
        for pid in sorted(pred_file_map.keys()):
            with rasterio.open(pred_file_map[pid]) as src:
                img_np = src.read().astype(np.float32)
            for b in range(CFG["NUM_BANDS"]):
                img_np[b] = (img_np[b] - MEANS[b]) / STDS[b]
            img_t = torch.tensor(img_np).unsqueeze(0).to(device)
            pred  = model.model(img_t).output.argmax(dim=1).squeeze(0).cpu().numpy().astype("int16")
            all_preds[pid] = pred
    sys.__stdout__.write(f"  Fallback done: {len(all_preds)}\n")


# ---------------------------------------------------------------------------
# RLE + submission
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
        fp = next((v for k,v in pred_file_map.items() if tid in k or k in tid), None)
        if fp:
            with rasterio.open(fp) as src:
                img_np = src.read().astype(np.float32)
            for b in range(CFG["NUM_BANDS"]):
                img_np[b] = (img_np[b] - MEANS[b]) / STDS[b]
            img_t = torch.tensor(img_np).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_arr = model.model(img_t).output.argmax(dim=1).squeeze(0).cpu().numpy().astype("int16")
        else:
            sys.__stdout__.write(f"  WARNING: no prediction for {tid}\n")
            rows.append({"id": tid, "rle_mask": "0 0"})
            issues.append(tid)
            continue
    flood_bin = (pred_arr == 1).astype(np.uint8)
    rows.append({"id": tid, "rle_mask": mask_to_rle(flood_bin)})
    sys.__stdout__.write(f"  {tid}: flood_px={flood_bin.sum():>8,}\n")

sub_df  = pd.DataFrame(rows)
sub_csv = os.path.join(CFG["OUT_DIR"], "submission_iter4.csv")
sub_df.to_csv(sub_csv, index=False)
sys.__stdout__.write(
    f"\nSubmission: {len(sub_df)} rows | "
    f"empty: {(sub_df['rle_mask']=='0 0').sum()} | issues: {len(issues)}\n"
)


# ---------------------------------------------------------------------------
# Training curve
# ---------------------------------------------------------------------------
metrics_csv = os.path.join(CFG["OUT_DIR"], "logs", "version_0", "metrics.csv")
if os.path.exists(metrics_csv):
    mdf  = pd.read_csv(metrics_csv)
    cols = [c for c in mdf.columns if any(k in c for k in
            ["epoch","val/mIoU","val/IoU_Flood","val/loss","train/loss"])]
    sys.__stdout__.write("\nTraining curve:\n")
    sys.__stdout__.write(mdf[cols].dropna(subset=["val/mIoU"]).to_string(index=False) + "\n")


# ---------------------------------------------------------------------------
# ZIP
# ---------------------------------------------------------------------------
summary = {
    "iteration"          : 4,
    "input_channels"     : CFG["BAND_NAMES"],
    "epochs_run"         : trainer.current_epoch,
    "duration_seconds"   : duration.total_seconds(),
    "best_val_IoU_Flood" : float(ckpt.best_model_score) if ckpt.best_model_score else None,
    "best_ckpt"          : best_ckpt,
    "config"             : CFG,
    "issues"             : issues,
}
json_path = os.path.join(CFG["OUT_DIR"], "iter4_summary.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)

zip_path = "/kaggle/working/submission_iter4.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub_csv,   "submission_iter4.csv")
    zf.write(json_path, "iter4_summary.json")
    if os.path.exists(metrics_csv):
        zf.write(metrics_csv, "training_metrics.csv")

sys.__stdout__.write(f"\nZIP: {zip_path} ({os.path.getsize(zip_path)/1024:.1f} KB)\n")
sys.__stdout__.write(f"val/IoU_Flood = {summary['best_val_IoU_Flood']}\n")
sys.__stdout__.write("Submit submission_iter4.csv\n")
