# Flood Detection — AISEHack Phase 2

SAR-based flood segmentation using the [Prithvi EO v2 300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL) geospatial foundation model. Competition: AISEHack Phase 2, Theme 1 — Flood Detection (West Bengal, May 2024).

## Problem

3-class pixel-wise segmentation of 512×512 SAR + optical patches:
- Class 0: No Flood
- Class 1: Flood (primary evaluation metric)
- Class 2: Water Body

Evaluation metric: IoU on the Flood class.

## Key EDA Findings

- Optical channels (Green, Red, NIR, SWIR) are cloud-saturated — KS ≈ 0.00, inter-band correlation > 0.99. Effectively useless for this event.
- SAR raw backscatter separates Flood vs WaterBody with KS = 0.40.
- SAR local texture (std in 3×3 and 7×7 windows) is the strongest discriminator: KS = 0.55.
- WaterBody pixels cluster at very low SAR backscatter (specular reflection from open water). Flood pixels sit at moderate backscatter due to shallow water over vegetation.
- Train/val split is well-balanced — no distribution mismatch.

## Repository Structure

```
├── eda/
│   ├── eda_part1.ipynb              # Initial EDA — band statistics, class balance
│   ├── eda_part2_spectral.ipynb     # Spectral separability analysis (KS tests)
│   ├── eda_part3_advanced.ipynb     # SAR texture, boundary analysis, train/val split audit
│   └── eda_analysis_notes.pdf       # Detailed written analysis of EDA findings
│
├── baseline/
│   └── helper_baseline.ipynb        # Official competition helper notebook
│
├── training/
│   ├── iter1_setup.py               # Iter 1 setup: frozen backbone, CE+Dice loss, corrected class weights
│   ├── iter1_train_predict.py       # Iter 1 train + predict
│   ├── iter2_setup.py               # Iter 2 setup: full finetune, two-phase LR, ReduceLROnPlateau
│   ├── iter2_train_predict.py       # Iter 2 train + predict
│   ├── iter3_setup.py               # Iter 3 setup: CosineAnnealing, train on all 69 patches
│   ├── iter3_train_predict.py       # Iter 3 train + predict
│   ├── iter4_setup.py               # Iter 4 setup: engineered SAR texture channels
│   ├── iter4_train_predict.py       # Iter 4 train + predict
│   ├── iter5_setup.py               # Iter 5 setup: label smoothing, heavy augmentation
│   ├── iter5_train_predict.py       # Iter 5 train + predict
│   ├── iter5_full_pipeline.py       # Iter 5 single-cell pipeline (engineered channels)
│   ├── iter6_full_pipeline.py       # Iter 6 single-cell pipeline (original channels + SelectIndices neck)
│   └── infer_from_checkpoint.py    # Standalone inference from any saved checkpoint
│
└── results/
    └── iter1_summary.json           # Iter 1 training summary and metrics
```

## Iteration Summary

| Iter | Key Change | val IoU_Flood | LB Score |
|------|-----------|---------------|----------|
| 1 | Frozen backbone, CE+Dice, corrected class weights | 0.174 | — |
| 2 | Full finetune, two-phase LR (backbone 10x lower) | — | — |
| 3 | CosineAnnealing, train on all 69 patches | — | — |
| 4 | Engineered channels: replace optical with SAR texture | 0.310 | 0.121 |
| 5 | Label smoothing, heavy augmentation, proper val split | 0.404 | improved |
| 6 | Original channels + SelectIndices neck, flood weight=5.0 | — | — |

## Architecture

- Backbone: `prithvi_eo_v2_300_tl` (300M parameter ViT, pretrained on HLS data)
- Decoder: UperNetDecoder (256 channels)
- Neck: SelectIndices [5, 11, 17, 23] → ReshapeTokensToImage (multi-scale features)
- Loss: CrossEntropy (label smoothing=0.1) + MulticlassDice, weighted 50/50
- Class weights: [0.3, 5.0, 1.0] — Flood weighted highest as it is the eval metric

## Engineered Input (iter4–5)

The 4 optical channels are cloud-saturated and carry no discriminative signal. They are replaced with SAR texture features:

```
Original: [SAR_HH, SAR_HV, Green, Red, NIR, SWIR]
Engineered: [SAR_HH, SAR_HV, HH_std3, HH_std7, HV_std3, HV_std7]
```

Where `HH_std3` = local standard deviation of SAR_HH in a 3×3 window (computed via `scipy.ndimage.uniform_filter`).

## Environment

- Platform: Kaggle (Tesla T4, 15GB VRAM)
- PyTorch 2.10.0+cu128
- lightning.pytorch >= 2.6.0
- terratorch 1.2.6

## Usage

Each iteration follows a two-cell pattern (setup + train/predict) or a single-cell pipeline for later iterations. Run on Kaggle with the competition dataset mounted at:

```
/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data/
```

Install dependencies (handled automatically at the top of each script):

```bash
pip install terratorch albumentations rasterio numpy==2.2.0
```
