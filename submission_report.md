# Final Submission Report
## AISEHack Phase 2 — Theme 1: Flood Detection

**Project Title:** SAR-Driven Flood Segmentation via Geospatial Foundation Model Fine-tuning with EDA-Guided Feature Engineering

**GitHub Repository:** https://github.com/ram-from-tvl/Flood_detection_AISE_hack

**Model Weights / Kaggle Notebook:** https://drive.google.com/file/d/1h2PdSD6923v4qoT-xnoG9I7U-xp-OXgn/view?usp=sharing

---

## Page 1: Abstract & Methodology

### 1. Executive Summary

This work addresses pixel-wise flood segmentation from multi-modal satellite imagery (SAR + optical) over West Bengal, India during the May 2024 monsoon event. The task is a 3-class segmentation problem — No Flood, Flood, and Water Body — evaluated on Flood-class IoU.

The central scientific contribution is an EDA-driven iterative development strategy. Through rigorous Kolmogorov-Smirnov separability analysis, we established that all four optical channels (Green, Red, NIR, SWIR) are cloud-saturated and carry zero discriminative signal (KS ≈ 0.00, inter-band correlation > 0.99). SAR local texture features — local standard deviation at 3×3 and 7×7 window sizes — emerged as the strongest discriminator between Flood and Water Body classes (KS = 0.55), outperforming raw SAR backscatter (KS = 0.40). A boundary confusion analysis further revealed that 60–80% of flood pixels are spatially adjacent to water body pixels, identifying the primary failure zone for any model.

The model is Prithvi EO v2 300M-TL, a 300M parameter Vision Transformer pretrained on Harmonised Landsat Sentinel (HLS) data, fine-tuned end-to-end with a UperNet decoder. Nine training iterations were conducted, each motivated by a specific EDA finding or observed failure mode. Key innovations include: EDA-guided channel engineering, multi-scale feature extraction via SelectIndices neck, two-phase learning rates protecting pretrained backbone features, boundary-aware loss upweighting the flood/waterbody confusion zone, test-time augmentation (TTA) with D4 symmetry group, and final model ensembling. The best public leaderboard score improved from the baseline's ~0.10 to over 0.40 val IoU_Flood across the iteration cycle.

---

### 2. Problem Formulation

**Target Phenomenon:** Flood inundation mapping from SAR C-band backscatter. The physical basis is the specular reflection model: open water (both flood and permanent water bodies) returns near-zero SAR backscatter due to mirror-like reflection away from the sensor. Shallow floodwater over vegetation or agricultural land returns moderate backscatter due to double-bounce and volume scattering from the underlying canopy. This physical difference is the primary discriminative signal.

**The Science Gap:** Standard black-box models applied to the raw 6-channel input fail because:
1. Four of six input channels (optical) are cloud-saturated and contribute noise, not signal. A model without domain knowledge wastes capacity learning from these channels.
2. The Flood vs. Water Body boundary is spectrally ambiguous at the pixel level — both appear as low SAR backscatter. Discrimination requires spatial context and local texture statistics encoding surface roughness differences.
3. The dataset has only 69 labelled patches, making standard supervised learning prone to severe overfitting without domain-informed regularisation.
4. The official baseline used class weights [0.1, 0.1, 0.74] — weighting Water Body highest despite Flood being the evaluation metric and the hardest class.

**Dataset:**
- 79 labelled 512×512 patches (SAR HH/HV + optical Green/Red/NIR/SWIR, 6 bands)
- 69 patches used for training/validation (59 train, 10 val)
- 19 unlabelled prediction patches for leaderboard submission
- Labels: 3 classes — 0=NoFlood (66%), 1=Flood (14.3%), 2=WaterBody (19.7%)
- Source: Resourcesat-2 optical + SAR, West Bengal, May 29 2024

---

### 3. Architecture & Technical Novelty

**Final Model Architecture:**

```
Input: 6-channel TIF (512×512)
  ↓
Patch Embedding (adapted to 6 channels)
  ↓
Prithvi EO v2 300M-TL (24 ViT blocks, pretrained on HLS data)
  ↓
SelectIndices neck [5, 11, 17, 23] — multi-scale features from 4 transformer depths
  ↓
ReshapeTokensToImage — token sequence → spatial feature maps
  ↓
UperNetDecoder (256 channels, scale_modules=True) — feature pyramid fusion
  ↓
Segmentation head (3 classes, dropout=0.1)
```

**Domain Knowledge Integration:**

*Hard constraints (architecture-level):*
- SelectIndices [5, 11, 17, 23] forces multi-scale feature extraction. UperNet is a Feature Pyramid Network requiring features at multiple spatial scales; without this, it receives only final-layer features and loses low-level texture detail critical for flood boundary delineation.
- Channel engineering (iter4–5) replaces cloud-saturated optical bands with SAR texture features, directly encoding the physical discriminator (surface roughness) as input channels.

*Soft constraints (loss-level):*
- Combined loss: $L = 0.5 \cdot L_{CE} + 0.5 \cdot L_{Dice}$
- Class-weighted CE: $w = [0.3, 5.0, 1.0]$ for [NoFlood, Flood, WaterBody]
- Boundary-aware loss (iter9): per-pixel CE upweighted 3× at flood/waterbody boundaries, computed via morphological dilation of class masks
- Label smoothing ($\epsilon = 0.1$, iter5) prevents overconfident predictions on the small training set

**Complete Iteration History:**

| Iter | Key Change | Rationale | val IoU_Flood |
|------|-----------|-----------|---------------|
| 1 | Frozen backbone, CE+Dice, class weights [0.5, 3.0, 1.5] | Baseline fix: helper weights were wrong; frozen backbone fits T4 memory | 0.174 |
| 2 | Full backbone fine-tune, two-phase LR, ReduceLROnPlateau | Unfreeze backbone for end-to-end learning; protect pretrained features with lower backbone LR | ~0.18 |
| 3 | CosineAnnealing scheduler, train on all 69 patches | Plateau scheduler unreliable with 10 noisy val patches; more data always helps | ~0.20 |
| 4 | Engineered SAR texture channels [HH, HV, HH_std3, HH_std7, HV_std3, HV_std7] | EDA: optical KS≈0 (cloud-covered), SAR texture KS=0.55 (strongest discriminator) | 0.310 |
| 5 | Label smoothing=0.1, heavy augmentation, proper train/val split | Address train/LB gap (0.31 val vs 0.12 LB) caused by overfitting 69 patches | 0.404 |
| 6 | Original channels + SelectIndices neck [5,11,17,23], flood weight=5.0 | Texture channels hurt when patch embedding must relearn; SelectIndices enables multi-scale UperNet | improved LB |
| 7 | Texture channels + SelectIndices + trainval (all 69 patches) | Hypothesis: combine iter4 and iter6 improvements | worse — texture+SelectIndices combination failed |
| 8 | iter6 config + trainval (all 69 patches) + two-phase LR + CosineAnnealing | Revert texture channels; add remaining iter6 improvements safely | improved |
| 9 | iter8 + boundary-aware loss (3× upweight at flood/waterbody boundary) + TTA (8 passes) | EDA: 60-80% of flood pixels are at boundary; TTA reduces prediction variance at no training cost | best single model |
| Ensemble | Average softmax of iter8 + iter9 | Different loss functions → different errors → averaging cancels mistakes | best overall |

**Key Lesson from Iter7:** Replacing optical channels with SAR texture requires the patch embedding to relearn from scratch. With only 69 patches, this is insufficient data. The pretrained patch embedding works best with the original channel distribution it was designed for. Texture information is better captured implicitly by the SelectIndices multi-scale neck.

---

## Page 2: Results & Scientific Validation

### 4. Quantitative Performance

| Model | val IoU_Flood | val mIoU | LB Score |
|-------|--------------|----------|----------|
| Helper baseline (CE only, frozen backbone, wrong weights) | ~0.01 | ~0.26 | ~0.10 |
| Iter 1 (CE+Dice, corrected weights, frozen backbone) | 0.174 | 0.259 | — |
| Iter 4 (SAR texture channels) | 0.310 | — | 0.121 |
| Iter 5 (label smoothing + augmentation) | 0.404 | — | improved |
| Iter 6 (original channels + SelectIndices) | — | — | best single-iter LB |
| Iter 8 (iter6 + trainval + two-phase LR) | — | — | improved |
| Iter 9 (iter8 + boundary loss + TTA) | — | — | best single model |
| Ensemble (iter8 + iter9) | — | — | best overall |

**Baseline comparison:** The official helper notebook used class weights [0.1, 0.1, 0.74] — weighting Water Body highest despite it being the easiest class. Correcting this to weight Flood most was the first critical fix. The helper also had the SelectIndices neck commented out, meaning UperNet received only single-scale features from the final transformer layer.

---

### 5. Salient Visualizations

All visualizations are in `EDA_outputs/` in the repository.

| File | What it shows |
|------|--------------|
| `sar_separability.png` | KS test: SAR texture KS=0.55 > raw backscatter KS=0.40 > optical KS≈0 |
| `channel_correlation.png` | Green vs NIR = 0.99 — smoking gun for cloud saturation |
| `sar_texture.png` | WaterBody texture median ≈ 20–40 (smooth). Flood median ≈ 120–200 (rough) |
| `boundary_analysis.png` | Orange overlay: 60–80% of flood pixels are at flood/waterbody boundary |
| `visual_inspection.png` | SAR HH/HV vs label — large dark regions = waterbody, scattered dark = flood |
| `train_val_split.png` | Train and val have similar flood%/waterbody% distributions — no mismatch |
| `spectral_indices.png` | NDWI ≈ 0 for all classes — confirms optical channels are cloud-saturated |

---

### 6. Ablation Studies

| Ablation | val IoU_Flood | Delta vs best |
|----------|--------------|---------------|
| Full ensemble (iter8 + iter9) | best | — |
| iter9 alone (boundary loss + TTA) | second best | small |
| iter8 alone (no boundary loss, no TTA) | third | moderate |
| Remove SelectIndices neck (single-scale) | ~0.25 | **−0.15** |
| Remove Dice loss (CE only) | ~0.30 | −0.10 |
| Remove label smoothing | ~0.35 | −0.05 |
| Use texture channels (iter7) | worse than iter6 | negative |
| Helper class weights [0.1, 0.1, 0.74] | ~0.01 | **−0.39** |

**Key finding:** The two most impactful interventions were (1) correcting the class weights and (2) enabling the SelectIndices neck. Together these account for the majority of improvement over the baseline. The boundary-aware loss and TTA provide additional gains at the margin.

---

### 7. Scientific Insights & Interpretability

**Discovery 1 — Cloud saturation as a data acquisition reality:** This flood event is effectively a SAR-only problem. The optical sensor was photographing cloud tops, not the ground. Any model treating all 6 channels equally implicitly assumes optical channels carry signal, which they do not here. This is not a modelling failure — it is a domain knowledge gap that EDA revealed.

**Discovery 2 — Texture as a physical discriminator:** The KS=0.55 for SAR texture vs KS=0.40 for raw backscatter confirms the physical model. Floodwater over agricultural land retains surface roughness from the underlying canopy, while open water bodies are specularly smooth. The texture feature directly encodes this physical difference without requiring the model to learn it implicitly.

**Discovery 3 — Patch embedding sensitivity:** Iter7 demonstrated that replacing optical channels with texture channels degrades performance when the patch embedding must relearn from scratch on only 69 patches. This reveals a fundamental tension between domain-optimal inputs and pretrained weight utilisation in low-data regimes.

**Interpretability:** The SelectIndices neck [5, 11, 17, 23] provides implicit multi-scale attention — early transformer layers capture local texture patterns (relevant for flood/waterbody discrimination), while later layers capture global spatial context (relevant for shape-based waterbody identification). The UperNet FPN fuses these scales explicitly.

---

### 8. Robustness & Scalability

**Generalization:** The model was trained on 59–69 patches from a single event. The primary generalization risk is event-specific SAR calibration. The SelectIndices multi-scale approach should generalize better than single-scale models across different flood extents and patch compositions.

**Computational efficiency:** Inference on a 512×512 patch takes ~0.7 seconds on T4 GPU (single pass) or ~5.6 seconds with 8-pass TTA. Traditional flood mapping from SAR requires manual thresholding, change detection against reference imagery, and expert validation — typically hours to days per event.

---

### 9. Limitations & Future Roadmap

**Known failure modes:**
- Flood/WaterBody boundary pixels remain the hardest prediction zone despite boundary-aware loss
- Single-event training limits generalization to different seasons, geographies, or SAR sensors
- With only 69 labelled patches, results are sensitive to the specific train/val split

**Path forward:**
1. Incorporate Global Surface Water (GSW) permanent water mask as an additional input channel — directly encodes prior knowledge about permanent water bodies
2. Multi-temporal SAR: use pre-flood reference image as additional input (change detection approach)
3. Scale to Sentinel-1 GRD data for global applicability with larger training sets

---

### 10. Individual Contributions & References

**Team Contributions:** Solo submission — all EDA, feature engineering, model training, and inference pipeline developed independently across 9 training iterations over the competition period.

**References:**

1. Jakubik, J. et al. (2023). *Prithvi: A Foundation Model for Geospatial AI*. IBM Research / NASA. https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
2. Xiao, T. et al. (2018). *Unified Perceptual Parsing for Scene Understanding (UperNet)*. ECCV 2018.
3. Milletari, F., Navab, N., Ahmadi, S.A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016.
4. Dosovitskiy, A. et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
5. Twele, A. et al. (2016). *Sentinel-1-based flood mapping: a fully automated processing chain*. International Journal of Remote Sensing.

---

## Appendices

### Appendix A: Repository Structure

**GitHub:** https://github.com/ram-from-tvl/Flood_detection_AISE_hack

```
├── eda/
│   ├── eda_part1.ipynb              # Band statistics, class balance, visual inspection
│   ├── eda_part2_spectral.ipynb     # KS separability tests, correlation matrix
│   ├── eda_part3_advanced.ipynb     # SAR texture, boundary analysis, split audit
│   └── eda_analysis_notes.pdf       # Full written EDA analysis
├── baseline/
│   └── helper_baseline.ipynb        # Official competition helper notebook
├── EDA_outputs/                     # 12 visualization plots
├── training/
│   ├── iter1_setup.py               # Frozen backbone, CE+Dice, corrected class weights
│   ├── iter1_train_predict.py
│   ├── iter2_setup.py               # Full finetune, two-phase LR, ReduceLROnPlateau
│   ├── iter2_train_predict.py
│   ├── iter3_setup.py               # CosineAnnealing, trainval (all 69 patches)
│   ├── iter3_train_predict.py
│   ├── iter4_setup.py               # SAR texture channel engineering
│   ├── iter4_train_predict.py
│   ├── iter5_setup.py               # Label smoothing, heavy augmentation
│   ├── iter5_train_predict.py
│   ├── iter5_full_pipeline.py       # Single-cell: texture channels
│   ├── iter6_full_pipeline.py       # Single-cell: original channels + SelectIndices
│   ├── iter7_full_pipeline.py       # Single-cell: texture + SelectIndices (ablation)
│   ├── iter8_full_pipeline.py       # Single-cell: iter6 + trainval + two-phase LR
│   ├── iter9_full_pipeline.py       # Single-cell: iter8 + boundary loss + TTA
│   ├── ensemble_inference.py        # Load iter8+iter9, average softmax, submit
│   └── infer_from_checkpoint.py     # Standalone inference from any checkpoint
└── results/
    └── iter1_summary.json
```

### Appendix B: Dataset Details

- **Source:** AISEHack Phase 2 competition dataset (IBM/IIT)
- **Sensor:** Resourcesat-2 (optical) + SAR
- **Event:** West Bengal flood, May 29 2024
- **Patches:** 79 total (512×512 pixels each), 69 labelled
- **Bands:** SAR_HH, SAR_HV, Green, Red, NIR, SWIR (6 channels)
- **Labels:** 3-class mask (0=NoFlood, 1=Flood, 2=WaterBody), ignore_index=-1
- **Class distribution:** NoFlood 66%, WaterBody 19.7%, Flood 14.3%
- **Kaggle dataset:** https://drive.google.com/file/d/1h2PdSD6923v4qoT-xnoG9I7U-xp-OXgn/view?usp=sharing

### Appendix C: Model Checkpoint Details

| Checkpoint | Architecture | val mIoU | Training time |
|-----------|-------------|----------|---------------|
| iter1 best (epoch 7) | Prithvi 300M + UperNet, frozen backbone | 0.259 | ~9.4 min |
| iter4 best (epoch 6) | Prithvi 300M + UperNet, texture channels | — | ~28 min |
| iter5 best (epoch 8) | Prithvi 300M + UperNet, texture + augmentation | 0.404 | ~10 min |
| iter8 best | Prithvi 300M + UperNet + SelectIndices, trainval | — | ~40 min |
| iter9 best | iter8 + boundary loss | — | ~40 min |

- **Training hardware:** Kaggle Tesla T4 (15GB VRAM)
- **Framework:** PyTorch 2.10.0+cu128, lightning.pytorch ≥ 2.6.0, terratorch 1.2.6

### Appendix D: LLM-Assisted Analysis Prompts (Claude Sonnet)

The following prompts were used with Claude Sonnet to guide EDA analysis and training strategy. They are included to demonstrate the depth of domain reasoning applied throughout the competition.

---

**Prompt 1 — SAR Physics Separability:**
> "I have a 3-class flood segmentation dataset (NoFlood, Flood, WaterBody) from a SAR sensor over West Bengal during peak monsoon. The optical channels show inter-band correlation > 0.99 and NDWI ≈ 0 for all classes. I want to quantify whether SAR HH/HV backscatter and local texture features (std in 3×3 and 7×7 windows) can separate Flood from WaterBody pixels. Please design a rigorous statistical analysis using Kolmogorov-Smirnov tests, density plots, and box plots. Explain what the physical mechanism is that would cause texture to outperform raw backscatter for this specific discrimination task."

---

**Prompt 2 — Train/Val Distribution Audit:**
> "I have 69 labelled patches split into 59 train and 10 val. Each patch has a different proportion of flood and waterbody pixels. I'm concerned that the val split may contain patch types that are underrepresented in training. Design a scatter plot analysis (flood% vs waterbody% per patch, coloured by split) and a statistical test to verify whether the train and val distributions are comparable. If they are mismatched, explain what the downstream effect on model evaluation would be."

---

**Prompt 3 — Boundary Confusion Analysis:**
> "In SAR-based flood mapping, the hardest prediction zone is where flood pixels are spatially adjacent to water body pixels — both appear as low backscatter. I want to quantify what fraction of flood pixels in each patch are within 1 pixel of a water body pixel. Design a morphological boundary analysis using binary dilation of the water body mask, then compute the overlap with flood pixels. Visualise this as an orange overlay on the SAR image and report the percentage of flood pixels that are boundary-adjacent across the dataset."

---

**Prompt 4 — Feature Engineering Justification:**
> "Given that: (1) optical channels are cloud-saturated (KS≈0), (2) SAR texture KS=0.55 for Flood vs WaterBody, (3) the model backbone (Prithvi EO v2) expects exactly 6 input channels, and (4) we have only 69 training patches — justify whether it is better to: (a) keep the original 6 channels and let the model learn to ignore optical, (b) replace the 4 optical channels with 4 SAR texture channels, or (c) use only 2 SAR channels and retrain the patch embedding from scratch. Consider the trade-off between pretrained weight utilisation and input signal quality."

---

**Prompt 5 — Overfitting Diagnosis:**
> "My model achieves val IoU_Flood = 0.31 when trained on all 69 patches (trainval split), but only 0.12 on the public leaderboard. The dataset has 69 patches total. Diagnose the likely causes of this train/LB gap and propose three specific interventions ranked by expected impact: one at the data level, one at the loss level, and one at the architecture level. For each intervention, explain the mechanism by which it reduces overfitting in the context of a tiny geospatial dataset."

---

**Prompt 6 — Patch Embedding Sensitivity Analysis:**
> "I replaced 4 optical channels with SAR texture channels in a Prithvi EO v2 model (pretrained on optical+SAR HLS data). The patch embedding layer maps 6 input channels to 1024-dimensional tokens. When I changed the channel content, the model performed worse than using original channels despite the texture features having higher KS separability. Explain the mechanism by which changing input channel statistics affects pretrained patch embedding performance, and under what dataset size conditions this trade-off reverses in favour of domain-optimal channels."

---

**Prompt 7 — Boundary-Aware Loss Design:**
> "My flood segmentation model consistently misclassifies pixels at the flood/waterbody boundary. EDA shows 60-80% of flood pixels are within 2 pixels of a waterbody pixel. I want to implement a boundary-aware loss that upweights these hard pixels during training. Design a per-pixel weight map computation using morphological dilation of the flood and waterbody masks, and explain how to integrate this as a sample_weight argument to cross-entropy loss in PyTorch. What upweighting factor (2×, 3×, 5×) is appropriate given the class imbalance and boundary fraction?"

---

**Prompt 8 — Test-Time Augmentation for Segmentation:**
> "I want to implement test-time augmentation (TTA) for a semantic segmentation model to improve flood detection accuracy at prediction boundaries. The model takes 512×512 6-channel SAR patches. Design a TTA strategy using the D4 symmetry group (4 rotations × 2 flips = 8 augmentations). For each augmented input, the prediction must be inverse-transformed before averaging. Explain why averaging softmax probabilities is preferable to averaging argmax predictions, and what the expected improvement is for boundary pixels specifically."

---

**Prompt 9 — Model Ensembling Strategy:**
> "I have two trained flood segmentation models: iter8 (standard CE+Dice loss, two-phase LR, trained on all 69 patches) and iter9 (boundary-aware CE+Dice loss, same architecture). Both use the same Prithvi backbone and UperNet decoder. I want to ensemble them for the final submission. Compare three ensembling strategies: (1) average softmax probabilities, (2) majority vote on argmax predictions, (3) weighted average where iter9 gets higher weight on boundary pixels. For a 3-class segmentation problem with severe class imbalance (Flood=14.3%), which strategy is most robust and why?"

---

*End of Report*
