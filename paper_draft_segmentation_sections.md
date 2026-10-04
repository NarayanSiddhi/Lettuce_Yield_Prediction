# Paper Draft — Segmentation & Yield Pipeline (Sections 2–4)

*Draft text for the yield-prediction paper. Adjust author names, dataset citations, and journal formatting as needed.*

---

## 2. Related Work

### 2.1 Non-destructive lettuce yield and growth monitoring

Fresh weight (FW) is a primary indicator of lettuce growth in controlled-environment agriculture. Traditional harvest-based measurement is accurate but destructive, labour-intensive, and incompatible with continuous monitoring in grow tents or plant factories. Computer vision has therefore been widely adopted for indirect estimation of lettuce biomass and morphological traits from RGB or RGB-D imagery.

Several studies fuse geometric and appearance features for FW prediction. Gao et al. (2022) proposed a multi-modal deep learning framework that segments lettuce leaves with a U-Net, extracts canopy geometry, and fuses colour, depth, and empirical traits to estimate shoot fresh weight, reporting an R² of 0.938 and RMSE of 25.3 g on plant-factory imagery. Xu et al. (2023) improved RGB-based FW estimation through data enrichment, network-structure changes, and a mean-squared-percentage-error loss, and further showed that RGB-D fusion reduces MAPE to approximately 8.5%. Wang et al. (2022) developed TMSCNet, a multi-branch network that jointly estimates fresh weight, dry weight, height, diameter, and leaf area from RGB-D inputs. More recent work includes MIFFNet, a multi-dimensional feature-fusion network for lettuce FW estimation (Xu et al., 2026), and three-dimensional point-cloud approaches that segment collective lettuce canopies before regressing weight (Li et al., 2024). Together, these studies establish that **accurate plant segmentation is an upstream prerequisite** for reliable feature extraction and yield modelling in hydroponic lettuce systems.

### 2.2 Classical colour-index segmentation

Before deep learning became dominant, plant–background separation relied on colour vegetation indices (CVIs) computed in RGB space. The Excess Green index (ExG) enhances green vegetation relative to soil and background (Woebbecke et al., 1995). Meyer and Neto (2008) compared ExG, normalised difference indices, and Excess Green minus Excess Red (ExG−ExR, also written ExGR), finding that ExGR with a fixed zero threshold outperformed Otsu-thresholded ExG in greenhouse and field settings. Riehle et al. (2021) evaluated ExG, ExR, and ExGR combined with Otsu or clustering for plant/soil segmentation. Knelshausen et al. (2020) combined index-based pre-segmentation with HSV and CIELAB thresholding for robust plant/background separation under variable illumination.

For lettuce specifically, optimised ExG binarisation has been used to estimate seedling leaf area in plant factories, where adaptive thresholding (O-ExG) was compared against fixed ExG and U-Net segmentation (Nakano et al., 2023). Classical methods remain relevant as **fast, training-free baselines**, although they often struggle with overlapping canopies, white cultivation infrastructure, and red/purple cultivars in dense hydroponic arrays (Zhao et al., 2024).

### 2.3 Deep learning and foundation-model segmentation

Convolutional architectures such as U-Net (Ronneberger et al., 2015) remain standard for semantic leaf segmentation when labelled training data are available; the Frontiers plant-factory study cited above achieved mIoU 0.982 with a U-Net lettuce masker (Gao et al., 2022). Instance segmentation models, including YOLOv8-seg variants, have been applied to lettuce in hydroponic and pot-cultivation scenarios for height measurement and multi-plant localisation (Zhao et al., 2024; Chen et al., 2023). Modified YOLOv8 architectures with BiFPN or Ghost modules further improve leaf-instance segmentation on phenotyping benchmarks (Wang et al., 2024).

Foundation models changed the landscape by enabling promptable, zero-shot segmentation. Kirillov et al. (2023) introduced the Segment Anything Model (SAM), and Zhao et al. (2023) proposed FastSAM as a faster CNN-based alternative suitable for near-real-time use. A Cornell greenhouse study compared SAM, FastSAM, and binary thresholding on high-density lettuce imagery, reporting SAM IoU 0.985 and FastSAM IoU-comparable performance with roughly threefold lower latency (Cornell eCommons, 2024). FLAsH (FastSAM-Based Lettuce Analysis for High-Density Cultivation) fine-tunes FastSAM for dense smart-farm lettuce, achieving mean average precision 0.788 for instance segmentation and MAPE below 10% for growth prediction tasks (Hyeseung et al., 2026). Hybrid pipelines that localise leaves with U-Net and refine boundaries with FastSAM have also reported strong IoU on complex natural scenes (Sharma et al., 2024).

Despite this progress, **comparative studies on the same grow-tent timelapse under identical post-processing remain scarce**, particularly studies that connect segmentation choice to downstream yield-feature quality in research-scale datasets with few labelled harvests.

### 2.4 Research gap addressed in this work

Prior literature provides strong methods in isolation—classical indices, supervised instance segmentation, and zero-shot foundation models—but rarely evaluates them on **the same hydroponic timelapse** with harmonised QA rules and explicit downstream feature extraction for FW prediction. Our work fills this gap by implementing and comparing three literature-backed families—ExGR+Otsu, SAM, and FastSAM—on matched trial-day frames from two lettuce trials, then committing the winning approach to the yield-modelling pipeline described in Section 4.

---

## 3. Segmentation Methods

We evaluated three segmentation approaches that can be applied without training new weights on our dataset: a classical vegetation-index baseline, and two zero-shot foundation models. All methods output instance-level binary masks per image. To ensure a fair comparison, **identical post-segmentation rules** (mask-size limits, plant-colour checks, and spatial rejection of pipe regions) were applied to every method (see Section 3.4).

### 3.1 Method A — ExGR with Otsu thresholding and connected components

**Literature basis:** Meyer and Neto (2008); Riehle et al. (2021); Nakano et al. (2023).

Given an RGB image, channels are normalised to \([0,1]\) and the Excess Green and Excess Red indices are computed as:

\[
\text{ExG} = 2G' - R' - B', \qquad \text{ExR} = 1.4R' - B'
\]
\[
\text{ExGR} = \text{ExG} - \text{ExR}
\]

The ExGR map is scaled to 8-bit and binarised using **Otsu's automatic threshold** (Otsu, 1979), which adapts to day-to-day lighting more robustly than a fixed zero threshold in our grow-tent conditions. Morphological opening and closing (5×5 elliptical kernel) remove speckle and fill small holes. Connected-component labelling (8-connectivity) yields candidate instances; components smaller than 8,000 pixels are discarded. Up to 32 largest instances per frame are retained. This pipeline is computationally lightweight and requires no GPU model weights.

### 3.2 Method B — SAM (Segment Anything Model, zero-shot)

**Literature basis:** Kirillov et al. (2023); Cornell eCommons (2024).

SAM is a promptable foundation model pre-trained on large-scale segmentation data. We use the **Ultralytics implementation** with the `sam_b.pt` checkpoint in automatic mask-generation mode (no fine-tuning on our lettuce images), consistent with zero-shot protocols in recent agricultural phenotyping work. For each full-resolution timelapse frame, SAM proposes a set of instance masks; masks are resized to the image grid, thresholded at 0.5, and filtered by area (8,000–1,200,000 pixels). SAM generally produces highly accurate boundaries in dense greenhouse scenes but at higher computational cost than FastSAM (Cornell eCommons, 2024).

### 3.3 Method C — FastSAM (Fast Segment Anything Model)

**Literature basis:** Zhao et al. (2023); Hyeseung et al. (2026).

FastSAM replaces SAM's transformer backbone with a faster CNN segmenter while retaining prompt-free, multi-instance behaviour. We use the **FastSAM-x** checkpoint via Ultralytics with confidence 0.25, IoU 0.7, and `retina_masks=True`. Inference resolution defaults to 1024 px with automatic fallback to smaller sizes on GPU memory limits. FastSAM is designed for settings where many regions must be segmented per frame—matching our multi-cup hydroponic tray—and has been explicitly applied to lettuce in high-density cultivation (FLAsH, 2026).

### 3.4 Common post-processing and feature pathway

Regardless of segmentation method, each candidate mask undergoes the same **`is_plant_crop` validation**:

| Rule | Purpose |
|------|---------|
| Mask area 8,000–250,000 px | Remove noise and full-frame blobs |
| Plant-colour fraction ≥ 0.10 (HSV green/red) | Prefer lettuce over grey infrastructure |
| White fraction ≤ 0.40 | Reject PVC pipes and reflective strips |
| Spatial heuristics (left edge, top bar, aspect ratio) | Suppress known pipe artefacts in our camera view |

Accepted masks are saved as RGBA crops. For the **production yield pipeline**, only FastSAM outputs were carried forward at full scale:

1. **QA filtering** (`filter_crops_qa.py`) — blur, colour, and size scoring  
2. **Cup assignment** (`assign_cup_ids.py`) — K-means on crop centroids (~14 cups/tray)  
3. **Daily cup features** (`build_daily_cup_features.py`) — best crop per cup per day  
4. **Checkpoint windows** (`build_master_dataset_from_cups.py`) — 4-day lookback before each harvest day  
5. **ML export** (`export_ml_datasets.py`) — modelling tables under `ml/`

The primary modelling file is `ml/checkpoint_trial1_image.csv` (eight labelled harvest checkpoints; image-derived window features + growth slopes). Sensor features are merged in companion tables for combined models.

---

## 4. Segmentation Comparison and Selection for Yield Prediction

### 4.1 Experimental protocol

We compared all three methods on **53 matched frames**: one representative image per trial-day (Trial 1 and Trial 2), selected at the hour closest to midday for consistent lighting. Each method segmented the same source files; runtime, mask counts, canopy coverage, and pairwise overlap were recorded. Side-by-side qualitative panels were generated for visual inspection (`segmentation/comparison/side_by_side/`).

**Metrics reported:**

| Metric | Definition |
|--------|------------|
| *Raw masks* | Instance count after size filtering only |
| *Plant crops* | Masks passing shared `is_plant_crop` rules |
| *Coverage* | Union of accepted plant masks ÷ image area |
| *Plant colour* | Mean fraction of plant-like HSV pixels inside accepted masks |
| *Time* | Wall-clock seconds per 4056×3040 frame |
| *Dice / IoU (pairwise)* | Overlap between methods' plant-union masks on the same frame |

Absolute IoU against manual annotations is reserved for future work; pairwise overlap measures **inter-method agreement**, not ground-truth accuracy.

### 4.2 Quantitative results

**Table 1.** Segmentation comparison on 53 matched trial-day frames (mean ± reporting format as aggregate means).

| Method | Time (s) | Raw masks | Plant crops | Coverage | Plant colour | Composite score* |
|--------|----------|-----------|-------------|----------|--------------|------------------|
| **FastSAM** | **1.53** | 32.0 | **21.0** | **0.129** | 0.641 | **0.747** |
| ExGR + Otsu | **0.60** | 23.0 | 17.1 | 0.061 | **0.711** | 0.691 |
| SAM | 4.55 | 31.6 | 21.7 | 0.112 | 0.671 | 0.663 |

\*Composite score = 0.30×cup-count proximity (target ~12) + 0.35×coverage + 0.20×plant colour + 0.15×speed (higher is better). Report individual metrics as primary evidence.

**Table 2.** Pairwise plant-union agreement (mean across 53 frames).

| Pair | Dice | IoU |
|------|------|-----|
| FastSAM vs SAM | **0.603** | **0.452** |
| FastSAM vs ExGR | 0.133 | 0.074 |
| ExGR vs SAM | 0.187 | 0.106 |

**Full-timelapse FastSAM deployment (821 frames):** 16,980 candidate crops; 3,637 passed QA (21.4%); 243 cup-day feature rows after cup assignment and daily aggregation.

### 4.3 Qualitative findings

Visual comparison confirms that **ExGR** often produces fragmented or merged regions and misses portions of the canopy in dense arrays—consistent with its low coverage (6.1%). **SAM** and **FastSAM** produce coherent instance masks on individual plants and pipes; the two deep models largely agree on plant extent (Dice 0.60), while both diverge substantially from ExGR. Pipe-rejection heuristics remain necessary for all methods because segmentation alone does not distinguish white PVC from pale plant tissue.

### 4.4 Discussion and method selection

ExGR is the fastest method and yields the highest per-mask plant-colour score, but its **canopy coverage is roughly half** that of FastSAM, and its overlap with both foundation models is poor (IoU < 0.11). This indicates that classical index thresholding is insufficient as the sole segmenter for our multi-cup grow-tent geometry.

SAM achieves the highest raw plant-crop count (21.7 per frame) and strong overlap with FastSAM, supporting Cornell's finding that the two models are statistically similar in dense lettuce scenes. However, SAM requires **~3× longer inference** (4.55 s vs 1.53 s per frame), which is prohibitive for batch processing of 800+ timelapse images.

**FastSAM** offers the best practical trade-off: highest vegetation coverage (12.9%), near-SAM plant counts, strong FastSAM–SAM agreement, and acceptable throughput. We therefore **selected FastSAM as the sole segmentation method for downstream yield prediction.**

### 4.5 Link to yield-modelling pipeline

Segmentation outputs feed the yield study as follows:

```
Timelapse RGB  →  FastSAM  →  QA + cup IDs  →  daily cup features
       ↓                                              ↓
  sensor logs  →  daily merge  →  checkpoint windows (4 d before harvest)
       ↓                                              ↓
  FW labels    →  master_checkpoint_dataset  →  ml/checkpoint_trial1_image.csv
```

- **Trials:** Trial 1 (primary; eight harvest checkpoints, days 0–28) and Trial 2 (exploratory; protocol-shift caveat on labels).  
- **Target variable:** `target_median_fw_g` (grams, median fresh weight at harvest).  
- **Image features:** Windowed means/std of mask area, plant-colour fraction, cup count, coverage, and 4-day growth deltas/slopes (`win_img_*` columns).  
- **Evaluation protocol for prediction (next stage):** leave-one-out cross-validation on eight checkpoints per trial; compare ≥4 regressors (Ridge, Random Forest, XGBoost, etc.) on identical splits.

ExGR and SAM results are retained as **benchmark comparisons** in this paper; only the FastSAM-derived feature tables are used for yield-model training and conclusion.

---

## References (segmentation & yield subset)

1. Chen et al. (2023). YOLO-EfficientNet for hydroponic lettuce growth-status identification. *Agriculture*.

2. Cornell eCommons (2024). *Computer vision and IoT based plant phenotyping and growth monitoring with 3D point clouds.* doi:10.7298/m9nx-g970

3. Gao et al. (2022). Automatic monitoring of lettuce fresh weight by multi-modal fusion based deep learning. *Frontiers in Plant Science*. doi:10.3389/fpls.2022.980581

4. Hyeseung et al. (2026). FLAsH: FastSAM-based lettuce analysis for high-density cultivation. *Smart Agricultural Technology*.

5. Kirillov et al. (2023). Segment anything. *arXiv:2304.02696*.

6. Knelshausen et al. (2020). Robust index-based semantic plant/background segmentation for RGB images. *Computers and Electronics in Agriculture*.

7. Li et al. (2024). Segmenting collective lettuce to predict fresh weight using three-dimensional point cloud. *Transactions of the CSAE*.

8. Meyer, G. E., & Neto, J. C. (2008). Verification of color vegetation indices for automated crop imaging applications. *Computers and Electronics in Agriculture*, 63(2), 282–293.

9. Nakano et al. (2023). Optimized excess-green image binarisation for accurate estimation of lettuce seedling leaf-area in a plant factory. *Environmental Control in Biology*. doi:10.2525/ecb.60.153

10. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE TSMC*.

11. Riehle et al. (2021). Vegetation indices for plant/soil segmentation in RGB images. *VISAPP*.

12. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. *MICCAI*.

13. Sharma et al. (2024). Integration of U-Net and FastSAM for accurate leaf image segmentation in complex backgrounds. *Engineering, Technology & Applied Science Research*. doi:10.48084/etasr.14464

14. Wang et al. (2022). TMSCNet: A three-stage multi-branch self-correcting trait estimation network for RGB and depth images of lettuce. *Plant Phenomics*.

15. Wang et al. (2024). Leaf segmentation using modified YOLOv8-seg models. *Life*.

16. Woebbecke, D. M., et al. (1995). Shape features for identifying young weeds using image analysis. *Transactions of the ASAE*.

17. Xu et al. (2023). Improving lettuce fresh weight estimation accuracy through RGB-D fusion. *Agronomy*. doi:10.3390/agronomy13102617

18. Xu et al. (2026). MIFFNet: A multidimensional image feature fusion network for lettuce fresh weight estimation. *Pattern Recognition*.

19. Zhao, X., et al. (2023). Fast segment anything. *arXiv:2306.12156*.

20. Zhao, Y., et al. (2024). Low-cost lettuce height measurement based on depth vision and lightweight instance segmentation model. *Agriculture*, 14(9), 1596.

---

*End of draft sections 2–4 (segmentation focus).*
