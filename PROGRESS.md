# Crab Burrow Segmentation — Progress Report

## Project Overview

This project develops automated detection of *Sesarma reticulatum* (purple marsh crab) burrow damage along salt-marsh channels using drone-collected high-resolution imagery (Wellfleet, MA). Crab burrowing is a key driver of New England marsh die-off and currently has to be mapped by hand from aerial photos or walking/kayak surveys — neither approach scales to the regional monitoring we need.

Our approach uses a two-tier deep-learning segmentation pipeline:

- **Model 1** trains on 1 cm imagery flown low along marsh channels and produces high-confidence polygons of damaged bank segments (5 bank-state classes plus "other").
- **Model 2** trains on 4 cm imagery flown at higher altitude across the full marsh, using Model 1's polygons as supervision plus hand-labeled non-bank polygons (trees, ponds, hummocks, mud, healthy marsh interior).

## Hardware and Imagery

- **Sensor**: MicaSense RedEdge-P (5 multispectral bands — B/G/R/RE/NIR — plus panchromatic)
- **Platform**: WingtraOne Gen II VTOL UAV
- **Spatial reference system**: EPSG:26919 (NAD83 / UTM zone 19N)
- **Low-altitude flight (M1)**: 1 cm panchromatic, 2 cm multispectral, pansharpened to 1 cm
- **High-altitude flight (M2)**: 4 cm multispectral

## Class Scheme

| Index | Class | Description |
|---|---|---|
| 0 | other | Non-bank features (water, trees, hummocks, ponds, marsh interior) |
| 1 | healthy_bank | Intact channel bank with healthy *Spartina* |
| 2 | eroding_non_crab | Bank erosion not caused by crabs (slumping, wave action) |
| 3 | crab_edge | Crab damage at channel edges (early-stage burrowing) |
| 4 | crab_platform | Crab damage on the marsh platform interior |
| 5 | collapsed | Collapsed/failed bank zones |

The pipeline uses an *open-world* labeling strategy: unlabeled pixels (ignore index 255) are excluded from training and evaluation, so the model learns only from explicitly labeled polygons. This avoids forcing exhaustive labeling of every pixel.

## What's Built

### 1. Shared Code Repository

A standalone GitHub repository (`crab_project/`) containing:

- `marsh_utils.py` — utility functions and project-wide constants (class scheme, conversion mappings, IO helpers, training and inference functions)
- `band_experiments.py` — harness for systematic band-combination experiments
- `synthetic/generate_synthetic_marsh.py` — synthetic data generator for pipeline testing
- `notebooks/` — Jupyter notebooks for training, production inference, and band experimentation
- `requirements.txt` — pinned dependencies

Notebooks pull the repo via `git clone` (with `git pull` for updates) at the top of each Colab session, so code is shared cleanly between training, production, and experimentation notebooks.

### 2. Synthetic Data Generator

A procedural synthetic-marsh generator produces realistic test datasets without requiring drone flights. This unblocked pipeline development ahead of the flight window. The data is driven by a screenshot given to Claude of entire Wellfleet marsh. Claude then used this to generate realistic data.

- Procedural channel and tributary geometry with biologically plausible characters (healthy, eroding, crab-damaged, mixed)
- Per-class spectral signatures calibrated to produce realistic NDVI/NDRE separation between classes
- DEM with channel-cutting topography (channels sit ~40 cm below marsh platform) and realistic surface roughness
- Two-tier output matching real-data layout: 60 m × 60 m at 1 cm for Model 1, 150 m × 150 m at 4 cm for Model 2
- Class balancing via biased character distribution so all 6 classes have adequate training representation

### 3. Derived Band Library

14+ derived band computers, each with an `ensure_*` wrapper that handles caching and dependency ordering:

**Spectral indices** (computed from pansharpened MS at 1 cm):
- NDVI, NDRE, SAVI, EVI, GNDVI, NDWI, CI-rededge

**DEM-derived geomorphology**:
- Slope (Sobel-based, with optional pre-smoothing)
- TPI (Topographic Position Index) at multiple scales — micro (5 cm), small (30 cm), large (2 m)
- Curvature, hillshade
- DEM roughness (local std)
- TRI (Terrain Ruggedness Index, Riley 1999)
- DEM range (max − min in local window)

**Channel-dependent bands**:
- Channel mask (derived from NDWI)
- Distance-to-channel raster
- Relative elevation above channel water level

**Texture bands**:
- Local std and Laplacian (from pan)
- Local range and local entropy (generic, applicable to pan, DEM, or any single-band raster)

### 4. Model 1 Training Pipeline

A complete training pipeline including:

- U-Net architecture with EfficientNet-B3 encoder, ImageNet-pretrained
- Multi-band input via configurable `BAND_SPEC`
- Spatial block-based train/val/test splits (anti-leakage at 100 m block size for real data)
- Mixed-precision training (AMP) with `GradScaler`
- Custom combined loss (cross-entropy + IoU) with `IGNORE_INDEX` handling
- Per-class IoU tracking during training
- Precision-coverage curves for per-class confidence threshold selection
- Class-channel permutation importance analysis for understanding which bands matter for which classes

### 5. Model 1 Production Inference

End-to-end inference for deployment on new flights:

- Tile imagery into patches matching training configuration
- Run U-Net inference with the trained model
- Apply per-class confidence thresholds calibrated on validation set
- Polygonize predicted masks per class
- Filter polygons by area and confidence
- Output GeoPackage / Shapefile compatible with QGIS for review

### 6. Band-Combination Experimentation Framework

`band_experiments.py` enables systematic comparison of band configurations:

- Declarative experiment definition (name + `band_spec` + setup steps)
- Resumable execution — skips experiments whose `summary.json` already exists
- Per-experiment artifacts: `best_model.pt`, `summary.json`, `channel_stats.json`
- Single-row-per-experiment CSV for quick at-a-glance comparison
- Multi-indexed DataFrame loaders for nuanced analysis:
  - Per-class IoU matrix across experiments
  - Per-band importance across classes
  - Per-class importance across bands
- Pandas-styler heatmaps and matplotlib publication-grade grids
- ~18 pre-defined experiments testing spectral, geomorphic, texture, and combined approaches

### 7. Decision Rules and Evaluation Tools

Beyond simple argmax, the pipeline supports multiple inference decision rules over the softmax outputs:

| Rule | What it does |
|---|---|
| `argmax` | Baseline — predict highest probability class |
| `argmax_abstain` | Predict only when max confidence exceeds threshold |
| `margin_abstain` | Predict only when top-2 classes are well-separated |
| `entropy_abstain` | Predict only when distribution is concentrated (low entropy) |
| `soft_cascade` | Aggregate P(bank classes) first, then disambiguate among bank classes |
| `priority` | Walk classes in priority order; first to pass threshold wins |

Each rule operates on cached softmax outputs, so threshold and rule sweeps run in milliseconds without re-inference. Confusion matrices (with abstention accounting) compare rules side-by-side.

## Preliminary Findings (Synthetic Data)

The synthetic-data confusion matrices revealed several insights ahead of real-data flights. These should be treated as hypotheses to confirm rather than conclusions:

1. **The dominant failure mode is intra-bank confusion**, specifically `crab_edge` getting predicted as `healthy_bank`. Roughly half of true `crab_edge` pixels go to `healthy_bank` under argmax. This is the discrimination that matters most ecologically and is the right target for further work.

2. **A two-stage (binary then fine-grained) cascade approach probably wouldn't help.** The confusion is between two bank classes, not bank-vs-not-bank. A first-stage "is it a bank" model would correctly route both confused classes to the second stage; the hard problem is the fine-grained discrimination within bank classes.

3. **Decision-rule choice affects the precision/recall tradeoff at export** but doesn't fix underlying class confusion — we can move along the precision-recall curve, not change the curve. The framework lets us pick an operating point appropriate for the downstream use case.

4. **DEM-derived roughness bands are the most promising features to add.** Crab burrows are literally surface holes; bands like TPI, TRI, and DEM range should pick up signal invisible to MS imagery. Real-data band experiments are the right place to confirm this.

5. **Synthetic results are not directly predictive of real-data results**, because synthetic data lacks the spectral and textural complexity of real marsh imagery. The infrastructure is validated; the band rankings will be revisited on real data.

## What's Pending

- Real drone flights (scheduled for the upcoming flight window)
- QGIS hand-labeling of crab polygons over real 1 cm imagery (Model 1 training data)
- Hand-labeling of 'other' polygons over real 4 cm imagery (Model 2 training data)
- Model 2 training notebook (structural copy of M1 with adjustments for the larger extent, lower resolution, and combined supervision sources)
- Model 2 production notebook
- Re-run of band experiments on real data
- Per-flight metadata capture: panel times, sun angle, cloud cover, tide stage
- Kayak ground-truth observations for validation
- Ecologist sign-off on the 6-class scheme

## Tech Stack

- Python 3.12, PyTorch 2.x, segmentation-models-pytorch
- Rasterio, GeoPandas, Shapely for geospatial I/O
- Albumentations for augmentation
- scikit-learn for evaluation metrics
- Pandas + Matplotlib for results analysis
- QGIS for label creation and result visualization
- Google Colab for training compute, Google Drive for data persistence
- GitHub for code; shared module structure for cross-notebook consistency

## Status Summary

| Component | Status |
|---|---|
| Shared utility library (`marsh_utils.py`) | Complete |
| Derived band library | Complete (14+ bands) |
| Synthetic data generator | Complete, validated end-to-end |
| Model 1 training notebook | Complete |
| Model 1 production notebook | Complete |
| Band-experiments notebook + framework | Complete |
| Model 2 training notebook | Pending (post real-data labeling) |
| Model 2 production notebook | Pending |
| Real-data labeling | Pending (post flights) |
| Ecologist class-scheme review | Pending |
