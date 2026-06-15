# Crab Burrow Segmentation — Progress Report

## Stephen Fickas, June 2026

## Project Overview

This project develops automated detection of *Sesarma reticulatum* (purple marsh crab) burrow damage along salt-marsh channels using drone-collected high-resolution imagery (Wellfleet, MA). Crab burrowing is a key driver of New England marsh die-off and currently has to be mapped by hand from aerial photos or kayak surveys — neither approach scales to the regional monitoring we need.

We deliberately chose semantic segmentation (pixel-level classification) over object detection (e.g., YOLO bounding boxes). This reflects what we ultimately want to measure: rather than counting individual burrows to estimate abundance, we quantify the *area* of marsh affected by crab damage. Damage extent is more directly tied to the ecological outcome (marsh loss, lost vegetation, lost sediment-trapping capacity) and more robust to the imaging variations — lighting, partial occlusion, burrow age, vegetation regrowth — that make consistent individual-burrow detection difficult, particularly at the 4cm resolution we have to work with.

Our approach uses a two-tier deep-learning segmentation pipeline:

- **Model 1** trains on 1 cm imagery flown low along marsh channels and produces high-confidence polygons of damaged bank segments (5 bank-state classes plus "other").
- **Model 2** trains on 4 cm imagery flown at a higher altitude across the full marsh, using Model 1's polygons as supervision plus hand-labeled non-bank polygons (trees, ponds, hummocks, mud, healthy marsh interior).

## Hardware and Imagery

- **Sensor**: MicaSense RedEdge-P (5 multispectral bands — B/G/R/RE/NIR — plus panchromatic)
- **Platform**: WingtraOne Gen II VTOL UAV
- **Spatial reference system**: EPSG:26919 (NAD83 / UTM zone 19N)
- **Low-altitude flight (M1)**: 1 cm panchromatic, 2 cm multispectral, pansharpened to 1 cm
- **High-altitude flight (M2)**: 4 cm multispectral

## Class Scheme

Note this is preliminary and may change as we get to view actual flight images. I have built it as a parameter that is easy to change in the pipeline.

Also note that, contrary to at least one paper the team has referenced, the U-Net architecture has no problem with multiple classes.

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

A procedural synthetic-marsh generator produces realistic test datasets without requiring drone flights. This unblocked pipeline development ahead of the flight window. Claude created this generator using a screenshot from QGis of the entire Wellfleet marsh along with images collected by team members who took photos while walking or kayaking the marsh. Fairly impressive.

- Procedural channel and tributary geometry with biologically plausible characters (healthy, eroding, crab-damaged, mixed)
- Per-class spectral signatures calibrated to produce realistic NDVI/NDRE separation between classes
- DEM with channel-cutting topography (channels sit ~40 cm below the marsh platform) and realistic surface roughness
- Two-tier output matching real-data layout: 60 m × 60 m at 1 cm for Model 1, 150 m × 150 m at 4 cm for Model 2
- Class balancing via biased character distribution so all 6 classes have adequate training representation

**How channels and bank states are generated.** The full synthetic marsh contains:

- One main channel (sinuous, with realistic noise)
- 8 main tributaries branching off at regular intervals
- ~10-20 sub-tributaries branching off the main tributaries

Each tributary is assigned a *character* — `healthy`, `mixed`, `crab`, or `eroding` — that controls the probability distribution of bank classes along its length. The bank zone of each channel is then split into segments, and each segment is sampled from its tributary's character distribution:

| Character | healthy_bank | eroding_non_crab | crab_edge | crab_platform | collapsed |
|---|---|---|---|---|---|
| healthy | 70% | 20% | 7% | 2% | 1% |
| eroding | 20% | 55% | 15% | 5% | 5% |
| crab | 5% | 10% | 35% | 30% | 20% |
| mixed | 35% | 20% | 20% | 15% | 10% |

The character distribution across the marsh's 8+ tributaries is biased toward `crab` and `eroding` to over-represent the rare bank-damage classes. Without this bias, the rare classes (`collapsed` especially) would have too few training pixels to learn from. With it, we get roughly:

| Class | Polygons per marsh (8 tribs + biased characters) |
|---|---|
| healthy_bank | ~15 |
| eroding_non_crab | ~30 |
| crab_edge | ~40 |
| crab_platform | ~30 |
| collapsed | ~20 |

This is more crab-damage-dense than a real marsh — which is the point: ensure the model sees enough examples of each class to learn it, since real-data scarcity is something we can't control.

**Why the spatial split needs care on synthetic data.** Our anti-leakage train/val/test split assigns entire 100 m blocks to each set (to prevent train/val patches from overlapping spatially). On real marsh imagery, hundreds of meters across, this is fine. On synthetic data, the small extents (30-60 m) yield very few blocks at 100 m, so the split often produces empty val or test sets. We address this by using `BLOCK_SIZE_M=3` (synthetic Model 1) or `BLOCK_SIZE_M=15` (synthetic Model 2), which still maintains spatial separation but produces 100+ blocks per dataset. The `EXTENT_M // 10` heuristic falls out of this and recovers `BLOCK_SIZE_M=100` for real-data extents automatically.

### 3. Derived Band Library

The model takes a configurable list of input bands — the panchromatic image alone is informative, but pairing it with derived bands gives the network access to physical signals (vegetation health, surface roughness, channel proximity) that are otherwise hard to learn from raw pixel values. We've implemented 14+ derived bands across four categories, each with an `ensure_*` wrapper that handles caching and dependency ordering.

Note that it may seem intuitive to add most or all of these bands into each image given to model 1, i.e., each image contains 14 bands of information. Isn't more information better? In general, no. In practice, 3 bands and perhaps a few more are typically the sweet spot. The challenge is to find the smallest subset of the bands that yields the best results. Below are the bands that are being considered.

**Spectral indices** (computed from the pansharpened MS at 1 cm). These transform the 5 raw multispectral bands into single-channel rasters that respond to specific properties of the surface:

| Band | Formula | What it captures | Why relevant |
|---|---|---|---|
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation greenness / chlorophyll absorption | Healthy *Spartina* high; bare mud and burrow-damaged surface low |
| NDRE | (NIR − RedEdge) / (NIR + RedEdge) | Same as NDVI but less saturated at high biomass | Sensitive to subtle plant stress before NDVI signals it |
| SAVI | 1.5 × (NIR − Red) / (NIR + Red + 0.5) | Soil-adjusted vegetation index | More robust when soil/mud contributes to the pixel signal |
| EVI | 2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1) | Enhanced vegetation index | Less saturated than NDVI at very high biomass, uses Blue for atmospheric correction |
| GNDVI | (NIR − Green) / (NIR + Green) | Green-based vegetation index | More chlorophyll-a sensitive than NDVI |
| NDWI | (Green − NIR) / (Green + NIR) | Water content / open water detection | Identifies channel water; used to derive channel mask |
| CI-rededge | NIR / RedEdge − 1 | Red-edge chlorophyll index | Highly sensitive to chlorophyll content in vegetation |

To give a concrete sense of how separable the classes are spectrally, the per-class NDVI and NDRE values from our synthetic-data spectra are:

| Class | NDVI | NDRE | Interpretation |
|---|---|---|---|
| water | −0.50 | −0.33 | Strongly negative — water absorbs NIR |
| collapsed | −0.24 | −0.13 | Negative — bare mud, no vegetation |
| crab_platform | −0.02 | −0.03 | Near zero — heavily damaged, no chlorophyll signal |
| crab_edge | 0.00 | −0.02 | Near zero — actively damaged surface |
| eroding_non_crab | +0.25 | +0.08 | Mild positive — sparse vegetation on eroding bank |
| marsh_platform | +0.71 | +0.25 | Strongly positive — healthy *Spartina* on platform |
| healthy_bank | +0.77 | +0.27 | Strongly positive — healthy bank vegetation |
| tree | +0.88 | +0.29 | Very high — dense tree canopy |

Two things this shows: (1) NDVI does most of the work — the spread from −0.5 to +0.88 is enormous; (2) several damage classes (`crab_edge`, `crab_platform`) cluster tightly near zero. This is part of why these classes are hard to distinguish from each other with spectral information alone, and why geomorphic and texture bands are likely to help on real data.

**DEM-derived geomorphology**. These transform elevation into per-pixel measures of surface shape and position. They are especially relevant because crab burrows are *literal holes* in the surface — geomorphic bands should expose signals invisible in MS imagery:

| Band | Method | What it captures | Why relevant |
|---|---|---|---|
| Slope | Sobel-based gradient magnitude, optional pre-smoothing | Local steepness | Distinguishes flat platform from steep banks |
| TPI-micro | Center pixel elevation minus mean over a 5 cm radius circular neighborhood | Cm-scale elevation anomalies | Picks up individual burrow pock-marks (typical burrow diameter ~3 cm) |
| TPI-small | Same, 30 cm radius | Decimeter-scale elevation anomalies | Picks up bank-edge transitions and small slump scarps |
| TPI-large | Same, 2 m radius | Meter-scale elevation anomalies | Captures the bank-vs-platform mass effect — wide ridges or depressions |
| Curvature | Second derivative of elevation | Surface convexity / concavity | Banks have characteristic curvature signatures |
| Hillshade | Simulated illumination from solar geometry | Synthetic shading — visualization aid | For QGIS review, not modeling |
| DEM roughness | Local standard deviation in a window | How rough the surface is | Burrowed areas have higher elevation variance than smooth marsh platform |
| TRI (Terrain Ruggedness Index) | Mean absolute difference between a pixel and its 8 neighbors (Riley 1999) | Local surface ruggedness | Classical ruggedness measure for fine-scale topography |
| DEM range | Max − min elevation in a window | Direct measure of elevation variability | Crab burrows show up as elevation pock-marks |

**Channel-dependent bands**. These are functions of channel location and capture the marsh's hydraulic and geometric organization around channels. They require the channel mask as an intermediate:

| Band | Method | What it captures | Why relevant |
|---|---|---|---|
| Channel mask | NDWI threshold + morphological cleanup | Binary water / not-water mask | Intermediate output; required for the other channel-dependent bands |
| Distance to channel | Euclidean distance transform from channel mask | How far each pixel is from the nearest channel water | Bank damage classes occur close to channels; this band makes that explicit |
| Relative elevation | DEM minus interpolated channel-water elevation | Height above the local channel water level | Normalizes terrain so a bank in one part of the marsh is comparable to a bank elsewhere with different absolute elevation |

**Texture bands**. These operate on a single-band raster (typically pan, but also work on DEM, NDVI, etc.) and measure spatial heterogeneity. They are generic — the same function can produce different bands depending on the source raster:

| Band | Method | What it captures | Why relevant |
|---|---|---|---|
| Local std | Standard deviation in a window | Pixel-value variance | Damaged or burrowed surfaces are visually more heterogeneous than smooth marsh |
| Laplacian | 2nd-derivative filter | Edge / discontinuity strength | Edges of bank slumps and burrow rims |
| Local range | Max − min in a window | Range of values | Simpler alternative to local std; very sensitive to outliers (burrows!) |
| Local entropy | Shannon entropy of quantized values in a window | Spatial information content | Captures texture complexity beyond simple variance |

**On window sizes for texture and DEM-roughness bands**: the right window depends on the feature scale we're trying to surface. Our defaults:

| Feature of interest | Typical scale | Window setting |
|---|---|---|
| Individual burrows / pock-marks | ~3-5 cm | 5-10 cm window |
| Burrow clusters, small bank features | ~20-30 cm | 30 cm window (our most common default) |
| Bank-vs-platform texture differences | ~1-2 m | 1-2 m window |

The same compute function (`compute_local_range`, `compute_local_std`, etc.) can be called multiple times with different window sizes to produce a multi-scale stack — analogous to multi-scale TPI. We haven't yet exhaustively explored multi-scale texture variants; that's a follow-up if single-scale results are insufficient.

Each band is computed once and cached as a GeoTIFF, then loaded as needed during training. The `BAND_SPEC` configuration in the training notebook controls which bands the model actually sees as input channels — letting us compare combinations without re-computing the derived bands themselves.

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

**On per-class confidence thresholds (precision-coverage).** During training, the network learns a 6-class softmax — for any pixel, six probabilities that sum to 1.0. A naïve inference rule (argmax) just picks the highest-probability class. But the ecological cost of false positives versus false negatives differs by class — for example, mistakenly calling healthy marsh "collapsed" wastes survey time, while missing actual collapsed marsh defeats the purpose. So we want a per-class confidence threshold that reflects each class's precision/recall tradeoff.

The way `pick_thresholds` works: for each class C, we walk the validation set and compute, at every candidate confidence threshold from 0.1 to 0.95, the precision and recall (or coverage) that threshold yields. Plotted, this is a downward-curving graph — higher thresholds mean higher precision but lower recall. We pick the threshold that hits a target precision (we use 0.9 by default), giving us the lowest confidence at which we can trust class-C predictions.

The output is a dictionary like `{3: 0.62, 4: 0.55, 5: 0.71}` — meaning a pixel predicted as `crab_edge` is only emitted as a polygon if its softmax probability for class 3 exceeds 0.62. Different classes get different thresholds because their precision-recall curves have different shapes. We compute thresholds only for `CLASSES_OF_INTEREST` (the classes we actually export — `crab_edge`, `crab_platform`, `collapsed`); the other classes use defaults.

**On permutation importance.** After training, we measure how much each input band actually contributes to model performance by shuffling that band's pixels (breaking its spatial correlation) and re-evaluating. The drop in per-class IoU is the band's importance. We report importance per (band, class) so we can see, for example, that NDVI matters a lot for `healthy_bank` but not for `crab_edge`. This drives the band-experiment iteration: bands with low importance for the classes that matter can probably be replaced with better candidates.

### 5. Model 1 Production Inference

End-to-end inference for deployment on new flights:

- Tile imagery into patches matching training configuration
- Run U-Net inference with the trained model, writing a per-pixel softmax raster (one float band per class) via Gaussian-weighted patch blending — this cached softmax is the single source artifact for the decision rules and the abstain bucket (section 8), so thresholds can be re-tuned without re-running the model
- Apply per-class confidence thresholds calibrated on validation set
- Polygonize predicted masks per class (now with optional exclusion of abstain-bucket pixels — see section 8 — so the confident map and the review queue partition cleanly)
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

**A worked example — `margin_abstain` and what it tells us.** Consider three pixels with the following 6-class softmax outputs:

| Pixel | other | healthy_bank | eroding | crab_edge | crab_platform | collapsed | argmax | top-2 margin |
|---|---|---|---|---|---|---|---|---|
| A | 0.05 | 0.85 | 0.05 | 0.03 | 0.01 | 0.01 | healthy_bank | 0.80 (confident) |
| B | 0.02 | 0.46 | 0.03 | 0.42 | 0.05 | 0.02 | healthy_bank | 0.04 (close!) |
| C | 0.03 | 0.18 | 0.21 | 0.19 | 0.20 | 0.19 | eroding | 0.01 (spread) |

`argmax` gives the same kind of answer for all three. But:

- Pixel A: model is confident — keep as `healthy_bank`
- Pixel B: model is hedging between `healthy_bank` and `crab_edge` — this is exactly the failure mode we see in confusion matrices. `margin_abstain(min_margin=0.15)` flags this as uncertain and abstains
- Pixel C: model has no idea — entropy-abstain catches this case more naturally

The point of `margin_abstain` over `argmax_abstain` is that it specifically detects *two-way ties* — and we can report which two classes were tied. If 80% of abstentions are `(healthy_bank, crab_edge)` pairs, we've found a specific actionable problem (these two classes need better discrimination) rather than a generic uncertainty issue. This pair-reporting is what the active-learning loop (section 8) is built on.

### 8. Active-Learning Labeling Loop

The original plan for real data was exhaustive QGIS hand-labeling of crab polygons over the 1 cm imagery. That doesn't scale: a real marsh is hundreds of meters across, and most of it is unambiguous (open water, healthy interior) that the model already gets right. Hand-labeling all of it spends the expert's time where it adds no information.

Instead, we now close an *active-learning* loop. Train Model 1, run it over the full raster, and use the model's own uncertainty to point the labeler at exactly the regions where a new label would teach it the most. The expert labels only those regions; the labels feed back into training; repeat. This is a well-established paradigm — superpixel-based active learning for semantic segmentation (Cai et al., CVPR 2021; Kim et al., ICCV 2023), with a long lineage in remote sensing (e.g. Tuia and colleagues on superpixel-based active learning for land cover). The one local twist is that we tag each review region with the *contested class pair* the model is torn between, which turns the queue into a set of specific, answerable two-way questions rather than a generic "uncertain" mask.

**The loop, end to end:**

1. **Cache the softmax.** Model 1 production already writes a per-pixel softmax raster (one float band per class) rather than hard labels. This is the single source artifact, and because it is cached, the abstain thresholds below are re-tunable with cheap CPU passes — no model re-run.

2. **Build the abstain bucket.** `build_abstain_raster` derives, from the cached softmax, a per-pixel `uint8` raster of *contested-pair codes*. For each pixel it takes the top-2 margin; where the margin is below `min_margin` it flags the pixel as either a *pair* (a specific two-way tie, codes 1–15, recording which two classes) when the top-2 hold most of the probability mass, or *diffuse* (code 100) when the mass is smeared across many classes. A JSON legend decodes the pair codes to class names, and `require_classes` scopes the flagging to ties involving the classes we care about.

3. **Partition the prediction.** `predictions_to_polygons` now optionally takes the abstain raster and *excludes* its flagged pixels from the confident per-class output. The result is a clean split governed by a single margin threshold: confident pixels become the prediction map (the deliverable), uncertain pixels become the review queue, and every pixel lands in exactly one bucket.

4. **Build superpixel containers.** Per-pixel uncertainty cannot be labeled by hand — it is thousands of scattered pixels. So we segment the imagery into *superpixels* (SLIC) — appearance-homogeneous pixel clumps that respect real edges — computed from a small curated band set: pan for structure, NDVI for the vegetation/damage axis, and a geomorphic band such as `tpi_small` for the spectrally-subtle crab/healthy edge. A per-band weighting lets the geomorphic band lead when busy pan texture would otherwise dominate the clustering. Superpixels are model-independent and built once per flight; their boundaries are also exported as a GeoPackage (with a QGIS style sidecar) for overlay on the ortho.

5. **Promote review polygons.** `build_abstain_review_polygons` intersects the abstain raster with the superpixel containers. For each container it computes the fraction of pair-abstain pixels and the dominant contested pair, then promotes containers above an abstain-fraction threshold into a GeoPackage review layer — one labelable unit each, tagged with its contested pair, abstain fraction, and pair purity, and sorted into a work queue.

6. **Label and retrain.** The expert opens the review layer in QGIS over the imagery (categorized by contested pair) and, for each container, answers the question the model framed — which of the two classes is this? — assigning the class, or splitting the container where it straddles a real boundary. Because we label open-world, these are simply additional labeled regions added to the sparse set; everything unlabeled stays ignore. The labels rasterize into Model 1 supervision, the model retrains, and the next production pass yields a smaller, sharper abstain bucket. The loop turns.

**Design decisions worth recording.**

- *Margin is the abstain rule, not a free choice.* The review queue depends on knowing *which two* classes are tied — the contested pair — and margin (top-1 minus top-2) is the only standard acquisition rule that yields a clean pair. `argmax_abstain` and `entropy_abstain` detect uncertainty but not a pair; `priority` and `soft_cascade` are prediction rules. So the abstain bucket is tuned via `min_margin`, not by swapping rules.
- *Pairs drive promotion; diffuse is set aside.* Pair abstentions are answerable two-way questions and make high-yield labels. Diffuse abstentions are usually genuinely low-signal pixels, or a signal about data problems or a missing class. They are tracked (`diffuse_frac`) but do not promote, though they remain a diagnostic to mine later.
- *Uniform container size is deliberate.* SLIC partitions into a roughly fixed number of equal-size containers and never merges adjacent ones, so a uniform stretch of water is carved into many same-size tiles. That is desirable here: the redundant water containers cost nothing (near-zero abstain fraction, never promoted), and a roughly constant container size keeps the abstain *fraction* meaning the same thing everywhere — a small crab cluster lands in its own container where it can trip the threshold, rather than being diluted inside a giant merged region.
- *Heterogeneity is handled, not eliminated.* A container can straddle a class boundary (low `pair_purity`). This is managed by appearance-homogeneous containers (small enough that most are pure), the contested-pair tag (which bounds the heterogeneity to two classes), the human split in QGIS, and optionally an automatic sieving step (excluding confidently-disagreeing pixels from the loss, in the spirit of Kim et al.) for gross spillover.

**Components added.**

- Superpixel builder (`build_superpixels`, `build_superpixels_from_bands`, `ensure_superpixels_from_bands`) — SLIC over a curated, optionally weighted band set; writes a `uint32` superpixel-ID GeoTIFF. Plus `write_boundaries_gpkg` for the QGIS overlay layer (no-fill outlines, `DN` = superpixel id) with a `.qml` style sidecar.
- Abstain-bucket builder (`build_abstain_raster`) — margin-based contested-pair codes plus diffuse, JSON legend, `require_classes` scoping.
- Partition edit to `predictions_to_polygons` — optional `abstain_raster_path` that withholds flagged pixels from the confident map.
- Review-polygon builder (`build_abstain_review_polygons`) — container aggregation, abstain-fraction promotion, contested-pair tagging, work-queue GeoPackage.
- Alignment checker (`check_alignment`) — verifies the softmax, abstain, and superpixel rasters share CRS, transform, and shape before any intersection.

**Parameters to set on real data.** The synthetic values are placeholders; the following are set once real results can be seen, roughly in this order:

- Superpixel `weights`, `compactness`, `target_superpixel_px` — set from the boundary overlay (on the geomorphic band, not just pan) before any labeling; raise the geomorphic weight until containers follow real burrow texture.
- `min_margin` and `mass_cutoff` (abstain bucket) and the per-class `CONFIDENCE_THRESHOLDS` (recomputed from scratch via `pick_thresholds` on a real validation set) — follow the first round of real labeling and the real confusion matrix.
- `min_abstain_frac` (promotion) — set against the labeling budget and the real distribution of container abstain fractions.
- `require_classes` — set from which confusions actually occur on real data (see the tie-diagnostics discussion below).

The pipeline architecture, the use of margin for the abstain bucket, and the class scheme itself are *not* data-tuned knobs and should not be revisited as part of this parameter sweep.

## Preliminary Findings (Synthetic Data)

The synthetic-data confusion matrices revealed several insights ahead of real-data flights. Given both the speculative class breakout and the synthetic nature of this data, I would use this more as a guide to the kinds of things we may find in our final results.

1. **The dominant failure mode is intra-bank confusion**, specifically `crab_edge` getting predicted as `healthy_bank`. Roughly half of true `crab_edge` pixels go to `healthy_bank` under argmax. This is the discrimination that matters most ecologically and is the right target for further work. (Note: this finding is from the early `pan + NDVI + NDRE` baseline; on the latest model the synthetic confusion has largely closed — see the update at the end of this section.)

2. **A two-stage (binary then fine-grained) cascade approach probably wouldn't help.** The confusion is between two bank classes, not bank-vs-not-bank. A first-stage "is it a bank" model would correctly route both confused classes to the second stage; the hard problem is the fine-grained discrimination within bank classes.

3. **Decision-rule choice affects the precision/recall tradeoff at export** but doesn't fix underlying class confusion — we can move along the precision-recall curve, not change the curve. The framework lets us pick an operating point appropriate for the downstream use case.

4. **DEM-derived roughness bands are the most promising features to add.** Crab burrows are literally surface holes; bands like TPI, TRI, and DEM range should pick up signal invisible to MS imagery. Real-data band experiments are the right place to confirm this.

5. **Synthetic results are not directly predictive of real-data results**, because synthetic data lacks the spectral and textural complexity of real marsh imagery. The infrastructure is validated; the band rankings will be revisited on real data.

**Concrete numbers from an early synthetic run** (baseline `pan + NDVI + NDRE`, validation set, recall normalization):

| True class | Predicted (top-2 destinations) | Recall on diagonal |
|---|---|---|
| healthy_bank | → healthy_bank | 1.00 (clean) |
| crab_edge | → healthy_bank (44%), crab_edge (49%) | 0.49 (the failure mode) |
| crab_platform | → crab_platform (96%), eroding (3%) | 0.96 (clean) |

The `crab_edge` → `healthy_bank` confusion (44% of true crab_edge pixels misclassified) was the dominant signal in that early run. `crab_platform` and `healthy_bank` are well-separated, so the network has the capacity to discriminate when the spectral / spatial difference is large enough — `crab_edge` was the regime where the synthetic signal wasn't sharp enough.

A separate caveat: an earlier training run also had several validation classes with zero pixels because of the spatial-split sparsity (small synthetic extent + 100 m block size). After widening the synthetic marsh to 60 m and dropping `BLOCK_SIZE_M` to 3, the validation set now has 5 of 6 classes represented (only `other` remains sparse because the 'other' polygons mostly sit in the M2 extent rather than the M1 window). This is good enough for synthetic-stage evaluation; class-0 representation will be addressed naturally by real-data labeling.

### Update — rule experiments and margin-tie diagnostics (latest synthetic model)

The full decision-rule sweep and the abstain machinery were re-run on the most recent Model 1 (a stronger band set than the `pan + NDVI + NDRE` baseline above). Two things stand out, both consistent with the active-learning design.

**The model is now near-saturated on synthetic data.** Under argmax, `crab_edge` recall is 0.86 (only ~6% leaking to `healthy_bank`), a large improvement over the 0.49 baseline — the synthetic `crab_edge`/`healthy_bank` confusion that was the earlier "dominant failure mode" has largely closed on this data. Across all rules — argmax, the priority and soft-cascade variants, and the abstain rules — the diagonals are within a point or two of each other and abstention rates are tiny (0.3%–2.5%). The rules behave exactly as designed: entropy abstains the least and is the least-targeted detector of this confusion (because the confusion is low-entropy), while `argmax_abstain` trades the most coverage for the best diagonal. This validates the implementation, but it also means synthetic has little headroom left to differentiate rules — the production rule choice and the abstain thresholds wait for real data, where genuine ambiguity will populate the bucket. (`priority_crab_only` is the one structural outlier, and correctly so: it routes all non-crab classes to "other" while preserving the crab classes — a crab-vs-everything operating point, not an improvement on the hard discrimination.)

**The margin-tie diagnostic reorders the story.** Reporting which pairs of classes the model is most often torn between, at `min_margin < 0.15` (0.8% of pixels):

| Tied pair | Count | Type | Effect on damage-*area* metric |
|---|---|---|---|
| crab_edge ↔ crab_platform | 8,803 | within-damage | none (both count as damage) |
| eroding_non_crab ↔ crab_platform | 8,073 | boundary-crossing | flips pixel in/out of damage total |
| healthy_bank ↔ eroding_non_crab | 6,280 | non-damage | none |
| healthy_bank ↔ crab_edge | 5,112 | boundary-crossing | flips pixel in/out of damage total |
| crab_edge ↔ collapsed | 2,322 | within-damage | none |

Note that this is *hesitation*, not error: both crab classes have high recall, so these are close calls the model usually resolves correctly — but they are exactly the high-information pixels for the labeling queue. The `crab_edge`↔`crab_platform` dominance is consistent with the spectral table, where both classes sit near NDVI ≈ 0.

This raises a labeling-priority question tied to our headline metric. Because we measure *area* of crab damage, a tie's value depends on whether resolving it changes whether a pixel counts as damage at all (the "Type" column above). The single largest tie (`crab_edge`↔`crab_platform`) barely moves the area metric, while the metric-relevant ties are the boundary-crossing ones. The current `require_classes` filter keeps any tie involving a listed class ("at least one in the set"), which *cannot* express "keep only boundary-crossing pairs" — that would need a separate "exactly one damage class" filter. We are deferring that refinement to real data: the tie structure will reorder on real imagery, and within-damage labels still sharpen the model even when they don't move the area metric, so the boundary-crossing filter only becomes worthwhile if labeling volume turns out to be the bottleneck. Whether `collapsed` counts as "damage" for this grouping is one more question for the ecologist sign-off.

## What's Pending

- Real drone flights (scheduled for the upcoming flight window)
- QGIS hand-labeling of crab polygons over real 1 cm imagery (Model 1 training data) — now driven by the active-learning review queue (section 8) rather than exhaustive labeling
- Hand-labeling of 'other' polygons over real 4 cm imagery (Model 2 training data)
- Model 2 training notebook (structural copy of M1 with adjustments for the larger extent, lower resolution, and combined supervision sources)
- Model 2 production notebook
- Re-run of band experiments on real data
- First-flight tuning of the active-learning parameters (superpixel weights/compactness/size from the boundary overlay; abstain `min_margin`/`mass_cutoff` and per-class `CONFIDENCE_THRESHOLDS` from the first real validation set; `min_abstain_frac` from the labeling budget)
- Running the active-learning loop on real flights (predict → abstain bucket → review queue → label → retrain), for as many rounds as the residual uncertainty warrants
- Decision on the damage-area pair filter / damage grouping (with the ecologist), if labeling volume warrants the boundary-crossing refinement
- Per-flight metadata capture: panel times, sun angle, cloud cover, tide stage
- Kayak ground-truth observations for validation
- Ecologist sign-off on the 6-class scheme

## Tech Stack

- Python 3.12, PyTorch 2.x, segmentation-models-pytorch
- Rasterio, GeoPandas, Shapely for geospatial I/O
- scikit-image for SLIC superpixels and morphology
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
| Model 1 production notebook | Complete (now caches softmax + partitions abstain pixels) |
| Band-experiments notebook + framework | Complete |
| Superpixel container builder | Complete (synthetic-validated) |
| Abstain-bucket builder | Complete (synthetic-validated) |
| Review-polygon (active-learning) builder | Complete (synthetic-validated) |
| Raster-alignment checker | Complete |
| Model 2 training notebook | Pending (post real-data labeling) |
| Model 2 production notebook | Pending |
| Real-data labeling | Pending (post flights) |
| Active-learning loop on real data | Pending (post flights) |
| Ecologist class-scheme review | Pending |

## Glossary

For readers from different backgrounds — terms used throughout this report.

**Abstain bucket** — The set of pixels where the chosen decision rule (margin) declines to commit a label, carved out of the prediction and routed to review instead. Stored as a per-pixel raster of contested-pair codes (section 8).

**Active learning** — A training strategy in which the model's own uncertainty selects which unlabeled data to label next, so expert effort goes where it most improves the model rather than to data the model already handles. Here, Model 1's uncertain regions are surfaced for targeted QGIS labeling and fed back into training.

**Argmax** — Machine learning models typically provide a probability for each class, given an input image. These probabilities are normalized to 1 across all classes, e.g., with 6 classes, the 6 probabilities would add to 1.0. This is called softmax. Argmax is a decision rule that selects the highest-probability class from a softmax output. The simplest possible inference rule. The problem arises when no single probability is dominant. That leads us to consider other approaches (see section 7).

**BAND_SPEC** — Configuration list specifying which derived bands the model receives as input channels (e.g., `[('pan_orthomosaic', 1), ('ndvi', 1), ('tpi_small', 1)]`).

**Block size (BLOCK_SIZE_M)** — Spatial-block size for our anti-leakage train/val/test split. Whole blocks go to one set, ensuring train and val patches don't overlap spatially. 100 m for real data, 3-15 m for synthetic.

**Channel (creek)** — A tidal-flow waterway through a salt marsh. The narrow channels that *Sesarma* burrow into are the focus of this project.

**Confusion matrix** — Table where rows are true classes and columns are predicted classes; the diagonal is correct predictions and off-diagonal entries are specific errors. Used to identify *which* classes are getting confused with *which* others.

**Container** — A single superpixel treated as one labelable unit in the active-learning loop. The abstain bucket is aggregated to containers, and high-abstain containers are promoted to the review queue.

**Contested pair** — The two classes a pixel's softmax is split between under a margin tie (e.g. `healthy_bank | crab_edge`). Recorded per pixel and per review container, so each labeling unit carries the specific two-way question the model couldn't answer.

**CRS (Coordinate Reference System)** — System for mapping 2D coordinates to real-world locations. We use EPSG:26919 (NAD83 / UTM zone 19N).

**DEM (Digital Elevation Model)** — Raster where each pixel's value is the surface elevation. Produced from drone imagery via photogrammetry.

**Encoder / U-Net** — U-Net is the deep-learning architecture used here. Its "encoder" half progressively reduces spatial resolution while learning hierarchical features; the "decoder" half upsamples back to pixel-level predictions, with skip connections from the encoder.

**EPSG** — Standard code system for spatial reference systems, originated by the European Petroleum Survey Group. EPSG:26919 corresponds to the UTM zone covering New England.

**GeoPackage / Shapefile** — File formats for vector geometry. Shapefile is the legacy (and most widely supported) format; GeoPackage is the modern alternative, and the one we now prefer (single file, no field-name truncation, multiple layers per file).

**GSD (Ground Sampling Distance)** — Spatial resolution of imagery: real-world distance covered by one pixel. 1 cm GSD = 1 cm per pixel. Determined by sensor and flight altitude.

**Hand-labeling** — Process of an expert drawing polygons in QGIS over imagery, providing ground truth that the model trains against. Under the active-learning loop, the expert mostly confirms/splits pre-drawn review containers rather than digitizing from scratch.

**IGNORE_INDEX (255)** — Special pixel label that the loss function skips during training. Used for unlabeled pixels in the open-world labeling scheme. Pixels with this value contribute nothing to training or evaluation.

**ImageNet** — Massive labeled image dataset used to pre-train general-purpose image classifiers. Our U-Net encoder is initialized from ImageNet weights, then fine-tuned on the marsh task.

**IoU (Intersection over Union)** — Segmentation accuracy metric for a single class: (predicted ∩ true) / (predicted ∪ true). Range 0-1; 1 = perfect overlap. Also called Jaccard index.

**Margin (top-2 margin) / margin sampling** — The gap between the highest and second-highest class probabilities for a pixel. A small margin marks a two-way tie. Selecting low-margin pixels for labeling is "margin sampling," a standard active-learning acquisition rule, and the one used for the abstain bucket because it yields the contested pair.

**mIoU (mean IoU)** — Mean IoU across all classes; a single-number summary of model performance.

**Multispectral (MS)** — Imagery captured in multiple narrow spectral bands. Our MicaSense camera captures 5 bands: Blue (475 nm), Green (560 nm), Red (668 nm), Red-edge (717 nm), and Near-Infrared (842 nm).

**NDVI / NDRE / etc.** — Spectral indices computed from MS bands (see section 3 tables for formulas and meaning).

**NIR (Near-Infrared)** — Spectral band at ~842 nm, just beyond the visible range. Strongly reflected by healthy vegetation, absorbed by water — the basis for several vegetation indices.

**Open-world labeling** — Approach where only some pixels are labeled and the rest are marked as ignore. The model learns from labeled pixels only. Contrast with closed-world labeling, where every pixel must be assigned a class.

**Orthomosaic** — Composite image stitched from many drone photos and reprojected to remove perspective distortion. The "ortho-" prefix means perspective effects (taller objects appearing displaced) have been corrected.

**Panchromatic (pan)** — Single-channel wide-band imagery, typically at higher resolution than the multispectral bands. Our pan is 1 cm GSD; the multispectral bands are 2 cm.

**Pansharpening** — Algorithm that combines a low-resolution multispectral image with a high-resolution panchromatic image to produce a high-resolution multispectral image. Our `pansharp_5band.tif` is the output: 5 spectral bands at 1 cm.

**Patches** — Small square crops of the imagery (e.g., 512×512 pixels) that the model trains on. The full marsh imagery is too big to feed into the network all at once.

**Precision / Recall** — For a given class C: precision is the fraction of pixels predicted as C that were actually C; recall is the fraction of true-C pixels the model identified. They trade off against each other and are tuned per-class via confidence thresholds.

**Precision-coverage curve** — Plot of precision vs. coverage (recall) as a confidence threshold sweeps from 0 to 1. Used to pick a threshold meeting a target precision while keeping recall as high as possible.

**QGIS** — Free, open-source GIS software. Used for label creation, exploring imagery, and reviewing model output polygons.

**Rasterize** — Convert vector polygons to a raster (pixel grid), where each pixel takes the class value of the polygon containing its center.

**Red-edge (RE)** — Spectral band at ~717 nm — the transition between red absorption and NIR reflection in vegetation. Particularly sensitive to chlorophyll content.

**Semantic segmentation** — Task of classifying every pixel of an image. Contrast with image classification (one label per image) or object detection (bounding boxes around objects).

***Sesarma reticulatum*** — The purple marsh crab, native to East Coast US salt marshes. Their burrowing activity in *Spartina* bank vegetation is a primary driver of marsh die-off.

**Softmax** — Function applied to raw model outputs (logits) to produce a probability distribution. The 6 output values per pixel are non-negative and sum to 1.

***Spartina alterniflora*** — Smooth cordgrass, the dominant vegetation of New England salt marshes. Healthy *Spartina* is what *Sesarma* feeds on and undermines.

**Superpixel / SLIC** — An appearance-homogeneous clump of adjacent pixels. SLIC (Simple Linear Iterative Clustering) seeds cluster centers on a regular grid and assigns each pixel to the nearest center in a combined colour-plus-position distance, producing compact, roughly equal-size regions that respect real edges. Used as the labeling containers for the abstain bucket (section 8).

**TPI (Topographic Position Index)** — Per-pixel measure of elevation relative to a local neighborhood mean. Positive = local high (ridge/bump), negative = local low (depression). Multi-scale.

**TRI (Terrain Ruggedness Index)** — Per-pixel measure of surface roughness: mean absolute elevation difference from a center pixel to its 8 neighbors. From Riley et al. 1999.
