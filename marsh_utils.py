# Standard scientific Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import datetime

# Geospatial
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box, mapping, Polygon
from shapely import wkt

# Image processing
import cv2
from PIL import Image

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Segmentation library
# May need to install: !pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp

# Augmentation
# May need to install: !pip install albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Metrics
from sklearn.metrics import confusion_matrix, classification_report

import os, json, math
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT

import geopandas as gpd
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

from scipy import ndimage
import segmentation_models_pytorch as smp

from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from contextlib import ExitStack

from collections import Counter

from torch.utils.data import Dataset


import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

#==============================================================

def recommended_batch_size():
    if not torch.cuda.is_available():
        return 1
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}  ({vram_gb:.1f} GB VRAM)")

    # Conservative batch sizes for U-Net + EfficientNet-B3 at 512x512x3, AMP on
    if vram_gb >= 80:
        return 32
    if vram_gb >= 40:
        return 16
    if vram_gb >= 22:
        return 8
    if vram_gb >= 15:
        return 4
    return 2

def compute_normalized_difference_raster(
    source_path,
    output_path,
    band_a_idx,
    band_b_idx,
    skip_if_exists=True,
    eps=1e-10,
):
    """
    Compute (band_a - band_b) / (band_a + band_b) from two bands of a
    multi-band raster, write to output_path as single-band float32 GeoTIFF.

    For NDVI: band_a = NIR, band_b = R
    For NDRE: band_a = NIR, band_b = RE

    Block-windowed reads and writes keep memory bounded on large rasters.
    Pixels where (a + b) ~ 0 are written as NaN. Source-nodata pixels are
    propagated as NaN in the output.
    """
    if skip_if_exists and os.path.exists(output_path):
        print(f"  exists, skipping: {output_path}")
        return

    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype='float32',
            nodata=np.nan,
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            for _, window in src.block_windows(1):
                bands = src.read(
                    [band_a_idx, band_b_idx], window=window
                ).astype(np.float32)
                a, b = bands[0], bands[1]

                denom = a + b
                with np.errstate(divide='ignore', invalid='ignore'):
                    index = (a - b) / denom
                index = np.where(np.abs(denom) < eps, np.nan, index).astype(np.float32)

                # Propagate source nodata if defined
                if src.nodata is not None:
                    nodata_mask = (
                        (bands[0] == src.nodata) | (bands[1] == src.nodata)
                    )
                    index = np.where(nodata_mask, np.nan, index)

                dst.write(index, 1, window=window)

    print(f"  wrote: {output_path}")


def ensure_indices(
    paths,
    ms_key='pansharp_ms',
    red_band=3, re_band=4, nir_band=5,
    ndvi_key='ndvi', ndre_key='ndre',
):
    """
    Ensure NDVI and NDRE rasters exist alongside the MS source.

    For each of NDVI and NDRE, checks paths[<key>] / disk for the file.
    If missing, computes it from paths[ms_key] using the specified band
    indices and writes it to disk next to the source. Updates the paths
    dict in place with the resolved output paths.

    Defaults assume RedEdge-P band order (B=1, G=2, R=3, RE=4, NIR=5).
    Override if your source uses a different layout.
    """
    source_path = paths[ms_key]
    src_dir = os.path.dirname(source_path)

    ndvi_path = paths.get(ndvi_key) or os.path.join(src_dir, 'ndvi.tif')
    ndre_path = paths.get(ndre_key) or os.path.join(src_dir, 'ndre.tif')

    print("NDVI:")
    compute_normalized_difference_raster(source_path, ndvi_path, nir_band, red_band)

    print("NDRE:")
    compute_normalized_difference_raster(source_path, ndre_path, nir_band, re_band)

def inspect_raster(path):
    """Print basic info about a raster file."""
    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        return None

    with rasterio.open(path) as src:
        print(f"\n=== {path} ===")
        print(f"  Bounds: {src.bounds}")
        print(f"  Size: {src.width} × {src.height} pixels")
        print(f"  Resolution: {src.res[0]:.4f} × {src.res[1]:.4f}")
        print(f"  CRS: {src.crs}")
        print(f"  Bands: {src.count}")
        print(f"  Dtype: {src.dtypes}")
        print(f"  Nodata: {src.nodata}")
        print(f"  Ground area: "
              f"{(src.bounds.right - src.bounds.left):.1f}m × "
              f"{(src.bounds.top - src.bounds.bottom):.1f}m")
    return src.bounds
    paths[ndvi_key] = ndvi_path
    paths[ndre_key] = ndre_path
    return paths

def visualize_orthomosaic_sample(path, window_size_m=10):
    """Show a random window from the orthomosaic."""
    with rasterio.open(path) as src:
        # Get a center window
        cx = (src.bounds.left + src.bounds.right) / 2
        cy = (src.bounds.bottom + src.bounds.top) / 2

        # Calculate window in pixels
        pixels_per_meter = 1 / src.res[0]
        window_pixels = int(window_size_m * pixels_per_meter)

        # Convert coords to pixel offsets
        py, px = src.index(cx, cy)
        window = ((py - window_pixels//2, py + window_pixels//2),
                  (px - window_pixels//2, px + window_pixels//2))

        img = src.read(window=window)

        # Display
        if img.shape[0] == 3:
            img = np.moveaxis(img, 0, -1)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        else:
            img = img[0]

        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.title(f"{window_size_m}m × {window_size_m}m sample @ ({cx:.1f}, {cy:.1f})")
        plt.axis('off')
        plt.show()

def get_block_id(x_world, y_world, block_size_m):
    """Map world coordinates (in CRS units, assumed meters) to integer block IDs."""
    return (int(x_world // block_size_m), int(y_world // block_size_m))


def assign_blocks_to_splits(block_ids, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Deterministically assign each unique block to train/val/test.
    block_ids: iterable of (block_x, block_y) tuples.
    Returns dict mapping (block_x, block_y) -> 'train' / 'val' / 'test'.
    """
    unique = sorted(set(block_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    n = len(unique)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    assignment = {}
    for i, block in enumerate(unique):
        if i < n_train:
            assignment[block] = 'train'
        elif i < n_train + n_val:
            assignment[block] = 'val'
        else:
            assignment[block] = 'test'
    return assignment

def _grids_match(src1, src2):
    """True iff two rasters share CRS, transform, and dimensions."""
    return (
        src1.crs == src2.crs
        and src1.transform == src2.transform
        and src1.width == src2.width
        and src1.height == src2.height
    )


def build_patches_with_splits_multi(
    paths,
    band_spec,
    polygons_gdf,
    patch_size=256,
    overlap=0.5,
    block_size_m=100,
    class_col='Class',
    ignore_value=255,
    priority=None,
    train_frac=0.7,
    val_frac=0.15,
    seed=42,
    require_labels=True,
    resampling=Resampling.bilinear,
):
    """
    Multi-raster patch extractor with spatial split assignment.

    Args:
        paths: dict mapping logical names to file paths.
        band_spec: list of (path_key, band_index) tuples, in stack order.
            Example: [("pan_orthomosaic", 1),
                      ("pansharp_ms", 5),
                      ("pansharp_ms", 4)]    # = pan, NIR, RE
        polygons_gdf: GeoDataFrame of labeled polygons; reprojected if needed.
        patch_size, overlap, block_size_m: as in single-raster version.
        class_col, ignore_value, priority: as in single-raster version.
        train_frac, val_frac, seed: as in single-raster version.
        require_labels: skip patches with zero labeled pixels.
        resampling: rasterio Resampling enum used by WarpedVRT when a
            raster's grid doesn't match the reference raster's grid.
            Reference grid = the first raster in band_spec.

    Yields:
        dict per patch with keys:
            image:            (C, H, W) ndarray, C = len(band_spec)
            mask:             (H, W) uint8 ndarray
            window:           rasterio Window in reference-raster coords
            transform:        affine transform for this patch
            labeled_fraction: float in [0, 1]
            split:            'train' / 'val' / 'test'
            block_id:         (block_x, block_y) tuple
    """
    if not band_spec:
        raise ValueError("band_spec must contain at least one entry")

    stride = max(1, int(patch_size * (1 - overlap)))

    # Unique raster paths in the order they first appear in band_spec
    raster_paths_ordered = []
    for key, _ in band_spec:
        p = paths[key]
        if p not in raster_paths_ordered:
            raster_paths_ordered.append(p)

    # Resolve band_spec keys to actual paths for downstream code
    band_spec_resolved = [(paths[key], idx) for key, idx in band_spec]

    with ExitStack() as stack:
        # First raster defines the reference grid
        ref_path = raster_paths_ordered[0]
        ref_src = stack.enter_context(rasterio.open(ref_path))

        sources = {ref_path: ref_src}

        # Open remaining rasters, wrap in WarpedVRT if their grid differs
        for p in raster_paths_ordered[1:]:
            src = stack.enter_context(rasterio.open(p))
            if _grids_match(src, ref_src):
                sources[p] = src
            else:
                sources[p] = stack.enter_context(WarpedVRT(
                    src,
                    crs=ref_src.crs,
                    transform=ref_src.transform,
                    width=ref_src.width,
                    height=ref_src.height,
                    resampling=resampling,
                ))

        # Project polygons to the reference CRS
        if polygons_gdf.crs != ref_src.crs:
            polygons_gdf = polygons_gdf.to_crs(ref_src.crs)

        # Priority ordering: higher-priority classes rasterized last (win overlaps)
        if priority is not None:
            rank = {c: i for i, c in enumerate(priority)}
            gdf = polygons_gdf.copy()
            gdf['_rank'] = gdf[class_col].map(lambda c: rank.get(c, len(priority)))
            gdf = gdf.sort_values('_rank', ascending=False)
        else:
            gdf = polygons_gdf

        shapes = list(zip(gdf.geometry, gdf[class_col].astype(np.uint8)))

        # Group requested bands by raster path so we read each raster once per patch
        bands_by_path = {}
        for path, band_idx in band_spec_resolved:
            bands_by_path.setdefault(path, []).append(band_idx)

        h, w = ref_src.height, ref_src.width

        # First pass: enumerate patch locations and their blocks
        patch_locations = []
        patch_blocks = []
        for row in range(0, h - patch_size + 1, stride):
            for col in range(0, w - patch_size + 1, stride):
                center_row = row + patch_size // 2
                center_col = col + patch_size // 2
                x_world, y_world = ref_src.xy(center_row, center_col)
                block = get_block_id(x_world, y_world, block_size_m)
                patch_locations.append((row, col))
                patch_blocks.append(block)

        block_to_split = assign_blocks_to_splits(
            patch_blocks, train_frac, val_frac, seed
        )

        # Second pass: read aligned windows from each raster, stack, rasterize mask
        for (row, col), block in zip(patch_locations, patch_blocks):
            window = Window(col, row, patch_size, patch_size)
            window_transform = ref_src.window_transform(window)

            # Read all needed bands from each raster in one read per raster
            arrays_by_key = {}
            for path, band_indices in bands_by_path.items():
                src = sources[path]
                arr = src.read(band_indices, window=window)  # (n_bands, H, W)
                for i, b in enumerate(band_indices):
                    arrays_by_key[(path, b)] = arr[i]

            # Stack in band_spec order
            image = np.stack(
                [arrays_by_key[(path, b)] for path, b in band_spec_resolved],
                axis=0,
            )

            # Rasterize mask
            if shapes:
                mask = rasterize(
                    shapes,
                    out_shape=(patch_size, patch_size),
                    transform=window_transform,
                    fill=ignore_value,
                    dtype=np.uint8,
                )
            else:
                mask = np.full((patch_size, patch_size), ignore_value, dtype=np.uint8)

            labeled_fraction = float((mask != ignore_value).mean())

            if require_labels and labeled_fraction == 0:
                continue

            yield {
                'image': image,
                'mask': mask,
                'window': window,
                'transform': window_transform,
                'labeled_fraction': labeled_fraction,
                'split': block_to_split[block],
                'block_id': block,
            }

class MarshSegmentationDataset(Dataset):
    """
    Holds a list of patch dicts and serves (image_tensor, mask_tensor) pairs.
    Applies an albumentations pipeline if provided.
    """
    def __init__(self, patches, augmentation=None):
        self.patches = patches
        self.augmentation = augmentation

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p = self.patches[idx]
        # rasterio gives (C, H, W); albumentations wants (H, W, C)
        image = p['image'].transpose(1, 2, 0).astype(np.float32)
        mask = p['mask'].astype(np.int64)

        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # back to (C, H, W) tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor

'''
Augmentation
D4 symmetry plus color jitter for train, identity for val/test.
 Albumentations handles image-and-mask jointly and uses nearest-neighbor for the mask by default, which preserves ignore_value=255 and class IDs through rotations.
'''
def get_augmentations(split, mean, std):
    if split == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ])
    return A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
    ])

def summarize_patches(patches, class_names=None, ignore_value=255):
    """
    Print patch counts and per-class pixel coverage per split.
    class_names: dict {class_int: human_name}, or None.
    """
    splits = Counter(p['split'] for p in patches)
    print(f"Total patches: {len(patches)}")
    print(f"Splits: {dict(splits)}\n")

    by_split = {'train': [], 'val': [], 'test': []}
    for p in patches:
        by_split[p['split']].append(p)

    all_classes = sorted({int(c) for p in patches
                          for c in np.unique(p['mask']) if c != ignore_value})

    for split_name, split_patches in by_split.items():
        if not split_patches:
            continue
        total_pixels = sum(p['mask'].size for p in split_patches)
        print(f"--- {split_name}: {len(split_patches)} patches, {total_pixels:,} pixels")
        for c in all_classes:
            n = sum(int((p['mask'] == c).sum()) for p in split_patches)
            label = class_names.get(c, '') if class_names else ''
            print(f"  class {c:3d} {label:20s} {n:>12,} ({100*n/total_pixels:5.2f}%)")
        n_ignore = sum(int((p['mask'] == ignore_value).sum()) for p in split_patches)
        print(f"  ignore (255)           {n_ignore:>12,} ({100*n_ignore/total_pixels:5.2f}%)")
        print()

def compute_channel_stats(train_patches, stats_path, force=False):
    """
    Compute per-channel mean and std for image normalization.

    Streams sum / sum-of-squares in float64 with NaN-aware skipping (important
    for index bands like NDVI, where pixels with tiny denominators are NaN).
    Caches the result to `stats_path` as JSON; reloads on subsequent calls
    unless `force=True`.

    `train_patches` can be any iterable that yields, per item, one of:
      - a dict with key 'image' (or 'img', 'x'),
      - a (image, mask) tuple/list,
      - an object with an .image attribute,
      - a bare (C, H, W) numpy ndarray.
    Torch tensors are auto-converted to numpy.

    Returns (channel_means, channel_stds) as float32 arrays of length C.
    """
    if not force and os.path.exists(stats_path):
        with open(stats_path) as f:
            d = json.load(f)
        print(f"Loaded cached channel stats from {stats_path}")
        for c, (m, s) in enumerate(zip(d['means'], d['stds'])):
            print(f"  channel {c}: mean={m:.4f}, std={s:.4f}")
        return (np.asarray(d['means'], dtype=np.float32),
                np.asarray(d['stds'],  dtype=np.float32))

    sums = sumsq = counts = None
    n_seen = 0
    try:
        total = len(train_patches)
    except TypeError:
        total = None

    for item in train_patches:
        img = None
        if isinstance(item, dict):
            for k in ('image', 'img', 'x'):
                if k in item:
                    img = item[k]; break
        elif isinstance(item, (tuple, list)) and len(item) > 0:
            img = item[0]
        elif hasattr(item, 'image'):
            img = item.image
        elif isinstance(item, np.ndarray):
            img = item
        if img is None:
            raise TypeError(f"Don't know how to extract image from {type(item).__name__}")

        if hasattr(img, 'detach'):                  # torch.Tensor → numpy
            img = img.detach().cpu().numpy()
        img = np.asarray(img, dtype=np.float64)
        if img.ndim == 2:
            img = img[np.newaxis]
        elif img.ndim != 3:
            raise ValueError(f"Expected (C,H,W) or (H,W); got {img.shape}")

        if sums is None:
            C = img.shape[0]
            sums   = np.zeros(C, dtype=np.float64)
            sumsq  = np.zeros(C, dtype=np.float64)
            counts = np.zeros(C, dtype=np.int64)

        for c in range(img.shape[0]):
            band = img[c].ravel()
            valid = np.isfinite(band)
            if not valid.any():
                continue
            v = band[valid]
            sums[c]   += v.sum()
            sumsq[c]  += (v * v).sum()
            counts[c] += v.size

        n_seen += 1
        if total and n_seen % max(1, total // 10) == 0:
            print(f"  {n_seen}/{total} patches...")

    if sums is None:
        raise ValueError("compute_channel_stats: no patches provided")

    means = sums / np.maximum(counts, 1)
    var   = sumsq / np.maximum(counts, 1) - means ** 2
    stds  = np.sqrt(np.maximum(var, 0.0))

    os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump({
            'means': means.tolist(),
            'stds':  stds.tolist(),
            'n_pixels_per_channel': counts.tolist(),
            'n_patches': n_seen,
        }, f, indent=2)
    print(f"Saved channel stats → {stats_path}")
    for c, (m, s) in enumerate(zip(means, stds)):
        print(f"  channel {c}: mean={m:.4f}, std={s:.4f}, n={counts[c]:,}")

    return means.astype(np.float32), stds.astype(np.float32)

def compute_channel_stats(patches, output_path, skip_if_exists=True):
    """
    Compute per-channel mean and std over a list of patches, streaming
    through them one at a time so the full set never has to fit in memory.
    NaN values (from NDVI/NDRE no-data) are ignored, with valid-pixel
    counts tracked per channel.

    If output_path already exists and skip_if_exists=True, loads and
    returns the cached values without recomputing.

    Returns (mean_array, std_array), each shape (C,).
    """
    if skip_if_exists and os.path.exists(output_path):
        print(f"  loading cached stats from {output_path}")
        with open(output_path) as f:
            stats = json.load(f)
        return np.array(stats['mean']), np.array(stats['std'])

    if not patches:
        raise ValueError("No patches provided")

    C = patches[0]['image'].shape[0]

    # Float64 accumulators — guard against cumulative FP error at scale
    sum_c    = np.zeros(C, dtype=np.float64)
    sum_sq_c = np.zeros(C, dtype=np.float64)
    n_valid  = np.zeros(C, dtype=np.int64)

    for p in patches:
        img = p['image'].astype(np.float64)               # (C, H, W)
        sum_c    += np.nansum(img,        axis=(1, 2))
        sum_sq_c += np.nansum(img ** 2,   axis=(1, 2))
        n_valid  += np.sum(~np.isnan(img), axis=(1, 2))

    if np.any(n_valid == 0):
        bad = np.where(n_valid == 0)[0].tolist()
        raise ValueError(f"No valid pixels in channel(s) {bad} — all NaN?")

    means = sum_c / n_valid
    # Variance = E[X^2] - E[X]^2; clamp tiny negatives from FP error
    vars_ = np.maximum(sum_sq_c / n_valid - means ** 2, 0.0)
    stds  = np.sqrt(vars_)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'mean':           means.tolist(),
            'std':            stds.tolist(),
            'n_patches':      len(patches),
            'n_valid_pixels': n_valid.tolist(),
        }, f, indent=2)
    print(f"  saved stats to {output_path}")

    return means, stds

#  TRAINING ================================

#Custom combined loss (CE + Dice, both ignore-aware).
class CombinedLoss(nn.Module):
    """
    CE + Dice for multi-class semantic segmentation.
    Pixels with target == ignore_index are excluded from both terms.
    """
    def __init__(self, num_classes, ignore_index=255,
                 ce_weight=1.0, dice_weight=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    def _dice(self, logits, targets):
        valid = (targets != self.ignore_index)               # (B, H, W)
        probs = F.softmax(logits, dim=1)                     # (B, C, H, W)

        # Replace ignore targets with 0 before one-hot, then mask
        safe = targets.clone()
        safe[~valid] = 0
        onehot = F.one_hot(safe, num_classes=self.num_classes) \
                  .permute(0, 3, 1, 2).float()               # (B, C, H, W)

        mask = valid.unsqueeze(1).float()                    # (B, 1, H, W)
        probs = probs * mask
        onehot = onehot * mask

        inter = (probs * onehot).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice = (2.0 * inter + 1e-7) / (denom + 1e-7)
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) \
             + self.dice_weight * self._dice(logits, targets)

#Per-class IoU metric (also ignore-aware)
class IoUMetric:
    def __init__(self, num_classes, ignore_index=255, classes_of_interest=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.classes_of_interest = (
            list(range(num_classes)) if classes_of_interest is None
            else list(classes_of_interest)
        )
        self.reset()

    def reset(self):
        self.inter = torch.zeros(self.num_classes, dtype=torch.float64)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, targets):
        preds = logits.argmax(dim=1)
        valid = (targets != self.ignore_index)
        for c in range(self.num_classes):
            p = (preds == c) & valid
            t = (targets == c) & valid
            self.inter[c] += (p & t).sum().double().cpu()
            self.union[c] += (p | t).sum().double().cpu()

    def compute(self):
        iou_per_class = self.inter / (self.union + 1e-7)
        of_interest = [c for c in self.classes_of_interest if self.union[c] > 0]
        mean_iou = iou_per_class[of_interest].mean().item() if of_interest else 0.0
        return iou_per_class, mean_iou


#Training and validation step functions (this is where AMP lands)
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, metric, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    metric.reset()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        metric.update(logits, masks)
        n_batches  += 1
    iou_per_class, mean_iou = metric.compute()
    return total_loss / n_batches, iou_per_class, mean_iou

#the training driver
def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs, num_classes, ignore_index=255,
          ckpt_path=None,
          device='cuda', class_names=None):
    scaler = GradScaler()
    metric = IoUMetric(num_classes=6, classes_of_interest=[3, 4, 5])
    best_iou = 0.0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, iou_per_class, mean_iou = validate(model, val_loader, criterion, metric, device)
        if scheduler is not None:
            scheduler.step()

        per_class_str = ', '.join(
            f"{(class_names or {}).get(c, c)}={iou_per_class[c].item():.3f}"
            for c in range(num_classes)
        )
        print(f"Epoch {epoch+1:3d}/{num_epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_mIoU={mean_iou:.4f}  |  {per_class_str}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'iou_per_class': iou_per_class.tolist(),
            }, ckpt_path)
            print(f"  → saved checkpoint  (val_mIoU={mean_iou:.4f})")

    return best_iou

import torch
import torch.nn.functional as F


class PrecisionCoverageMetric:
    """
    Per-class precision, recall, and coverage at a list of confidence thresholds.

    For each class c and threshold t:
      - predicted[c, t]: pixels where argmax == c AND max-softmax >= t
      - correct[c, t]  : predicted[c, t] AND target == c
      - precision[c, t] = correct[c, t] / predicted[c, t]
      - recall[c, t]    = correct[c, t] / total_actual[c]
      - coverage[c, t]  = predicted[c, t] / total_valid_pixels

    Ignores pixels where target == ignore_index in all denominators.
    """
    def __init__(self, num_classes, thresholds=None, ignore_index=255):
        self.num_classes = num_classes
        self.thresholds = (
            [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            if thresholds is None else list(thresholds)
        )
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        T = len(self.thresholds)
        self.predicted   = torch.zeros(self.num_classes, T, dtype=torch.float64)
        self.correct     = torch.zeros(self.num_classes, T, dtype=torch.float64)
        self.actual      = torch.zeros(self.num_classes,    dtype=torch.float64)
        self.total_valid = 0

    @torch.no_grad()
    def update(self, logits, targets):
        # Probabilities and confident predictions
        probs = F.softmax(logits.float(), dim=1)         # (B, C, H, W)
        confidence, preds = probs.max(dim=1)             # (B, H, W) each
        valid = (targets != self.ignore_index)           # (B, H, W)

        # Filter to valid pixels to avoid wasted work and ignore-pollution
        confidence_v = confidence[valid]                 # (N,)
        preds_v      = preds[valid]                      # (N,)
        targets_v    = targets[valid]                    # (N,)
        self.total_valid += int(valid.sum().item())

        # Per-class ground-truth pixel counts
        for c in range(self.num_classes):
            self.actual[c] += (targets_v == c).sum().double().cpu()

        # Vectorize across thresholds: (N, T) boolean of "confidence >= t"
        thresholds_t = torch.tensor(
            self.thresholds, device=confidence_v.device, dtype=confidence_v.dtype
        )
        high_conf = confidence_v.unsqueeze(1) >= thresholds_t.unsqueeze(0)  # (N, T)

        # For each class, count predicted and correct at every threshold
        for c in range(self.num_classes):
            pred_is_c = (preds_v == c).unsqueeze(1)              # (N, 1)
            pred_c_high = pred_is_c & high_conf                  # (N, T)
            self.predicted[c] += pred_c_high.sum(dim=0).double().cpu()

            correct_mask = (targets_v == c).unsqueeze(1)         # (N, 1)
            self.correct[c]   += (pred_c_high & correct_mask).sum(dim=0).double().cpu()

    def compute(self):
        # Precision: correct / predicted (per class, per threshold)
        precision = self.correct / (self.predicted + 1e-7)
        # Recall:    correct / actual_total_class (per class, per threshold)
        recall    = self.correct / (self.actual.unsqueeze(1) + 1e-7)
        # Coverage: predicted / total_valid (per class, per threshold)
        coverage  = self.predicted / max(self.total_valid, 1)

        return {
            'thresholds':  list(self.thresholds),
            'precision':   precision,   # (C, T)
            'recall':      recall,      # (C, T)
            'coverage':    coverage,    # (C, T)
            'n_predicted': self.predicted,
            'n_correct':   self.correct,
            'n_actual':    self.actual,
            'total_valid': self.total_valid,
        }

@torch.no_grad()
def evaluate_precision_coverage(model, loader, num_classes,
                                 thresholds=None, ignore_index=255, device='cuda'):
    metric = PrecisionCoverageMetric(
        num_classes=num_classes,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
    model.eval()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        logits = model(images)
        metric.update(logits, masks)
    return metric.compute()

import matplotlib.pyplot as plt


def plot_precision_coverage(results, class_names=None, classes_of_interest=None):
    """Two-panel plot: precision-vs-threshold and coverage-vs-threshold."""
    thresholds = results['thresholds']
    precision  = results['precision'].numpy()
    coverage   = results['coverage'].numpy()

    num_classes = precision.shape[0]
    classes_of_interest = (
        list(range(num_classes)) if classes_of_interest is None else classes_of_interest
    )
    names = class_names or {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for c in classes_of_interest:
        label = names.get(c, f'class {c}')
        axes[0].plot(thresholds, precision[c], marker='o', label=label)
        axes[1].plot(thresholds, coverage[c],  marker='o', label=label)

    axes[0].set_xlabel('Confidence threshold')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Per-class precision vs confidence threshold')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel('Confidence threshold')
    axes[1].set_ylabel('Coverage (fraction of all valid pixels)')
    axes[1].set_title('Per-class coverage vs confidence threshold')
    axes[1].set_yscale('log')   # coverage spans orders of magnitude
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_precision_vs_coverage(results, class_names=None, classes_of_interest=None):
    """Tradeoff curve: precision (y) vs coverage (x), with thresholds annotated."""
    thresholds = results['thresholds']
    precision  = results['precision'].numpy()
    coverage   = results['coverage'].numpy()

    num_classes = precision.shape[0]
    classes_of_interest = (
        list(range(num_classes)) if classes_of_interest is None else classes_of_interest
    )
    names = class_names or {}

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in classes_of_interest:
        label = names.get(c, f'class {c}')
        ax.plot(coverage[c], precision[c], marker='o', label=label)
        # Annotate threshold values at every other point to reduce clutter
        for i in range(0, len(thresholds), 2):
            ax.annotate(f'{thresholds[i]:.2f}',
                        (coverage[c][i], precision[c][i]),
                        fontsize=7, alpha=0.6,
                        xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('Coverage (fraction of valid pixels predicted as class)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-coverage tradeoff (confidence thresholds annotated)')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    plt.tight_layout()
    return fig

def print_precision_coverage_table(results, class_names=None, classes_of_interest=None):
    thresholds = results['thresholds']
    precision  = results['precision'].numpy()
    recall     = results['recall'].numpy()
    coverage   = results['coverage'].numpy()

    num_classes = precision.shape[0]
    classes_of_interest = (
        list(range(num_classes)) if classes_of_interest is None else classes_of_interest
    )
    names = class_names or {}

    for c in classes_of_interest:
        label = names.get(c, f'class {c}')
        print(f"\nClass {c} ({label}):")
        print(f"  {'threshold':>9}  {'precision':>10}  {'recall':>10}  {'coverage':>10}")
        for i, t in enumerate(thresholds):
            print(f"  {t:>9.2f}  {precision[c][i]:>10.4f}  "
                  f"{recall[c][i]:>10.4f}  {coverage[c][i]:>10.4f}")

def pick_thresholds(pc_results, classes_of_interest, target_precision=0.9):
    """
    For each class, pick the lowest confidence threshold that achieves
    at least target_precision on the validation set. Maximizes coverage
    subject to a precision floor.

    Returns dict mapping class -> chosen threshold.
    """
    thresholds = pc_results['thresholds']
    precision  = pc_results['precision'].numpy()      # (C, T)
    coverage   = pc_results['coverage'].numpy()       # (C, T)

    chosen = {}
    for cls in classes_of_interest:
        valid = np.where(precision[cls] >= target_precision)[0]
        if len(valid) > 0:
            # Lowest threshold meeting the bar = highest coverage
            i = valid[0]
            chosen[cls] = float(thresholds[i])
            print(f"  class {cls}: threshold={thresholds[i]:.2f}  "
                  f"precision={precision[cls][i]:.3f}  "
                  f"coverage={coverage[cls][i]:.3f}")
        else:
            # Target unattainable for this class — pick the highest threshold
            # available and warn
            i = -1
            chosen[cls] = float(thresholds[i])
            print(f"  class {cls}: WARNING no threshold reaches "
                  f"precision={target_precision}, using {thresholds[i]:.2f} "
                  f"(precision={precision[cls][i]:.3f})")
    return chosen

def config_to_dict(config_cls):
    return {
        k: v for k, v in vars(config_cls).items() if not k.startswith('_')
    }

def save_training_artifacts(output_dir, model, channel_means, channel_stds, training_summary,cfg):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

    bundle = {
        'config':         config_to_dict(cfg),
        'normalization': {
            'mean': [float(x) for x in channel_means],
            'std':  [float(x) for x in channel_stds],
        },
        'training_summary': training_summary,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(bundle, f, indent=2)
    print(f"Saved artifacts to {output_dir}")

#Analyze channel importance
@torch.no_grad()
def channel_zero_ablation(model, loader, num_classes, classes_of_interest,
                          channel_names=None, ignore_index=255, device='cuda'):
    """
    Inference-time channel ablation: zero out each channel one at a time
    and measure mean IoU vs baseline.
    Returns dict with baseline, per-channel mIoU, and per-channel drop.
    """
    model.eval()

    def run_loader_with_mask(zero_channel=None):
        m = IoUMetric(num_classes, ignore_index, classes_of_interest)
        for images, masks in loader:
            images = images.to(device, non_blocking=True).clone()
            masks  = masks.to(device,  non_blocking=True)
            if zero_channel is not None:
                images[:, zero_channel] = 0
            logits = model(images)
            m.update(logits, masks)
        return m.compute()

    # Baseline (no ablation)
    base_iou_per_class, base_mean_iou = run_loader_with_mask(None)

    # Discover channel count from first batch
    sample_imgs, _ = next(iter(loader))
    n_channels = sample_imgs.shape[1]
    names = channel_names or [f'ch{c}' for c in range(n_channels)]

    results = {
        'baseline_mean_iou':    base_mean_iou,
        'baseline_iou_per_class': base_iou_per_class.tolist(),
        'per_channel': {},
    }

    for ch in range(n_channels):
        iou_per_class, mean_iou = run_loader_with_mask(ch)
        results['per_channel'][names[ch]] = {
            'mean_iou':        mean_iou,
            'iou_per_class':   iou_per_class.tolist(),
            'drop_from_base':  base_mean_iou - mean_iou,
        }

    return results


@torch.no_grad()
def channel_permutation_importance(model, loader, num_classes, classes_of_interest,
                                    channel_names=None, ignore_index=255,
                                    device='cuda', n_repeats=3):
    """
    Permute each channel's pixel values spatially within each sample,
    measure mean IoU drop. Repeat n_repeats times for stability.
    """
    model.eval()

    def baseline():
        m = IoUMetric(num_classes, ignore_index, classes_of_interest)
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            logits = model(images)
            m.update(logits, masks)
        return m.compute()

    base_iou_per_class, base_mean_iou = baseline()

    sample_imgs, _ = next(iter(loader))
    n_channels = sample_imgs.shape[1]
    names = channel_names or [f'ch{c}' for c in range(n_channels)]

    results = {
        'baseline_mean_iou': base_mean_iou,
        'per_channel': {},
    }

    for ch in range(n_channels):
        drops = []
        for _ in range(n_repeats):
            m = IoUMetric(num_classes, ignore_index, classes_of_interest)
            for images, masks in loader:
                images = images.to(device, non_blocking=True).clone()
                B, C, H, W = images.shape
                # Permute pixels within each sample's channel
                flat = images[:, ch].view(B, -1)
                idx  = torch.stack([torch.randperm(H * W, device=device) for _ in range(B)])
                permuted = flat.gather(1, idx).view(B, H, W)
                images[:, ch] = permuted
                masks = masks.to(device, non_blocking=True)
                logits = model(images)
                m.update(logits, masks)
            _, mi = m.compute()
            drops.append(base_mean_iou - mi)
        results['per_channel'][names[ch]] = {
            'drops':      drops,
            'mean_drop':  sum(drops) / len(drops),
            'std_drop':   (sum((d - sum(drops)/len(drops))**2 for d in drops) / len(drops)) ** 0.5,
        }

    return results

import numpy as np
import torch


def _per_class_iou_on_loader(model, loader, num_classes, ignore_index, device,
                             perturb_channel=None, rng=None):
    """Per-class IoU on a loader. If perturb_channel is given, that channel is
    shuffled across the batch dim before each forward pass."""
    intersection = np.zeros(num_classes, dtype=np.int64)
    union        = np.zeros(num_classes, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                images = batch['image']
                targets = batch.get('mask') or batch.get('target') or batch.get('label')
            else:
                images, targets = batch[0], batch[1]
            images, targets = images.to(device), targets.to(device)

            if perturb_channel is not None:
                B = images.shape[0]
                images = images.clone()
                if B > 1:
                    perm = torch.randperm(B, generator=rng, device=device)
                    images[:, perturb_channel] = images[perm, perturb_channel]
                else:
                    # batch size 1 fallback: shuffle pixels within the single patch
                    band = images[0, perturb_channel].flatten()
                    idx = torch.randperm(band.numel(), generator=rng, device=device)
                    images[0, perturb_channel] = band[idx].reshape_as(images[0, perturb_channel])

            preds = model(images).argmax(dim=1)
            valid = (targets != ignore_index)
            for c in range(num_classes):
                p = (preds == c) & valid
                t = (targets == c) & valid
                intersection[c] += (p & t).sum().item()
                union[c]        += (p | t).sum().item()

    return np.where(union > 0, intersection / np.maximum(union, 1), np.nan)


def channel_permutation_importance_per_class(
    model, val_loader, num_classes,
    ignore_index=255, n_repeats=3, device='cuda', seed=42,
    class_names=None, band_names=None,
):
    """Per-class permutation importance.

    For each (channel, repeat): shuffle that channel across the batch
    dimension (preserves the channel's marginal distribution, breaks its
    correlation with labels and with the other channels), recompute per-class
    IoU on the val set, and record the drop vs. baseline.

    Returns dict with:
        baseline_iou : (num_classes,)            per-class IoU with intact inputs
        drops        : (n_channels, num_classes, n_repeats)
        drops_mean   : (n_channels, num_classes)
        drops_std    : (n_channels, num_classes)
        band_names   : list[str]                 channel labels used in print
        class_names  : list[str]                 class labels used in print
    """
    # baseline
    baseline = _per_class_iou_on_loader(model, val_loader, num_classes,
                                        ignore_index, device)

    # channel count from a sample batch
    first = next(iter(val_loader))
    sample_img = first['image'] if isinstance(first, dict) else first[0]
    n_channels = sample_img.shape[1]

    master = np.random.default_rng(seed)
    drops = np.zeros((n_channels, num_classes, n_repeats), dtype=np.float64)

    for ch in range(n_channels):
        for rep in range(n_repeats):
            gen = torch.Generator(device=device).manual_seed(int(master.integers(2**31)))
            after = _per_class_iou_on_loader(model, val_loader, num_classes,
                                             ignore_index, device,
                                             perturb_channel=ch, rng=gen)
            drops[ch, :, rep] = baseline - after

    drops_mean = drops.mean(axis=2)
    drops_std  = drops.std(axis=2)

    # name lookups
    if class_names is None:
        cnames = [f"class{c}" for c in range(num_classes)]
    elif hasattr(class_names, 'get'):
        cnames = [class_names.get(c, f"class{c}") for c in range(num_classes)]
    else:
        cnames = [class_names[c] if c < len(class_names) else f"class{c}"
                  for c in range(num_classes)]
    if band_names is None:
        bnames = [f"ch{ch}" for ch in range(n_channels)]
    else:
        bnames = list(band_names)
        while len(bnames) < n_channels:
            bnames.append(f"ch{len(bnames)}")

    # ---- print summary ----
    print("Baseline per-class IoU:")
    for c, name in enumerate(cnames):
        v = baseline[c]
        print(f"  {name:20s} {('  nan' if np.isnan(v) else f'{v:.4f}')}")
    print()
    print(f"Per-class permutation drops (mean ± std over {n_repeats} repeats):")
    print(f"  {'channel':>12s}  " + "  ".join(f"{n[:14]:>14s}" for n in cnames))
    for ch in range(n_channels):
        cells = []
        for c in range(num_classes):
            if np.isnan(drops_mean[ch, c]):
                cells.append("           nan")
            else:
                cells.append(f"{drops_mean[ch,c]:+.3f}±{drops_std[ch,c]:.3f}")
        print(f"  {bnames[ch]:>12s}  " + "  ".join(f"{c:>14s}" for c in cells))

    return {
        'baseline_iou': baseline,
        'drops':        drops,
        'drops_mean':   drops_mean,
        'drops_std':    drops_std,
        'band_names':   bnames,
        'class_names':  cnames,
    }
