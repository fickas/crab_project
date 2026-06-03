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
    class_col=Config.CLASS_COLUMN,
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
