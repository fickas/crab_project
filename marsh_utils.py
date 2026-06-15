# ── Standard library ──
import os
import json
import math
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from contextlib import ExitStack
from collections import Counter

# ── Scientific Python ──
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics import confusion_matrix, classification_report

# ── Geospatial: rasterio ──
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize, shapes
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT

# ── Geospatial: shapely + geopandas ──
import geopandas as gpd
from shapely.geometry import (LineString, Point, Polygon, MultiPolygon,
                              box, mapping, shape)
from shapely.ops import unary_union
from shapely import wkt

# ── Image processing ──
import cv2
from PIL import Image

# ── Deep learning ──
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import  GradScaler

# ── Segmentation + augmentation ──
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Synthetic-marsh constants (used by generate_marsh_geometry, assign_polygon_labels,
# make_class_index_raster, composite_spectra, and the synth-data main())
CRS = "EPSG:26919"
SEED = 42
# ── Class scheme (single source of truth — don't redefine in notebooks) ──
CLASS_NAMES = {
    0: 'other',
    1: 'healthy_bank',
    2: 'eroding_non_crab',
    3: 'crab_edge',
    4: 'crab_platform',
    5: 'collapsed',
}
# Derived: name → QGIS-style 1-indexed (for shapefile writes / synthetic gen)
CLASSES = {name: idx + 1 for idx, name in CLASS_NAMES.items()}
# Derived: QGIS-style 1-indexed → model 0-indexed
QGIS_TO_MODEL = {q: m for m, q in zip(CLASS_NAMES.keys(), CLASSES.values())}
MODEL_TO_QGIS = {m: q for q, m in QGIS_TO_MODEL.items()}

# Bank classes you want exported as polygons (model-0-indexed)
CRAB_CLASSES = ['crab_edge', 'crab_platform', 'collapsed']
CLASSES_OF_INTEREST = [m for m, name in CLASS_NAMES.items() if name in CRAB_CLASSES]

# Rasterization priority — higher class index wins overlaps
PRIORITY = [5, 4, 3, 2, 1, 0]

IGNORE_INDEX = 255

# Per-class spectral signatures used by the synthetic data generator.
# Order: Blue (475nm), Green (560nm), Red (668nm), RedEdge (717nm), NIR (842nm).
# Values are reflectance in [0, 1], chosen to make NDVI/NDRE differentiable
# between healthy, eroding, and crab-damaged states (see verification table
# we computed earlier).
SPECTRA = {
    'water':            np.array([0.045, 0.055, 0.030, 0.020, 0.010]),
    'marsh_platform':   np.array([0.055, 0.095, 0.050, 0.180, 0.300]),
    'healthy_bank':     np.array([0.050, 0.100, 0.045, 0.200, 0.350]),
    'eroding_non_crab': np.array([0.090, 0.130, 0.125, 0.180, 0.210]),
    'crab_edge':        np.array([0.105, 0.140, 0.160, 0.165, 0.160]),
    'crab_platform':    np.array([0.110, 0.135, 0.150, 0.155, 0.145]),
    'collapsed':        np.array([0.125, 0.150, 0.170, 0.150, 0.105]),
    'tree':             np.array([0.030, 0.075, 0.030, 0.250, 0.450]),
}

# Integer indices used in the class-index raster during rasterization;
# offset by 10 to avoid colliding with anything that uses 0–9.
SPECTRAL_IDX   = {k: 10 + i for i, k in enumerate(SPECTRA.keys())}
IDX_TO_SPECTRA = {v: SPECTRA[k] for k, v in SPECTRAL_IDX.items()}

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

    paths[ndvi_key] = ndvi_path     # ← add these
    paths[ndre_key] = ndre_path     # ←
    return paths

#++++++  MORE BAND DERIVATIONS

"""
Additions to marsh_utils.py — derived raster bands beyond NDVI/NDRE.

Copy whatever you decide to use into marsh_utils.py. Each compute_* function
does the math + raster I/O; each ensure_* function is the cache-and-update-paths
wrapper following the existing ensure_indices pattern.

Conventions:
  - Outputs are float32, with NaN as nodata.
  - DEFLATE + predictor=3 compression for floats (better than LZW for non-int data).
  - Inputs read with NaN-aware logic (existing nodata gets converted to NaN on read).
  - All ensure_* functions are idempotent: skip recompute if output exists,
    and update `paths` dict in place.

Default band order assumes RedEdge-P: B=1, G=2, R=3, RE=4, NIR=5.
"""
import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy import ndimage


# ============================================================================
# Low-level I/O helpers
# ============================================================================
def _read_band_as_float(src, band_idx):
    """Read a band as float32, converting nodata to NaN."""
    arr = src.read(band_idx).astype(np.float32)
    nd = src.nodatavals[band_idx - 1] if src.nodatavals else None
    if nd is not None and not np.isnan(nd):
        arr[arr == nd] = np.nan
    return arr


def _write_derived_raster(out_path, data, ref_src, dtype='float32'):
    """Write a single-band derived raster with the same georeferencing as ref_src."""
    profile = ref_src.profile.copy()
    profile.update(
        count=1, dtype=dtype,
        nodata=float('nan') if dtype.startswith('float') else None,
        compress='DEFLATE',
        predictor=3 if dtype.startswith('float') else 2,
        tiled=True, blockxsize=256, blockysize=256,
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data.astype(dtype), 1)


def _gsd_meters(src):
    """Pixel size in meters from a rasterio source (assumes square pixels in a projected CRS)."""
    return abs(src.transform.a)


# ============================================================================
# Spectral indices (computed from a multispectral source)
# ============================================================================
def compute_savi_raster(src_path, out_path, red_band=3, nir_band=5, L=0.5):
    """SAVI = (NIR - R) / (NIR + R + L) * (1 + L)
    Soil-adjusted vegetation index. Less biased by bare soil than NDVI;
    useful for mixed pixels at damaged banks. L=0.5 is the canonical mid-cover
    value (use 0.25 for dense, 1.0 for sparse cover)."""
    if os.path.exists(out_path):
        print(f"  SAVI already exists at {out_path}, skipping")
        return
    with rasterio.open(src_path) as src:
        red = _read_band_as_float(src, red_band)
        nir = _read_band_as_float(src, nir_band)
        savi = (nir - red) / (nir + red + L) * (1.0 + L)
        _write_derived_raster(out_path, savi, src)
    print(f"  wrote SAVI to {out_path}")


def compute_evi_raster(src_path, out_path, blue_band=1, red_band=3, nir_band=5,
                       G=2.5, C1=6.0, C2=7.5, L=1.0):
    """EVI = G * (NIR - R) / (NIR + C1*R - C2*B + L)
    Enhanced vegetation index. Resists atmospheric and soil effects better
    than NDVI; particularly useful in mixed-substrate marsh edges."""
    if os.path.exists(out_path):
        print(f"  EVI already exists at {out_path}, skipping")
        return
    with rasterio.open(src_path) as src:
        blue = _read_band_as_float(src, blue_band)
        red  = _read_band_as_float(src, red_band)
        nir  = _read_band_as_float(src, nir_band)
        denom = nir + C1 * red - C2 * blue + L
        evi = G * (nir - red) / denom
        # EVI can blow up for very dark pixels; clip extremes
        evi = np.where(np.abs(denom) < 1e-4, np.nan, evi)
        evi = np.clip(evi, -2.0, 2.0)
        _write_derived_raster(out_path, evi, src)
    print(f"  wrote EVI to {out_path}")


def compute_ci_rededge_raster(src_path, out_path, re_band=4, nir_band=5):
    """CIred-edge = NIR / RE - 1
    Chlorophyll Index using red-edge. Not a normalized-difference form, so
    less redundant with NDVI than NDRE. Sensitive to chlorophyll content."""
    if os.path.exists(out_path):
        print(f"  CIred-edge already exists at {out_path}, skipping")
        return
    with rasterio.open(src_path) as src:
        re_ = _read_band_as_float(src, re_band)
        nir = _read_band_as_float(src, nir_band)
        ci = np.where(re_ > 1e-4, nir / re_ - 1.0, np.nan)
        _write_derived_raster(out_path, ci, src)
    print(f"  wrote CIred-edge to {out_path}")


# GNDVI and NDWI are just normalized-difference indices.
# Reuse your existing compute_normalized_difference_raster(src, out, num, den).
# Convenience wrappers:
def compute_gndvi_raster(src_path, out_path, green_band=2, nir_band=5):
    """GNDVI = (NIR - G) / (NIR + G). Chlorophyll index using green instead of red."""
    from marsh_utils import compute_normalized_difference_raster
    if os.path.exists(out_path):
        print(f"  GNDVI already exists at {out_path}, skipping"); return
    compute_normalized_difference_raster(src_path, out_path, nir_band, green_band)


def compute_ndwi_raster(src_path, out_path, green_band=2, nir_band=5):
    """NDWI (McFeeters) = (G - NIR) / (G + NIR). Highlights water and saturated mud."""
    from marsh_utils import compute_normalized_difference_raster
    if os.path.exists(out_path):
        print(f"  NDWI already exists at {out_path}, skipping"); return
    compute_normalized_difference_raster(src_path, out_path, green_band, nir_band)


# ============================================================================
# DEM-derived bands
# ============================================================================
def compute_slope_raster(dem_path, out_path, units='degrees'):
    """Slope from DEM using Horn's 3x3 method.
    units: 'degrees' (0..90) or 'radians' or 'percent'."""
    if os.path.exists(out_path):
        print(f"  Slope already exists at {out_path}, skipping"); return
    with rasterio.open(dem_path) as src:
        dem = _read_band_as_float(src, 1)
        cell = _gsd_meters(src)
        # Horn's method (Sobel-style kernels normalized for slope)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / (8.0 * cell)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / (8.0 * cell)
        # Fill NaNs with edge values for the convolution, mask result afterwards
        nan_mask = np.isnan(dem)
        dem_filled = np.where(nan_mask, np.nanmean(dem), dem)
        dx = ndimage.convolve(dem_filled, kx, mode='nearest')
        dy = ndimage.convolve(dem_filled, ky, mode='nearest')
        slope_rad = np.arctan(np.sqrt(dx * dx + dy * dy))
        if units == 'degrees':
            slope = np.degrees(slope_rad)
        elif units == 'radians':
            slope = slope_rad
        elif units == 'percent':
            slope = np.tan(slope_rad) * 100.0
        else:
            raise ValueError(f"units must be degrees|radians|percent, got {units}")
        slope[nan_mask] = np.nan
        _write_derived_raster(out_path, slope, src)
    print(f"  wrote slope ({units}) to {out_path}")


def compute_tpi_raster(dem_path, out_path, neighborhood_m=2.0):
    """Topographic Position Index: dem - mean(dem in neighborhood).
    Positive = relative ridge/hummock; negative = relative depression.
    Picks up 'this bank is slumped relative to surrounding marsh' directly."""
    if os.path.exists(out_path):
        print(f"  TPI already exists at {out_path}, skipping"); return
    with rasterio.open(dem_path) as src:
        dem = _read_band_as_float(src, 1)
        cell = _gsd_meters(src)
        window_px = max(3, int(round(neighborhood_m / cell)))
        if window_px % 2 == 0:
            window_px += 1   # odd window centers cleanly
        # Mean of valid pixels in window (ignoring NaN)
        # Use uniform_filter with NaN handling: filter ones and values separately
        valid = (~np.isnan(dem)).astype(np.float32)
        dem_zeroed = np.where(np.isnan(dem), 0.0, dem)
        sum_f = ndimage.uniform_filter(dem_zeroed, size=window_px, mode='nearest')
        cnt_f = ndimage.uniform_filter(valid,      size=window_px, mode='nearest')
        mean_dem = np.where(cnt_f > 0, sum_f / cnt_f, np.nan)
        tpi = dem - mean_dem
        _write_derived_raster(out_path, tpi, src)
    print(f"  wrote TPI ({neighborhood_m}m window) to {out_path}")


def compute_curvature_raster(dem_path, out_path):
    """Curvature via Laplacian of elevation.
    Positive = concave (depression-like), negative = convex (ridge-like).
    Simpler than full Zevenbergen-Thorne profile/plan curvature."""
    if os.path.exists(out_path):
        print(f"  Curvature already exists at {out_path}, skipping"); return
    with rasterio.open(dem_path) as src:
        dem = _read_band_as_float(src, 1)
        cell = _gsd_meters(src)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / (cell * cell)
        nan_mask = np.isnan(dem)
        dem_filled = np.where(nan_mask, np.nanmean(dem), dem)
        curv = ndimage.convolve(dem_filled, kernel, mode='nearest')
        curv[nan_mask] = np.nan
        _write_derived_raster(out_path, curv, src)
    print(f"  wrote curvature to {out_path}")


def compute_hillshade_raster(dem_path, out_path, azimuth_deg=315.0, altitude_deg=45.0):
    """Hillshade (synthetic illumination from DEM). Output 0..255.
    Often visually useful for QGIS inspection; less obviously useful as a model
    input since slope+aspect carry the same info, but cheap to include."""
    if os.path.exists(out_path):
        print(f"  Hillshade already exists at {out_path}, skipping"); return
    with rasterio.open(dem_path) as src:
        dem = _read_band_as_float(src, 1)
        cell = _gsd_meters(src)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / (8.0 * cell)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / (8.0 * cell)
        nan_mask = np.isnan(dem)
        dem_filled = np.where(nan_mask, np.nanmean(dem), dem)
        dx = ndimage.convolve(dem_filled, kx, mode='nearest')
        dy = ndimage.convolve(dem_filled, ky, mode='nearest')
        slope = np.arctan(np.sqrt(dx * dx + dy * dy))
        aspect = np.arctan2(dy, -dx)
        az_rad = np.deg2rad(360.0 - azimuth_deg + 90.0)
        alt_rad = np.deg2rad(altitude_deg)
        hs = (np.cos(alt_rad) * np.cos(slope) +
              np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
        hs = np.clip(hs * 255.0, 0, 255).astype(np.uint8)
        hs[nan_mask] = 0
        # Write as uint8 (not float32) since it's just a visualization layer
        profile = src.profile.copy()
        profile.update(count=1, dtype='uint8', nodata=0,
                       compress='LZW', predictor=2, tiled=True,
                       blockxsize=256, blockysize=256)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(hs, 1)
    print(f"  wrote hillshade to {out_path}")


# ============================================================================
# Channel-dependent bands (require a binary channel mask raster)
# ============================================================================
def ensure_channel_mask_from_ndwi(paths, ndwi_key='ndwi', threshold=0.0,
                                  out_key='channel_mask',
                                  morph_close_m=0.5, min_area_m2=10.0):
    """Build a binary channel mask from NDWI thresholding + cleanup.

    Pixels with NDWI > threshold are 'water'. Apply morphological closing to
    bridge thin gaps, then remove connected components smaller than min_area_m2.

    Idempotent: skip if mask file exists. Updates paths dict in place.
    """
    src_path = paths[ndwi_key]
    src_dir = os.path.dirname(src_path)
    out_path = paths.get(out_key) or os.path.join(src_dir, 'channel_mask.tif')

    if os.path.exists(out_path):
        print(f"  channel mask already exists at {out_path}, skipping")
        paths[out_key] = out_path
        return paths

    with rasterio.open(src_path) as src:
        ndwi = _read_band_as_float(src, 1)
        cell = _gsd_meters(src)
        mask = (ndwi > threshold) & ~np.isnan(ndwi)
        # Morphological closing to bridge narrow channel breaks
        close_px = max(1, int(round(morph_close_m / cell)))
        mask = ndimage.binary_closing(mask, iterations=close_px)
        # Remove small connected components
        labeled, n = ndimage.label(mask)
        if n > 0:
            sizes = ndimage.sum(mask, labeled, range(1, n + 1))
            min_pix = int(min_area_m2 / (cell * cell))
            too_small = np.where(sizes < min_pix)[0] + 1
            for lid in too_small:
                mask[labeled == lid] = False
        # Write as uint8
        profile = src.profile.copy()
        profile.update(count=1, dtype='uint8', nodata=255,
                       compress='LZW', predictor=2, tiled=True,
                       blockxsize=256, blockysize=256)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
    paths[out_key] = out_path
    print(f"  wrote channel mask to {out_path}")
    return paths


def compute_distance_to_channel_raster(channel_mask_path, out_path):
    """Euclidean distance (in meters) from each pixel to nearest channel pixel."""
    if os.path.exists(out_path):
        print(f"  distance-to-channel already exists at {out_path}, skipping"); return
    with rasterio.open(channel_mask_path) as src:
        mask = src.read(1).astype(bool)
        cell = _gsd_meters(src)
        # distance_transform_edt: distance to nearest zero pixel
        # We want distance to nearest channel (mask==True), so invert
        dist_px = ndimage.distance_transform_edt(~mask)
        dist_m = dist_px.astype(np.float32) * cell
        _write_derived_raster(out_path, dist_m, src)
    print(f"  wrote distance-to-channel to {out_path}")


def compute_relative_elevation_raster(dem_path, channel_mask_path, out_path):
    """Height of each pixel above the nearest channel pixel.

    Both rasters must be on the same grid (same shape, transform). If they
    aren't, resample one to the other before calling this. Most useful when
    DEM and channel_mask are both at imagery GSD or at the DEM's native GSD.
    """
    if os.path.exists(out_path):
        print(f"  relative elevation already exists at {out_path}, skipping"); return
    with rasterio.open(dem_path) as dem_src, rasterio.open(channel_mask_path) as ch_src:
        if dem_src.shape != ch_src.shape or dem_src.transform != ch_src.transform:
            raise ValueError(
                f"DEM ({dem_src.shape} @ {dem_src.transform}) and channel mask "
                f"({ch_src.shape} @ {ch_src.transform}) must be on identical grids"
            )
        dem = _read_band_as_float(dem_src, 1)
        mask = ch_src.read(1).astype(bool)
        # For each pixel, find indices of nearest channel pixel
        _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
        nearest_channel_dem = dem[tuple(indices)]
        rel_elev = dem - nearest_channel_dem
        _write_derived_raster(out_path, rel_elev, dem_src)
    print(f"  wrote relative elevation to {out_path}")


# ============================================================================
# Texture / structure bands (from pan or any single-band raster)
# ============================================================================
def compute_local_std_raster(src_path, out_path, band=1, window_m=0.3):
    """Local standard deviation in a window (texture).
    Distinguishes mud / wrack / bare from short Spartina even when mean
    reflectance is similar. window_m is the window edge length in meters."""
    if os.path.exists(out_path):
        print(f"  local std already exists at {out_path}, skipping"); return
    with rasterio.open(src_path) as src:
        img = _read_band_as_float(src, band)
        cell = _gsd_meters(src)
        win = max(3, int(round(window_m / cell)))
        if win % 2 == 0:
            win += 1
        nan_mask = np.isnan(img)
        img_filled = np.where(nan_mask, np.nanmean(img), img)
        mean_f  = ndimage.uniform_filter(img_filled,           size=win, mode='nearest')
        sqmean  = ndimage.uniform_filter(img_filled * img_filled, size=win, mode='nearest')
        var = np.maximum(sqmean - mean_f * mean_f, 0.0)
        std = np.sqrt(var)
        std[nan_mask] = np.nan
        _write_derived_raster(out_path, std, src)
    print(f"  wrote local std ({window_m}m window) to {out_path}")


def compute_laplacian_raster(src_path, out_path, band=1):
    """Laplacian (second derivative) of a single band — picks up edges and
    fine-scale texture in the pan band."""
    if os.path.exists(out_path):
        print(f"  Laplacian already exists at {out_path}, skipping"); return
    with rasterio.open(src_path) as src:
        img = _read_band_as_float(src, band)
        nan_mask = np.isnan(img)
        img_filled = np.where(nan_mask, np.nanmean(img), img)
        lap = ndimage.laplace(img_filled)
        lap[nan_mask] = np.nan
        _write_derived_raster(out_path, lap, src)
    print(f"  wrote Laplacian to {out_path}")

def compute_local_range(src_path, out_path, window_m=0.3):
    """Local max - min in a window. Roughness/texture indicator.
    On DEM: how much elevation varies across the window — picks up burrow pock-marks.
    On pan: brightness range — picks up texture transitions."""
    import rasterio, numpy as np
    from scipy import ndimage
    with rasterio.open(src_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        gsd_m = abs(src.transform.a)
    window_px = max(3, int(round(window_m / gsd_m)))
    if window_px % 2 == 0:
        window_px += 1
    out = (ndimage.maximum_filter(data, size=window_px) -
           ndimage.minimum_filter(data, size=window_px)).astype(np.float32)
    profile.update(dtype='float32', count=1, compress='deflate', predictor=3)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(out, 1)


def compute_local_entropy(src_path, out_path, window_m=0.3, n_bins=16):
    """Shannon entropy of quantized values in a local window. Higher = more variety.
    Crab-burrowed areas have more spatial heterogeneity than smooth marsh platform."""
    import rasterio, numpy as np
    from scipy import ndimage
    with rasterio.open(src_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        gsd_m = abs(src.transform.a)
    valid = np.isfinite(data) & (data > 0)
    vmin, vmax = (np.percentile(data[valid], [1, 99]) if valid.any() else (0, 1))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    bin_idx = np.clip(((data - vmin) / (vmax - vmin) * n_bins).astype(np.int32),
                      0, n_bins - 1)
    window_px = max(3, int(round(window_m / gsd_m)))
    if window_px % 2 == 0:
        window_px += 1
    entropy = np.zeros_like(data, dtype=np.float32)
    for b in range(n_bins):
        p = ndimage.uniform_filter((bin_idx == b).astype(np.float32), size=window_px)
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy += np.where(p > 0, -p * np.log(p), 0)
    profile.update(dtype='float32', count=1, compress='deflate', predictor=3)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(entropy, 1)


def compute_dem_tri(dem_path, out_path):
    """Terrain Ruggedness Index (Riley 1999): mean absolute elevation difference
    from center pixel to 8 neighbors. Direct measure of micro-topographic roughness."""
    import rasterio, numpy as np
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    tri = np.zeros_like(dem)
    n = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            tri += np.abs(dem - np.roll(np.roll(dem, dy, axis=0), dx, axis=1))
            n += 1
    tri = (tri / n).astype(np.float32)
    profile.update(dtype='float32', count=1, compress='deflate', predictor=3)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(tri, 1)


# ensure_* wrappers
def ensure_local_range(paths, src_key='pan_orthomosaic', window_m=0.3, out_key='local_range'):
    return _ensure_one(paths, src_key, out_key, f'{out_key}.tif',
                       compute_local_range, window_m=window_m)

def ensure_local_entropy(paths, src_key='pan_orthomosaic', window_m=0.3, out_key='local_entropy'):
    return _ensure_one(paths, src_key, out_key, f'{out_key}.tif',
                       compute_local_entropy, window_m=window_m)

def ensure_dem_tri(paths, dem_key='dem_high_res', out_key='dem_tri'):
    return _ensure_one(paths, dem_key, out_key, f'{out_key}.tif', compute_dem_tri)
  
# ============================================================================
# ensure_* wrappers — follow the existing ensure_indices pattern
# ============================================================================
def _ensure_one(paths, source_key, out_key, default_filename, compute_fn, **kwargs):
    """Generic ensure-and-update helper."""
    source_path = paths[source_key]
    src_dir = os.path.dirname(source_path)
    out_path = paths.get(out_key) or os.path.join(src_dir, default_filename)
    compute_fn(source_path, out_path, **kwargs)
    paths[out_key] = out_path
    return paths


def ensure_savi(paths, ms_key='pansharp_ms', red_band=3, nir_band=5, L=0.5,
                out_key='savi'):
    return _ensure_one(paths, ms_key, out_key, 'savi.tif',
                       compute_savi_raster, red_band=red_band, nir_band=nir_band, L=L)


def ensure_evi(paths, ms_key='pansharp_ms', blue_band=1, red_band=3, nir_band=5,
               out_key='evi'):
    return _ensure_one(paths, ms_key, out_key, 'evi.tif',
                       compute_evi_raster, blue_band=blue_band, red_band=red_band,
                       nir_band=nir_band)


def ensure_gndvi(paths, ms_key='pansharp_ms', green_band=2, nir_band=5, out_key='gndvi'):
    return _ensure_one(paths, ms_key, out_key, 'gndvi.tif',
                       compute_gndvi_raster, green_band=green_band, nir_band=nir_band)


def ensure_ndwi(paths, ms_key='pansharp_ms', green_band=2, nir_band=5, out_key='ndwi'):
    return _ensure_one(paths, ms_key, out_key, 'ndwi.tif',
                       compute_ndwi_raster, green_band=green_band, nir_band=nir_band)


def ensure_ci_rededge(paths, ms_key='pansharp_ms', re_band=4, nir_band=5,
                      out_key='ci_rededge'):
    return _ensure_one(paths, ms_key, out_key, 'ci_rededge.tif',
                       compute_ci_rededge_raster, re_band=re_band, nir_band=nir_band)


def ensure_slope(paths, dem_key='dem_high_res', units='degrees', out_key='slope'):
    return _ensure_one(paths, dem_key, out_key, 'slope.tif',
                       compute_slope_raster, units=units)


def ensure_tpi(paths, dem_key='dem_high_res', neighborhood_m=2.0, out_key='tpi'):
    return _ensure_one(paths, dem_key, out_key, f'{out_key}.tif',
                       compute_tpi_raster, neighborhood_m=neighborhood_m)


def ensure_curvature(paths, dem_key='dem_high_res', out_key='curvature'):
    return _ensure_one(paths, dem_key, out_key, 'curvature.tif',
                       compute_curvature_raster)


def ensure_hillshade(paths, dem_key='dem_high_res',
                     azimuth_deg=315.0, altitude_deg=45.0, out_key='hillshade'):
    return _ensure_one(paths, dem_key, out_key, 'hillshade.tif',
                       compute_hillshade_raster,
                       azimuth_deg=azimuth_deg, altitude_deg=altitude_deg)


def ensure_distance_to_channel(paths, channel_mask_key='channel_mask',
                               out_key='dist_to_channel'):
    return _ensure_one(paths, channel_mask_key, out_key, 'dist_to_channel.tif',
                       compute_distance_to_channel_raster)


def ensure_relative_elevation(paths, dem_key='dem_high_res',
                              channel_mask_key='channel_mask',
                              out_key='rel_elevation'):
    """Special-cased because it needs two source rasters."""
    dem_path = paths[dem_key]
    ch_path = paths[channel_mask_key]
    out_path = paths.get(out_key) or os.path.join(
        os.path.dirname(dem_path), 'rel_elevation.tif')
    compute_relative_elevation_raster(dem_path, ch_path, out_path)
    paths[out_key] = out_path
    return paths


def ensure_local_std(paths, src_key='pan_orthomosaic', band=1, window_m=0.3,
                    out_key='local_std'):
    return _ensure_one(paths, src_key, out_key, 'local_std.tif',
                       compute_local_std_raster, band=band, window_m=window_m)


def ensure_laplacian(paths, src_key='pan_orthomosaic', band=1, out_key='laplacian'):
    return _ensure_one(paths, src_key, out_key, 'laplacian.tif',
                       compute_laplacian_raster, band=band)


# ============================================================================
# Aggregate convenience
# ============================================================================
def ensure_optional_bands(paths, which=('slope', 'savi', 'ndwi'), **kwargs):
    """Run a set of ensure_* functions in one call.

    Pass which=('slope', 'savi', 'ndwi', 'dist_to_channel', ...) — any subset of:
        savi, evi, gndvi, ndwi, ci_rededge,
        slope, tpi, curvature, hillshade,
        channel_mask, dist_to_channel, rel_elevation,
        local_std, laplacian

    Order matters for derived-from-derived: 'channel_mask' should come before
    'dist_to_channel' / 'rel_elevation' (it's their input).

    Returns paths (updated in place).
    """
    registry = {
        'savi':            ensure_savi,
        'evi':             ensure_evi,
        'gndvi':           ensure_gndvi,
        'ndwi':            ensure_ndwi,
        'ci_rededge':      ensure_ci_rededge,
        'slope':           ensure_slope,
        'tpi':             ensure_tpi,
        'curvature':       ensure_curvature,
        'hillshade':       ensure_hillshade,
        'channel_mask':    ensure_channel_mask_from_ndwi,
        'dist_to_channel': ensure_distance_to_channel,
        'rel_elevation':   ensure_relative_elevation,
        'local_std':       ensure_local_std,
        'laplacian':       ensure_laplacian,
    }
    for name in which:
        if name not in registry:
            raise ValueError(f"Unknown band {name!r}. Valid: {list(registry)}")
        registry[name](paths, **kwargs.get(name, {}))
    return paths

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


def build_patches_with_splits_multi(paths, polygons_gdf, cfg, resampling=Resampling.bilinear):
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
    band_spec=cfg.BAND_SPEC
    patch_size=cfg.PATCH_SIZE
    overlap=cfg.OVERLAP
    block_size_m=cfg.BLOCK_SIZE_M
    priority=cfg.PRIORITY
    train_frac=cfg.TRAIN_FRAC
    val_frac=cfg.VAL_FRAC
    seed=cfg.SEED
    require_labels=cfg.REQUIRE_LABELS
    class_col=cfg.CLASS_COLUMN
    ignore_value = cfg.IGNORE_INDEX
  
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

    for i, p in enumerate(patches):
        if (i+1) % max(1, len(patches)//10) == 0:
            print(f"  {i+1}/{len(patches)} patches...")
        img = p['image']
        if hasattr(img, 'detach'):
           img = img.detach().cpu().numpy()
        img = img.astype(np.float64)
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
    # Derive autocast parameters from the inputs
    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
    amp_enabled = scaler is not None
  
    model.train()
    total_loss, n_batches = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type, enabled=amp_enabled):
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
    # Derive autocast parameters from the inputs
    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
    amp_enabled = device_type == 'cuda'
  
    model.eval()
    total_loss, n_batches = 0.0, 0
    metric.reset()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        with torch.amp.autocast(device_type, enabled=amp_enabled):
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

#  PRODUCTION ==========================

def _gaussian_weight(patch_size, sigma_ratio=0.3):
    """Gaussian falloff centered on the patch — suppresses edge artifacts."""
    coords = np.linspace(-1, 1, patch_size)
    x, y = np.meshgrid(coords, coords)
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma_ratio ** 2)).astype(np.float32)


def _patch_positions(h, w, patch_size, stride):
    """Sliding-window positions ensuring the right/bottom edges are covered."""
    rows = list(range(0, h - patch_size + 1, stride))
    if rows and rows[-1] != h - patch_size:
        rows.append(h - patch_size)
    cols = list(range(0, w - patch_size + 1, stride))
    if cols and cols[-1] != w - patch_size:
        cols.append(w - patch_size)
    return [(r, c) for r in rows for c in cols]

def predict_full_raster(
    model,
    cfg,
    paths,
    channel_means,
    channel_stds,
    output_path,
    device='cuda',
    use_amp=True,
):
    """
    Sliding-window inference over the full reference raster.

    Writes a multi-band GeoTIFF with one band per class, each containing the
    per-pixel softmax probability for that class. Overlapping patch predictions
    are blended with a Gaussian weighting centered on each patch.

    Patch size, overlap, batch size, band spec, normalization stats, class
    names, and N_CLASSES all come from cfg. The reference raster (and output
    grid) is the first raster in cfg.BAND_SPEC. Other rasters are aligned via
    WarpedVRT if their grids differ.
    """
    band_spec = [tuple(item) for item in cfg.BAND_SPEC]
    mean = np.asarray(channel_means, dtype=np.float32)
    std  = np.asarray(channel_stds,  dtype=np.float32)

    stride = max(1, int(cfg.PATCH_SIZE * (1 - cfg.OVERLAP)))
    weight = _gaussian_weight(cfg.PATCH_SIZE)

    # Resolve unique source raster paths in band_spec order
    raster_paths_ordered = []
    for key, _ in band_spec:
        p = paths[key]
        if p not in raster_paths_ordered:
            raster_paths_ordered.append(p)
    band_spec_resolved = [(paths[key], idx) for key, idx in band_spec]

    with ExitStack() as stack:
        # Reference raster defines the output grid
        ref_src = stack.enter_context(rasterio.open(raster_paths_ordered[0]))
        sources = {raster_paths_ordered[0]: ref_src}

        # Open and align other rasters
        for p in raster_paths_ordered[1:]:
            src = stack.enter_context(rasterio.open(p))
            if _grids_match(src, ref_src):
                sources[p] = src
            else:
                sources[p] = stack.enter_context(WarpedVRT(
                    src, crs=ref_src.crs, transform=ref_src.transform,
                    width=ref_src.width, height=ref_src.height,
                    resampling=Resampling.bilinear,
                ))

        # Group bands by raster for efficient reads
        bands_by_path = {}
        for path, band_idx in band_spec_resolved:
            bands_by_path.setdefault(path, []).append(band_idx)

        h, w = ref_src.height, ref_src.width
        positions = _patch_positions(h, w, cfg.PATCH_SIZE, stride)
        print(f"Inference grid: {h}×{w}, {len(positions)} patches "
              f"({cfg.PATCH_SIZE}×{cfg.PATCH_SIZE}, stride {stride})")

        # In-memory accumulators
        probs_acc   = np.zeros((cfg.N_CLASSES, h, w), dtype=np.float32)
        weights_acc = np.zeros((h, w),                dtype=np.float32)

        model.eval()
        for batch_start in range(0, len(positions), cfg.BATCH_SIZE):
            batch_positions = positions[batch_start: batch_start + cfg.BATCH_SIZE]

            # Read and normalize each patch in the batch
            batch_images = []
            for row, col in batch_positions:
                window = Window(col, row, cfg.PATCH_SIZE, cfg.PATCH_SIZE)
                arrays_by_key = {}
                for path, band_indices in bands_by_path.items():
                    src = sources[path]
                    arr = src.read(band_indices, window=window)
                    for i, b in enumerate(band_indices):
                        arrays_by_key[(path, b)] = arr[i]

                image = np.stack(
                    [arrays_by_key[(p, b)] for p, b in band_spec_resolved],
                    axis=0,
                ).astype(np.float32)

                # Replace NaN from NDVI/NDRE with 0 before normalization
                image = np.nan_to_num(image, nan=0.0)
                image = (image - mean[:, None, None]) / std[:, None, None]
                batch_images.append(image)

            batch = torch.from_numpy(np.stack(batch_images)).to(device, non_blocking=True)

            with torch.no_grad():
                device_type = next(model.parameters()).device.type        # 'cuda' or 'cpu'
                amp_enabled = use_amp and device_type == 'cuda'
                with torch.amp.autocast(device_type, enabled=amp_enabled):
                    logits = model(batch)
                    probs  = torch.softmax(logits, dim=1)
            probs_np = probs.float().cpu().numpy()      # (B, C, H, W)

            # Gaussian-weighted accumulation
            for i, (row, col) in enumerate(batch_positions):
                probs_acc[:, row:row + cfg.PATCH_SIZE, col:col + cfg.PATCH_SIZE] \
                    += probs_np[i] * weight
                weights_acc[row:row + cfg.PATCH_SIZE, col:col + cfg.PATCH_SIZE] \
                    += weight

            if (batch_start // cfg.BATCH_SIZE) % 50 == 0:
                done = min(batch_start + cfg.BATCH_SIZE, len(positions))
                print(f"  {done}/{len(positions)} patches")

        # Normalize accumulated probs by accumulated weights
        weights_safe = np.maximum(weights_acc, 1e-8)
        probs_final  = probs_acc / weights_safe[None]

        # Write multi-band GeoTIFF — one band per class
        profile = ref_src.profile.copy()
        profile.update(
            count=cfg.N_CLASSES,
            dtype='float32',
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=None,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            for c in range(cfg.N_CLASSES):
                dst.write(probs_final[c].astype(np.float32), c + 1)
                dst.set_band_description(c + 1, cfg.CLASS_NAMES.get(c, f'class_{c}'))

        print(f"Wrote predictions: {output_path}")
        print(f"  shape: {probs_final.shape}  classes: {list(cfg.CLASS_NAMES.values())}")

import rasterio

def check_alignment(reference_path, *other_paths, precision=1e-6, verbose=True):
    """Check that rasters share CRS, transform, and shape with a reference.

    Pixel-for-pixel intersection (abstain raster x superpixel raster) is only
    valid when all three match. Returns True iff every other raster aligns with
    the reference; prints a per-raster report when verbose.
    """
    def grid_of(path):
        with rasterio.open(path) as src:
            return src.crs, src.transform, (src.height, src.width)

    ref_crs, ref_tf, ref_shape = grid_of(reference_path)
    if verbose:
        print(f"reference: {reference_path}")
        print(f"  crs={ref_crs.to_string() if ref_crs else None}  "
              f"size={ref_shape[1]}x{ref_shape[0]}  px=({ref_tf.a:g},{ref_tf.e:g})  "
              f"origin=({ref_tf.c:g},{ref_tf.f:g})")

    all_ok = True
    for p in other_paths:
        crs, tf, shape = grid_of(p)
        crs_ok, tf_ok, shape_ok = (crs == ref_crs), tf.almost_equals(ref_tf, precision=precision), (shape == ref_shape)
        ok = crs_ok and tf_ok and shape_ok
        all_ok = all_ok and ok
        if verbose:
            print(f"  [{'OK' if ok else 'XX'}] {p}")
            if not crs_ok:   print(f"        crs differs: {crs.to_string() if crs else None}")
            if not shape_ok: print(f"        size differs: {shape[1]}x{shape[0]} vs ref {ref_shape[1]}x{ref_shape[0]}")
            if not tf_ok:    print(f"        grid differs: px=({tf.a:g},{tf.e:g}) origin=({tf.c:g},{tf.f:g})")
    if verbose:
        print("ALIGNED" if all_ok else "NOT ALIGNED -- fix before intersecting")
    return all_ok
  
"""
abstain.py -- build the per-pixel "abstain bucket" from a cached softmax raster.

Lives in the Model 1 production stage, but derived from the CACHED softmax (the
same artifact the decision rules use) rather than welded into inference -- so the
abstain thresholds can be re-tuned with a cheap CPU pass, no model re-run.

For each valid pixel it looks at the top-2 class probabilities:
  margin = P1 - P2
  abstain      when margin < min_margin
    -> pair    when the top-2 hold most of the mass (P1 + P2 >= mass_cutoff):
               a genuine two-way tie; record WHICH two classes.
    -> diffuse otherwise: the mass is smeared across many classes.

Output: a single-band uint8 GeoTIFF aligned to the softmax raster, where
  0          = not abstain (confident) or outside footprint
  1..15      = abstain-pair, encoding the unordered contested class pair
  100        = abstain-diffuse
plus a JSON legend mapping each code to its class-pair names.

Requires: rasterio, numpy.
"""

import os
import json
import itertools
import numpy as np
import rasterio

DIFFUSE_CODE = 100

DEFAULT_CLASS_NAMES = [
    "other", "healthy_bank", "eroding_non_crab",
    "crab_edge", "crab_platform", "collapsed",
]


def _pair_codes(n_classes):
    """Map each unordered class pair to a small integer code (1..C(n,2))."""
    pairs = list(itertools.combinations(range(n_classes), 2))
    return {pair: k + 1 for k, pair in enumerate(pairs)}, pairs

"""
abstain_review.py -- intersect the abstain bucket with the superpixel containers
and promote high-abstain containers into a tagged review layer for QGIS.

For every container (superpixel id > 0) it computes, from the per-pixel abstain
codes:
  n_pixels      container size
  n_pair        pixels flagged as a two-way tie (codes 1..15)
  n_diffuse     pixels flagged diffuse (code 100)
  abstain_frac  n_pair / n_pixels         <- the promotion metric
  contested_pair  the container's DOMINANT pair (mode over its pair pixels)
  pair_purity   share of pair pixels that are the dominant pair (low => mixed)

Promotion is driven by PAIR abstentions only (diffuse is tracked but does not
promote -- pairs are the high-yield labeling target, diffuse is a diagnostic).
Output: a GeoPackage of promoted container polygons, one labelable unit each,
tagged with the contested pair and sorted into a work queue.

Requires: rasterio, numpy, geopandas, shapely.
"""

import os
import json
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

_SCHEMA = ["container_id", "n_pixels", "n_pair", "n_diffuse",
           "abstain_frac", "diffuse_frac", "pair_purity",
           "pair_code", "contested_pair"]


def build_abstain_review_polygons(
    superpixel_path,
    abstain_path,
    out_gpkg,
    min_abstain_frac=0.30,
    min_pair_pixels=0,
    legend_path=None,
    layer="abstain_review",
):
    """Promote high-abstain containers to a tagged review GeoPackage.

    min_abstain_frac : promote a container when its pair-abstain fraction >= this.
    min_pair_pixels  : also require at least this many pair pixels (guards tiny
                       containers tripping the fraction on a couple of pixels).
    legend_path      : pair-code -> [classA, classB] JSON; defaults to the
                       '<abstain>_legend.json' written by build_abstain_raster.

    Returns the GeoDataFrame (also written to out_gpkg). Empty if none promoted.
    """
    if legend_path is None:
        legend_path = os.path.splitext(abstain_path)[0] + "_legend.json"
    legend = json.load(open(legend_path)) if os.path.exists(legend_path) else {}

    with rasterio.open(superpixel_path) as ssrc:
        seg = ssrc.read(1).astype(np.int64)
        crs, transform = ssrc.crs, ssrc.transform
        sshape = (ssrc.height, ssrc.width)
    with rasterio.open(abstain_path) as asrc:
        ab = asrc.read(1)
        ashape = (asrc.height, asrc.width)
    if sshape != ashape:
        raise ValueError(f"grid mismatch: superpixels {sshape} vs abstain {ashape}")

    n_ids = int(seg.max())
    if n_ids == 0:
        return _empty(crs)

    valid = seg > 0
    size = np.bincount(seg[valid].ravel(), minlength=n_ids + 1)

    is_pair = (ab >= 1) & (ab <= 15) & valid
    is_diffuse = (ab == 100) & valid
    pair_count = np.bincount(seg[is_pair].ravel(), minlength=n_ids + 1)
    diffuse_count = np.bincount(seg[is_diffuse].ravel(), minlength=n_ids + 1)

    # Dominant pair code per container via a (container_id, code) histogram.
    cid = seg[is_pair].ravel()
    code = ab[is_pair].ravel().astype(np.int64)
    hist = np.bincount(cid * 16 + code,
                       minlength=(n_ids + 1) * 16).reshape(n_ids + 1, 16)
    dom_code = hist.argmax(axis=1)
    dom_count = hist.max(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        abstain_frac = np.where(size > 0, pair_count / size, 0.0)
        diffuse_frac = np.where(size > 0, diffuse_count / size, 0.0)
        pair_purity = np.where(pair_count > 0, dom_count / pair_count, 0.0)

    promote = (abstain_frac >= min_abstain_frac) & (pair_count >= max(1, min_pair_pixels))
    if not promote[1:].any():
        print(f"no containers promoted at min_abstain_frac={min_abstain_frac}")
        return _empty(crs)

    promoted_mask = promote[seg]                 # per-pixel; promote[0] is False
    seg_i32 = seg.astype(np.int32)

    feats = []
    for geom, cid_f in shapes(seg_i32, mask=promoted_mask,
                              transform=transform, connectivity=4):
        c = int(cid_f)
        if c == 0:
            continue
        dc = int(dom_code[c])
        names = [n for n in legend.get(str(dc), [f"code_{dc}"]) if n]
        feats.append({
            "container_id": c,
            "n_pixels": int(size[c]),
            "n_pair": int(pair_count[c]),
            "n_diffuse": int(diffuse_count[c]),
            "abstain_frac": round(float(abstain_frac[c]), 3),
            "diffuse_frac": round(float(diffuse_frac[c]), 3),
            "pair_purity": round(float(pair_purity[c]), 3),
            "pair_code": dc,
            "contested_pair": "|".join(names),
            "geometry": shape(geom),
        })

    gdf = gpd.GeoDataFrame(feats, crs=crs)
    gdf = gdf.sort_values(["contested_pair", "abstain_frac"],
                          ascending=[True, False]).reset_index(drop=True)

    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)
    os.makedirs(os.path.dirname(out_gpkg) or ".", exist_ok=True)
    gdf.to_file(out_gpkg, layer=layer, driver="GPKG")

    print(f"promoted {gdf['container_id'].nunique()} containers -> {out_gpkg}")
    print(gdf.groupby("contested_pair")["abstain_frac"].agg(["count", "mean"]).round(3))
    return gdf


def _empty(crs):
    return gpd.GeoDataFrame(columns=_SCHEMA, geometry=[], crs=crs)

def build_abstain_raster(
    softmax_path,
    out_path,
    min_margin=0.15,
    mass_cutoff=0.80,
    class_names=None,
    require_classes=None,    # only flag pairs that involve one of these class indices
    write_legend=True,
):
    """Derive a per-pixel abstain-code raster from a cached softmax GeoTIFF.

    softmax_path : multi-band float GeoTIFF, one band per class, per-pixel
                   probabilities (need not be perfectly normalized). Nodata /
                   outside-footprint pixels should read as all-zero or non-finite.
    out_path     : single-band uint8 abstain-code raster to write.
    min_margin   : abstain when (P1 - P2) < this.
    mass_cutoff  : among abstains, pair-case when (P1 + P2) >= this, else diffuse.
    require_classes : optional iterable of class indices; if given, only pairs that
                   include one of these get a code (others -> 0). e.g. restrict to
                   the damage classes so you don't queue irrelevant ties.

    Returns (out_path, stats_dict) where stats_dict maps a label -> pixel count.
    """
    with rasterio.open(softmax_path) as src:
        sm = src.read().astype(np.float32)          # (C, H, W)
        profile = src.profile.copy()
    sm = np.transpose(sm, (1, 2, 0))                # (H, W, C)
    C = sm.shape[-1]
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES[:C] if C <= len(DEFAULT_CLASS_NAMES) \
            else [f"class_{k}" for k in range(C)]

    valid = np.isfinite(sm).all(axis=-1) & (sm.sum(axis=-1) > 0.5)

    order = np.argsort(-sm, axis=-1)                # descending
    i = order[..., 0]
    j = order[..., 1]
    P1 = np.take_along_axis(sm, i[..., None], axis=-1)[..., 0]
    P2 = np.take_along_axis(sm, j[..., None], axis=-1)[..., 0]

    margin = P1 - P2
    mass = P1 + P2
    abstain = valid & (margin < min_margin)
    pair_case = abstain & (mass >= mass_cutoff)
    diffuse_case = abstain & ~pair_case

    code_of, pairs = _pair_codes(C)
    code_map = np.zeros((C, C), dtype=np.uint8)
    for (a, b), k in code_of.items():
        code_map[a, b] = k
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    pair_code = code_map[lo, hi]

    out = np.zeros(sm.shape[:2], dtype=np.uint8)
    out[pair_case] = pair_code[pair_case]
    out[diffuse_case] = DIFFUSE_CODE

    if require_classes is not None:
        req = set(int(c) for c in require_classes)
        for (a, b), k in code_of.items():
            if a not in req and b not in req:
                out[out == k] = 0

    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "abstain_code")

    legend = {str(k): [class_names[a], class_names[b]] for (a, b), k in code_of.items()}
    legend[str(DIFFUSE_CODE)] = ["diffuse", "diffuse"]
    if write_legend:
        with open(os.path.splitext(out_path)[0] + "_legend.json", "w") as f:
            json.dump(legend, f, indent=2)

    # stats: how many pixels per code (the diagnostic -- e.g. healthy|crab share)
    stats = {}
    present, counts = np.unique(out[out > 0], return_counts=True)
    for code, n in zip(present.tolist(), counts.tolist()):
        label = "diffuse" if code == DIFFUSE_CODE else "|".join(legend[str(code)])
        stats[label] = n
    return out_path, stats
  
import os
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
import geopandas as gpd


def predictions_to_polygons(
    prob_raster_path,
    cfg,
    classes=None,
    min_area_m2=1.0,
    morph_close_pixels=3,
    morph_open_pixels=2,
    abstain_raster_path=None,
):
    """
    Extract polygons from a multi-band probability raster.
    For each class of interest:
      1. Identify pixels where argmax == class AND max_prob >= threshold[class]
      2. Exclude pixels flagged in the abstain raster (sent to the review queue)
      3. Apply morphological closing (fill small gaps) and opening (drop noise)
      4. Run connected components -> one region per contiguous blob
      5. Vectorize to polygons, attach mean confidence and area
      6. Drop polygons below min_area_m2

    If abstain_raster_path is given, any pixel with abstain_code > 0 is withheld
    from every class mask, so the confident polygons and the abstain bucket
    partition cleanly (a pixel lands in exactly one). The exclusion respects
    whatever require_classes was used to build the abstain raster.

    Returns:
        GeoDataFrame with columns:
            class, class_name, threshold, mean_confidence, area_m2, geometry
    """
    if classes is None:
        classes = cfg.CLASSES_OF_INTEREST
    thresholds = cfg.CONFIDENCE_THRESHOLDS or {}
    with rasterio.open(prob_raster_path) as src:
        crs       = src.crs
        transform = src.transform
        probs     = src.read()                      # (C, H, W)
    argmax   = probs.argmax(axis=0).astype(np.int32)   # (H, W) best class per pixel
    max_prob = probs.max(axis=0)                       # (H, W) max softmax per pixel

    # Pixels the abstain bucket owns are withheld from every confident mask.
    abstain_mask = None
    if abstain_raster_path is not None:
        with rasterio.open(abstain_raster_path) as asrc:
            abstain_mask = asrc.read(1) > 0            # any flagged pixel: pair or diffuse
        if abstain_mask.shape != argmax.shape:
            raise ValueError(
                "abstain raster grid does not match prob raster grid "
                f"({abstain_mask.shape} vs {argmax.shape}) -- check alignment"
            )

    all_features = []
    for cls in classes:
        threshold = thresholds.get(cls, 0.5)
        mask = (argmax == cls) & (max_prob >= threshold)
        if abstain_mask is not None:
            mask &= ~abstain_mask                      # abstained pixels -> review queue, not here
        if not mask.any():
            print(f"  class {cls} ({cfg.CLASS_NAMES[cls]}): "
                  f"no pixels above threshold {threshold}")
            continue
        # Morphology cleanup
        if morph_close_pixels > 0:
            mask = ndimage.binary_closing(mask, iterations=morph_close_pixels)
        if morph_open_pixels > 0:
            mask = ndimage.binary_opening(mask, iterations=morph_open_pixels)
        if not mask.any():
            print(f"  class {cls} ({cfg.CLASS_NAMES[cls]}): "
                  f"no pixels remain after morphology")
            continue
        # Connected components -- each region gets a unique label
        labeled, n_components = ndimage.label(mask)
        # Mean confidence per labeled region (vectorized over all labels at once)
        mean_confs = ndimage.mean(
            max_prob, labels=labeled, index=range(1, n_components + 1)
        )
        # Vectorize: rasterio.features.shapes returns one polygon per labeled region
        labeled_i32 = labeled.astype(np.int32)
        for geom, label_id in shapes(labeled_i32, mask=mask, transform=transform):
            label_id = int(label_id)
            if label_id == 0:
                continue
            geom_obj = shape(geom)
            area_m2 = geom_obj.area     # projected CRS -> meters^2
            if area_m2 < min_area_m2:
                continue
            all_features.append({
                'class':           cls,
                'class_name':      cfg.CLASS_NAMES[cls],
                'threshold':       threshold,
                'mean_confidence': float(mean_confs[label_id - 1]),
                'area_m2':         float(area_m2),
                'geometry':        geom_obj,
            })
    if not all_features:
        return gpd.GeoDataFrame(
            columns=['class', 'class_name', 'threshold',
                     'mean_confidence', 'area_m2'],
            geometry=[], crs=crs,
        )
    gdf = gpd.GeoDataFrame(all_features, crs=crs)
    gdf = gdf.sort_values(['class', 'area_m2'], ascending=[True, False]).reset_index(drop=True)
    print(f"\nExtracted {len(gdf)} polygons:")
    print(gdf.groupby('class_name')['area_m2'].agg(['count', 'sum', 'mean']).round(2))
    return gdf

#  SYNTHETIC DATA GENERATION ===========================

# ============================================================================
# Geometry generators
# ============================================================================
def sinuous_line(start, end, n_points=100, amplitude=8.0, n_waves=2.5,
                 noise_amp=2.0, rng=None):
    """Generate a sinuous LineString between two points."""
    if rng is None: rng = np.random.default_rng()
    t = np.linspace(0, 1, n_points)
    xs = np.interp(t, [0, 1], [start[0], end[0]])
    ys = np.interp(t, [0, 1], [start[1], end[1]])
    dx, dy = end[0]-start[0], end[1]-start[1]
    L = np.hypot(dx, dy)
    px, py = -dy/L, dx/L
    phase = rng.uniform(0, 2*np.pi)
    sine = amplitude * np.sin(t * n_waves * 2 * np.pi + phase) * np.sin(t * np.pi)
    noise = rng.normal(0, noise_amp, n_points)
    noise = np.convolve(noise, np.ones(7)/7, mode='same')
    perturb = sine + noise
    return LineString(list(zip(xs + perturb * px, ys + perturb * py)))


def branching_tributary(parent_line, parent_t, length, angle_deg,
                       amplitude=4.0, n_waves=1.5, rng=None):
    """Branch off a parent line at parameterized position parent_t."""
    if rng is None: rng = np.random.default_rng()
    coords = list(parent_line.coords)
    idx = max(1, min(len(coords)-2, int(parent_t * (len(coords) - 1))))
    tx = coords[idx+1][0] - coords[idx-1][0]
    ty = coords[idx+1][1] - coords[idx-1][1]
    tlen = np.hypot(tx, ty)
    tx, ty = tx/tlen, ty/tlen
    a = np.deg2rad(angle_deg)
    bx = tx*np.cos(a) - ty*np.sin(a)
    by = tx*np.sin(a) + ty*np.cos(a)
    start = (coords[idx][0], coords[idx][1])
    end = (start[0] + bx*length, start[1] + by*length)
    return sinuous_line(start, end, n_points=60, amplitude=amplitude,
                        n_waves=n_waves, noise_amp=1.0, rng=rng)


def generate_marsh_geometry(bounds, rng):
    """Generate channels, banks, trees, hummock, and ponds.

    Bumped tributary count (5 → 8) and sub-tributary frequency so the M1 window
    contains more bank polygons across all class characters.
    """
    xmin, ymin, xmax, ymax = bounds.bounds
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # ── Main channel — north-south sinuous, slight diagonal ──
    main = sinuous_line(
        start=(xmin + 25, ymax - 3),
        end=(xmin + 35, ymin + 3),
        n_points=140, amplitude=10.0, n_waves=3.0, noise_amp=2.0, rng=rng,
    )

    # ── Main tributaries — denser spacing along the main channel ──
    tribs = []
    for parent_t, angle, length in [
        (0.12, 70,  55),
        (0.22, 85,  65),
        (0.32, 95,  75),
        (0.42, 80,  70),
        (0.52, 75,  60),
        (0.62, 90,  72),
        (0.72, 100, 65),
        (0.82, 85,  55),
    ]:
        tribs.append(branching_tributary(main, parent_t, length, angle, rng=rng))

    # ── Sub-tributaries — higher branching probability ──
    sub_tribs = []
    for trib in tribs:
        if rng.random() > 0.2:                  # was 0.3
            for _ in range(rng.integers(1, 4)): # was 1..3
                t_branch = rng.uniform(0.3, 0.8)
                angle    = rng.choice([-60, -45, 45, 60]) + rng.normal(0, 10)
                length   = rng.uniform(15, 35)
                sub_tribs.append(branching_tributary(
                    trib, t_branch, length, angle,
                    amplitude=1.5, n_waves=1.2, rng=rng,
                ))

    # ── Water and bank buffers ──
    main_water  = main.buffer(4.0, cap_style=2)
    trib_waters = [t.buffer(1.0, cap_style=2) for t in tribs]
    sub_waters  = [s.buffer(0.5, cap_style=2) for s in sub_tribs]
    all_water   = unary_union([main_water] + trib_waters + sub_waters).intersection(bounds)

    main_bank  = main_water.buffer(2.5).difference(main_water)
    trib_bank  = unary_union([w.buffer(1.0).difference(w) for w in trib_waters])
    sub_bank   = unary_union([w.buffer(0.4).difference(w) for w in sub_waters])
    all_banks  = (unary_union([main_bank, trib_bank, sub_bank])
                  .intersection(bounds).difference(all_water))

    # ── Trees along the bottom edge ──
    bottom_pts = [(xmin, ymin)]
    for x in np.linspace(xmin, xmax, 30):
        bottom_pts.append((x, ymin + 12 + rng.uniform(-4, 12)))
    bottom_pts.append((xmax, ymin))
    bottom_trees = Polygon(bottom_pts).buffer(0)

    # ── Trees along the right edge ──
    right_pts = [(xmax, ymax)]
    for y in np.linspace(ymax, ymin + 30, 25):
        right_pts.append((xmax + rng.uniform(-12, -2), y))
    right_pts.append((xmax, ymin + 30))
    right_trees = Polygon(right_pts).buffer(0)

    all_trees = unary_union([bottom_trees, right_trees]).intersection(bounds).difference(all_water)

    # ── Hummock — vegetated island in the marsh ──
    hum_center = (cx + 30, cy + 15)
    hum_pts = []
    for theta in np.linspace(0, 2*np.pi, 30):
        r = 10 + rng.uniform(-3, 3)
        hum_pts.append((hum_center[0] + r*np.cos(theta),
                        hum_center[1] + r*np.sin(theta)))
    hummock = Polygon(hum_pts).buffer(0).intersection(bounds).difference(all_water)

    # ── Ponds — small marsh-interior depressions ──
    ponds = []
    for _ in range(3):
        pond_cx = rng.uniform(xmin + 40, xmax - 20)     # renamed from cx
        pond_cy = rng.uniform(ymin + 35, ymax - 20)     # renamed from cy
        r       = rng.uniform(1.5, 3.0)
        p_pts = []
        for theta in np.linspace(0, 2*np.pi, 16):
            rr = r + rng.uniform(-0.3, 0.3)
            p_pts.append((pond_cx + rr*np.cos(theta),
                          pond_cy + rr*np.sin(theta)))
        ponds.append(Polygon(p_pts))
    all_ponds = (unary_union(ponds).intersection(bounds)
                 .difference(all_water).difference(all_banks))

    return dict(
        main_channel=main,
        tributaries=tribs + sub_tribs,
        all_water=all_water, all_banks=all_banks,
        all_trees=all_trees, hummock=hummock, all_ponds=all_ponds,
    )

def generate_other_handlabels(geom, bounds, rng,
                               n_marsh_interior=10,
                               n_mud_patches=5,
                               n_wrack=3):
    """Generate synthetic 'hand-labeled' polygons for Model 2's 'other' class.
    
    Combines geometric features (trees, hummock, ponds) with simulated marsh-interior,
    mud-patch, and wrack-line polygons — capturing the variety a human would label
    in QGIS over real 4cm imagery.
    
    All output polygons get Class = CLASSES['other'].
    """
    from shapely.geometry import Polygon, MultiPolygon
    
    xmin, ymin, xmax, ymax = bounds.bounds
    extent_x = xmax - xmin
    extent_y = ymax - ymin
    rows = []
    
    def _add_geom(g, min_area=0.5):
        if g.is_empty: return
        if isinstance(g, MultiPolygon):
            for p in g.geoms:
                if p.area > min_area:
                    rows.append({'Class': CLASSES['other'], 'geometry': p})
        else:
            rows.append({'Class': CLASSES['other'], 'geometry': g})
    
    # 1) Pre-existing geometric features
    _add_geom(geom['all_trees'], min_area=1.0)
    _add_geom(geom['hummock'])
    _add_geom(geom['all_ponds'])
    
    # 2) Random marsh-interior 'other' patches (irregular, mid-size)
    placed, attempts = 0, 0
    while placed < n_marsh_interior and attempts < 100:
        attempts += 1
        cx = rng.uniform(xmin + 0.15*extent_x, xmax - 0.15*extent_x)
        cy = rng.uniform(ymin + 0.20*extent_y, ymax - 0.15*extent_y)
        r = rng.uniform(0.8, 2.5)
        n_verts = rng.integers(8, 14)
        pts = []
        for theta in np.linspace(0, 2*np.pi, n_verts, endpoint=False):
            rr = r * rng.uniform(0.6, 1.4)   # irregular boundary
            pts.append((cx + rr*np.cos(theta), cy + rr*np.sin(theta)))
        poly = Polygon(pts)
        if poly.intersects(geom['all_water']) or poly.intersects(geom['all_banks']):
            continue
        rows.append({'Class': CLASSES['other'], 'geometry': poly})
        placed += 1
    
    # 3) Mud patches near channels (small, beyond the bank zone)
    placed, attempts = 0, 0
    while placed < n_mud_patches and attempts < 100:
        attempts += 1
        try:
            channel_pt = geom['all_water'].representative_point()
        except Exception:
            break
        angle = rng.uniform(0, 2*np.pi)
        dist  = rng.uniform(3, 8)
        cx, cy = channel_pt.x + dist*np.cos(angle), channel_pt.y + dist*np.sin(angle)
        if not (xmin < cx < xmax and ymin < cy < ymax):
            continue
        r = rng.uniform(0.4, 1.2)
        n_verts = rng.integers(6, 10)
        pts = [(cx + r*rng.uniform(0.7, 1.3)*np.cos(t),
                cy + r*rng.uniform(0.7, 1.3)*np.sin(t))
               for t in np.linspace(0, 2*np.pi, n_verts, endpoint=False)]
        poly = Polygon(pts)
        if poly.intersects(geom['all_water']) or poly.intersects(geom['all_banks']):
            continue
        rows.append({'Class': CLASSES['other'], 'geometry': poly})
        placed += 1
    
    # 4) Wrack-like elongated polygons (debris deposited by tides)
    placed, attempts = 0, 0
    while placed < n_wrack and attempts < 50:
        attempts += 1
        cx = rng.uniform(xmin + 0.2*extent_x, xmax - 0.2*extent_x)
        cy = rng.uniform(ymin + 0.2*extent_y, ymax - 0.2*extent_y)
        angle  = rng.uniform(0, 2*np.pi)
        length = rng.uniform(2.0, 5.0)
        width  = rng.uniform(0.3, 0.6)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        local_pts = [(-length/2, -width/2), (length/2, -width/2),
                     (length/2,  width/2), (-length/2,  width/2)]
        pts = [(cx + p[0]*cos_a - p[1]*sin_a, cy + p[0]*sin_a + p[1]*cos_a)
               for p in local_pts]
        poly = Polygon(pts)
        if poly.intersects(geom['all_water']) or poly.intersects(geom['all_banks']):
            continue
        rows.append({'Class': CLASSES['other'], 'geometry': poly})
        placed += 1
    
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=CRS)
def fill_missing_classes_from_channel_mask(polygons_gdf, bounds, channel_mask_path, rng,
                                            min_per_class=3, bank_buffer_m=4.0,
                                            verbose=True):
    """Like fill_missing_classes but derives water and banks from a channel_mask raster.
    
    More reliable than the centerlines variant — uses the actual water footprint
    from the rasterized mask rather than reconstructing it via buffer assumptions.
    """
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union
    
    with rasterio.open(channel_mask_path) as src:
        mask = src.read(1).astype(np.uint8)
        transform = src.transform
    
    water_polys = [shape(g) for g, v in shapes(mask, mask=mask.astype(bool),
                                                transform=transform) if v == 1]
    if not water_polys:
        if verbose:
            print("  No water in mask — skipping bank-class fill.")
        return polygons_gdf
    
    all_water = unary_union(water_polys)
    all_banks = all_water.buffer(bank_buffer_m).difference(all_water)
    
    # For bank placement, use the boundary of the water polygon
    boundary = all_water.boundary
    channels = list(boundary.geoms) if hasattr(boundary, 'geoms') else [boundary]
    
    geom = {
        'main_channel': channels[0] if channels else None,
        'tributaries':  channels[1:],
        'all_water':    all_water,
        'all_banks':    all_banks,
    }
    
    return fill_missing_classes(polygons_gdf, geom, bounds, rng,
                                 min_per_class=min_per_class, verbose=verbose)
                                              
# ============================================================================
# Polygon label assignment
# ============================================================================
def split_channel_buffer_segments(channel_line, water_buf, bank_buf, n_segments, rng):
    """Divide a channel's bank zone into N segments along its length."""
    L = channel_line.length
    if L < 5.0:
        return []
    segments = []
    for i in range(n_segments):
        t0, t1 = i/n_segments, (i+1)/n_segments
        t0e = max(0, t0 - 0.01); t1e = min(1, t1 + 0.01)
        n_pts = max(4, int(L * (t1e - t0e) / 0.5))
        ts = np.linspace(t0e, t1e, n_pts)
        coords = [channel_line.interpolate(t, normalized=True).coords[0] for t in ts]
        sub_line = LineString(coords)
        seg = sub_line.buffer(bank_buf, cap_style=2).difference(sub_line.buffer(water_buf, cap_style=2))
        if seg.area > 0.1:
            segments.append(seg)
    return segments


def assign_bank_class_per_segment(n_segments, character, rng):
    """Probabilistically assign a bank class to each segment based on channel character."""
    weights_by_char = {
        'healthy': {'healthy_bank': 0.70, 'eroding_non_crab': 0.20, 'crab_edge': 0.07,
                    'crab_platform': 0.02, 'collapsed': 0.01},
        'eroding': {'healthy_bank': 0.20, 'eroding_non_crab': 0.55, 'crab_edge': 0.15,
                    'crab_platform': 0.05, 'collapsed': 0.05},
        'crab':    {'healthy_bank': 0.05, 'eroding_non_crab': 0.10, 'crab_edge': 0.35,
                    'crab_platform': 0.30, 'collapsed': 0.20},
        'mixed':   {'healthy_bank': 0.35, 'eroding_non_crab': 0.20, 'crab_edge': 0.20,
                    'crab_platform': 0.15, 'collapsed': 0.10},
    }
    weights = weights_by_char[character]
    classes = list(weights.keys())
    probs   = np.array(list(weights.values()))
    probs   = probs / probs.sum()
    return rng.choice(classes, size=n_segments, p=probs).tolist()


def assign_polygon_labels(geom, rng):
    """Generate a labeled polygon GeoDataFrame from the marsh geometry."""
    rows = []
    all_water = geom['all_water']

    def clean(seg):
        """Trim a segment polygon so it doesn't overlap channel water."""
        if seg.is_empty:
            return seg
        return seg.difference(all_water)

    main_segs = split_channel_buffer_segments(geom['main_channel'], 4.0, 6.5, 15, rng)
    main_classes = assign_bank_class_per_segment(len(main_segs), 'healthy', rng)
    for seg, c in zip(main_segs, main_classes):
        cleaned = clean(seg)
        if cleaned.area > 0.05:
            rows.append({'Class': CLASSES[c], 'geometry': cleaned})

    characters = ['healthy', 'mixed', 'crab', 'eroding', 'mixed', 'crab', 'healthy',
                  'mixed', 'eroding', 'crab']
    for i, trib in enumerate(geom['tributaries']):
        if trib.length < 8:
            n_seg = 3
        elif trib.length < 30:
            n_seg = 5
        else:
            n_seg = 8
        char = characters[i % len(characters)]
        # Use generous buffer so polygons cover the full bank zone, then trim water
        segs = split_channel_buffer_segments(trib, 0.4, 1.6, n_seg, rng)
        if not segs:
            continue
        seg_classes = assign_bank_class_per_segment(len(segs), char, rng)
        for seg, c in zip(segs, seg_classes):
            cleaned = clean(seg)
            if cleaned.area > 0.05:
                rows.append({'Class': CLASSES[c], 'geometry': cleaned})

    other_polys = []
    if not geom['all_trees'].is_empty:
        if isinstance(geom['all_trees'], MultiPolygon):
            other_polys.extend([p for p in geom['all_trees'].geoms if p.area > 1.0])
        else:
            other_polys.append(geom['all_trees'])
    if not geom['hummock'].is_empty:
        other_polys.append(geom['hummock'])
    if not geom['all_ponds'].is_empty:
        if isinstance(geom['all_ponds'], MultiPolygon):
            other_polys.extend([p for p in geom['all_ponds'].geoms if p.area > 0.5])
        else:
            other_polys.append(geom['all_ponds'])
    for p in other_polys:
        rows.append({'Class': CLASSES['other'], 'geometry': p})

    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=CRS)
    gdf = gdf[gdf.area > 0.05].reset_index(drop=True)
    return gdf


def ensure_all_classes_in_m1(polygons_m1, m1_bounds, geom, rng):
    """Make sure all 6 classes appear in M1 polygons.

    For missing bank classes (2-6): relabel the largest existing bank polygons
    (geometry stays natural — ring along channel — and spectral burn uses the
    new class).  For missing 'other' (1) or if not enough banks to relabel,
    add a small synthetic polygon in marsh interior.
    """
    if len(polygons_m1) == 0:
        present = set()
    else:
        present = set(int(c) for c in polygons_m1['Class'].tolist())
    missing = sorted({1, 2, 3, 4, 5, 6} - present)
    if not missing:
        return polygons_m1

    print(f"  M1 missing classes {missing} — relabeling/adding to cover all 6")

    # Relabel existing bank polygons for any missing bank classes
    bank_missing = [c for c in missing if c != 1]
    bank_polys   = polygons_m1[polygons_m1['Class'].isin([2, 3, 4, 5, 6])]
    sorted_idx   = bank_polys.geometry.area.sort_values(ascending=False).index.tolist()
    relabeled    = 0
    for cls, idx in zip(bank_missing, sorted_idx):
        polygons_m1.at[idx, 'Class'] = cls
        relabeled += 1

    # Anything still missing (no banks to relabel, or class 1) → add new polygons
    still_missing = bank_missing[relabeled:] + ([1] if 1 in missing else [])
    new_rows = []
    xmin, ymin, xmax, ymax = m1_bounds.bounds
    for cls in still_missing:
        for _ in range(40):
            cx = rng.uniform(xmin + 2, xmax - 2)
            cy = rng.uniform(ymin + 2, ymax - 2)
            r  = rng.uniform(0.5, 1.0)
            pts = [(cx + r*np.cos(t), cy + r*np.sin(t)) for t in np.linspace(0, 2*np.pi, 12)]
            poly = Polygon(pts)
            if poly.intersects(geom['all_water']):
                continue
            if polygons_m1.intersects(poly).any():
                continue
            new_rows.append({'Class': cls, 'geometry': poly})
            break

    if new_rows:
        new_gdf = gpd.GeoDataFrame(new_rows, crs=polygons_m1.crs)
        polygons_m1 = gpd.GeoDataFrame(
            pd.concat([polygons_m1, new_gdf], ignore_index=True),
            crs=polygons_m1.crs,
        )
    return polygons_m1

def fill_missing_classes(polygons_gdf, geom, bounds, rng,
                         min_per_class=3, verbose=True):
    """Add synthetic polygons for any classes that are missing or sparse.
    
    For bank classes (healthy_bank, eroding, crab_*, collapsed), places polygons
    along channels. For 'other', places them in marsh interior. Skips classes
    that already have >= min_per_class polygons.
    
    Args:
        polygons_gdf: existing labeled GeoDataFrame with 'Class' column
        geom: marsh geometry dict (needs 'all_water', 'all_banks',
              'main_channel', 'tributaries')
        bounds: spatial bounds (shapely box)
        rng: numpy random generator
        min_per_class: minimum polygons each class should have
        verbose: print what gets added
    
    Returns: augmented GeoDataFrame (original + new synthetic polygons).
    """
    from shapely.geometry import Polygon
    
    existing_counts = (polygons_gdf['Class'].value_counts().to_dict()
                       if len(polygons_gdf) > 0 else {})
    
    additions = []
    bank_class_names = {'healthy_bank', 'eroding_non_crab',
                        'crab_edge', 'crab_platform', 'collapsed'}
    
    for class_name, qgis_val in CLASSES.items():
        current  = existing_counts.get(qgis_val, 0)
        n_needed = max(0, min_per_class - current)
        if n_needed == 0:
            continue
        if verbose:
            print(f"  Class '{class_name}' (QGIS={qgis_val}): "
                  f"{current} existing, adding {n_needed}")
        
        if class_name == 'other':
            additions.extend(_make_interior_polygons(
                geom, bounds, rng, n_needed, qgis_val))
        elif class_name in bank_class_names:
            additions.extend(_make_bank_polygons(
                geom, rng, n_needed, qgis_val))
    
    if not additions:
        return polygons_gdf
    
    new_polys = gpd.GeoDataFrame(additions, geometry='geometry', crs=CRS)
    out = gpd.GeoDataFrame(
        pd.concat([polygons_gdf, new_polys], ignore_index=True),
        crs=polygons_gdf.crs or CRS,
    )
    return out


def _make_bank_polygons(geom, rng, n, class_qgis):
    """Create n small irregular polygons near channel banks."""
    from shapely.geometry import Polygon
    all_channels = [geom['main_channel']] + list(geom['tributaries'])
    rows, attempts = [], 0
    while len(rows) < n and attempts < 200:
        attempts += 1
        channel = all_channels[rng.integers(0, len(all_channels))]
        if channel.length < 2:
            continue
        # Random point along channel, then perpendicular offset for bank
        t = rng.uniform(0.1, 0.9)
        center = channel.interpolate(t, normalized=True)
        offset_dist  = rng.uniform(1.0, 3.0)
        offset_angle = rng.choice([-1, 1]) * (np.pi/2 + rng.uniform(-0.3, 0.3))
        cx = center.x + offset_dist * np.cos(offset_angle)
        cy = center.y + offset_dist * np.sin(offset_angle)
        # Small irregular polygon
        r = rng.uniform(0.6, 1.5)
        pts = [(cx + r*rng.uniform(0.7, 1.3)*np.cos(theta),
                cy + r*rng.uniform(0.7, 1.3)*np.sin(theta))
               for theta in np.linspace(0, 2*np.pi, 10, endpoint=False)]
        poly = Polygon(pts)
        if poly.intersects(geom['all_water']):
            continue
        rows.append({'Class': class_qgis, 'geometry': poly})
    return rows


def _make_interior_polygons(geom, bounds, rng, n, class_qgis):
    """Create n polygons in marsh interior (away from water and banks)."""
    from shapely.geometry import Polygon
    xmin, ymin, xmax, ymax = bounds.bounds
    extent_x, extent_y = xmax - xmin, ymax - ymin
    rows, attempts = [], 0
    while len(rows) < n and attempts < 200:
        attempts += 1
        cx = rng.uniform(xmin + 0.15*extent_x, xmax - 0.15*extent_x)
        cy = rng.uniform(ymin + 0.20*extent_y, ymax - 0.15*extent_y)
        r = rng.uniform(0.8, 2.0)
        pts = [(cx + r*rng.uniform(0.7, 1.3)*np.cos(theta),
                cy + r*rng.uniform(0.7, 1.3)*np.sin(theta))
               for theta in np.linspace(0, 2*np.pi, 10, endpoint=False)]
        poly = Polygon(pts)
        if poly.intersects(geom['all_water']) or poly.intersects(geom['all_banks']):
            continue
        rows.append({'Class': class_qgis, 'geometry': poly})
    return rows

# Add to marsh_utils.py — wraps fill_missing_classes with channel_centerlines input
def fill_missing_classes_from_centerlines(polygons_gdf, bounds, channel_centerlines, rng,
                                           min_per_class=3, verbose=True,
                                           water_buffer_m=2.0, bank_buffer_m=5.0):
    """Wrapper around fill_missing_classes for use when only channel_centerlines
    are available (e.g., in the M2 training notebook in synthetic mode).
    
    Reconstructs a minimal geom dict from the centerlines + buffer widths.
    """
    from shapely.ops import unary_union
    channels = list(channel_centerlines.geometry)
    if not channels:
        if verbose:
            print("  No channels in centerlines — skipping bank-class fill.")
        return polygons_gdf
    all_water = unary_union([c.buffer(water_buffer_m) for c in channels])
    all_banks = unary_union([c.buffer(bank_buffer_m) for c in channels]).difference(all_water)
    geom = {
        'main_channel': channels[0],
        'tributaries':  channels[1:],
        'all_water':    all_water,
        'all_banks':    all_banks,
    }
    return fill_missing_classes(polygons_gdf, geom, bounds, rng,
                                 min_per_class=min_per_class, verbose=verbose)
# ============================================================================
# Raster rendering
# ============================================================================
def make_class_index_raster(geom, polygons_gdf, bounds, resolution_m):
    """Create a uint8 raster with spectral class indices for every pixel."""
    xmin, ymin, xmax, ymax = bounds.bounds
    width  = int(round((xmax - xmin) / resolution_m))
    height = int(round((ymax - ymin) / resolution_m))
    transform = from_origin(xmin, ymax, resolution_m, resolution_m)
    print(f"  Raster: {width}×{height} pixels at {resolution_m*100:.1f}cm GSD "
          f"({width*height/1e6:.1f}M pixels)")

    idx_raster = np.full((height, width), SPECTRAL_IDX['marsh_platform'], dtype=np.uint8)

    def burn(shape, value):
        if shape is None or shape.is_empty:
            return
        rasterize([(shape, value)], out=idx_raster, transform=transform,
                  fill=0, default_value=value, all_touched=False)

    # Trees, hummock, ponds
    if not geom['all_trees'].is_empty:
        burn(geom['all_trees'], SPECTRAL_IDX['tree'])
    if not geom['hummock'].is_empty:
        burn(geom['hummock'], SPECTRAL_IDX['tree'])
    if not geom['all_ponds'].is_empty:
        burn(geom['all_ponds'], SPECTRAL_IDX['water'])

    # Bank polygons (matches their labels)
    class_to_spectral = {
        CLASSES['healthy_bank']:     SPECTRAL_IDX['healthy_bank'],
        CLASSES['eroding_non_crab']: SPECTRAL_IDX['eroding_non_crab'],
        CLASSES['crab_edge']:        SPECTRAL_IDX['crab_edge'],
        CLASSES['crab_platform']:    SPECTRAL_IDX['crab_platform'],
        CLASSES['collapsed']:        SPECTRAL_IDX['collapsed'],
    }
    bank_polys = polygons_gdf[polygons_gdf['Class'].isin(class_to_spectral.keys())]
    if len(bank_polys) > 0:
        shapes = [(row.geometry, class_to_spectral[row['Class']])
                  for _, row in bank_polys.iterrows()]
        rasterize(shapes, out=idx_raster, transform=transform,
                  fill=0, default_value=0, all_touched=False)

    # Water (highest priority - overlays everything)
    if not geom['all_water'].is_empty:
        rasterize([(geom['all_water'], SPECTRAL_IDX['water'])],
                  out=idx_raster, transform=transform,
                  fill=0, default_value=SPECTRAL_IDX['water'], all_touched=False)

    return idx_raster, transform


def composite_spectra(idx_raster, rng):
    """Convert class index raster into a 5-band reflectance raster (float32)."""
    height, width = idx_raster.shape
    ms = np.zeros((5, height, width), dtype=np.float32)
    for idx_value, spectrum in IDX_TO_SPECTRA.items():
        mask = (idx_raster == idx_value)
        if not mask.any():
            continue
        for band in range(5):
            ms[band][mask] = spectrum[band]

    for band in range(5):
        # Low-frequency texture
        noise_lf = rng.normal(0, 1.0, (height // 16 + 2, width // 16 + 2))
        noise_lf = ndimage.zoom(noise_lf, 16, order=1)[:height, :width]
        noise_lf = ndimage.gaussian_filter(noise_lf, sigma=8)
        noise_lf *= 0.015
        # Per-pixel noise
        noise_hf = rng.normal(0, 0.008, (height, width))
        ms[band] += noise_lf.astype(np.float32) + noise_hf.astype(np.float32)

    np.clip(ms, 0.0, 1.0, out=ms)
    return ms


def make_pan(ms, rng):
    """Synthesize a broadband panchromatic band as a weighted MS sum."""
    weights = np.array([0.20, 0.30, 0.30, 0.10, 0.10])
    pan = (weights[:, None, None] * ms).sum(axis=0)
    pan += rng.normal(0, 0.005, pan.shape).astype(np.float32)
    np.clip(pan, 0.0, 1.0, out=pan)
    return pan


def compute_index(num, den):
    """Normalized difference index with denominator safety."""
    out = np.full_like(num, np.nan, dtype=np.float32)
    denom = num + den
    valid = denom > 0.005
    out[valid] = (num[valid] - den[valid]) / denom[valid]
    return out


def write_geotiff(path, data, transform, dtype, count=None, nodata=None):
    """Write a tiled, compressed GeoTIFF.
    LZW + horizontal predictor for integers; DEFLATE + floating predictor for floats.
    """
    if data.ndim == 2:
        data = data[np.newaxis]
    count = count or data.shape[0]
    profile = {
        'driver': 'GTiff', 'height': data.shape[1], 'width': data.shape[2],
        'count': count, 'dtype': dtype, 'crs': CRS, 'transform': transform,
        'tiled': True, 'blockxsize': 256, 'blockysize': 256,
    }
    if dtype.startswith('uint') or dtype.startswith('int'):
        profile.update(compress='LZW', predictor=2)
    else:
        profile.update(compress='DEFLATE', predictor=3, zlevel=6)
    if nodata is not None:
        profile['nodata'] = nodata
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, 'w', **profile) as dst:
        if dtype.startswith('uint'):
            scaled = np.clip(data * 10000.0, 1, np.iinfo(dtype).max).astype(dtype)
            dst.write(scaled)
        else:
            dst.write(data.astype(dtype))


def downsample_block_mean(arr, factor):
    """Average over factor×factor blocks. Handles 2D and 3D arrays."""
    h, w = arr.shape[-2:]
    h2, w2 = h // factor, w // factor
    if arr.ndim == 2:
        arr_crop = arr[:h2*factor, :w2*factor]
        return arr_crop.reshape(h2, factor, w2, factor).mean(axis=(1, 3))
    arr_crop = arr[..., :h2*factor, :w2*factor]
    n = arr_crop.shape[0]
    return arr_crop.reshape(n, h2, factor, w2, factor).mean(axis=(2, 4))


def write_dem(path, bounds, dem_gsd_m=0.01, smoothing_sigma_m=0.3, geom=None):
    """Synthetic DEM at imagery resolution. Optional channel-cutting via geom."""
    xmin, ymin, xmax, ymax = bounds.bounds
    width  = int(round((xmax - xmin) / dem_gsd_m))
    height = int(round((ymax - ymin) / dem_gsd_m))
    transform = from_origin(xmin, ymax, dem_gsd_m, dem_gsd_m)

    # Base gradient: upland (right) high → channel side (left) low
    x_grad = np.linspace(0.5, 2.5, width)
    dem = np.tile(x_grad, (height, 1)).astype(np.float32)

    # Cut channels into the DEM if geometry was passed (makes slope/TPI meaningful)
    if geom is not None and 'all_water' in geom and not geom['all_water'].is_empty:
        from rasterio.features import rasterize
        water_mask = rasterize(
            [(geom['all_water'], 1)],
            out_shape=(height, width), transform=transform,
            fill=0, default_value=1, dtype='uint8',
        ).astype(np.float32)
        dem -= 0.4 * water_mask    # channels sit ~40 cm below marsh platform

    dem += np.random.default_rng(123).normal(0, 0.05, dem.shape).astype(np.float32)
    sigma_px = max(1.0, smoothing_sigma_m / dem_gsd_m)
    dem = ndimage.gaussian_filter(dem, sigma=sigma_px)
    write_geotiff(path, dem, transform, dtype='float32')


# ============================================================================
# End-to-end pipeline
# ============================================================================
def generate_dataset(output_dir, bounds, geom, polygons_gdf, dataset_name,
                     pan_gsd_m=None, ms_gsd_m=None, seed_offset=0):
    """Generate raw raster outputs (no derived indices — ensure_indices() does that).

    Writes (depending on which GSDs are provided):
      ms_5band.tif       at ms_gsd_m  (raw multispectral, B/G/R/RE/NIR stacked)
      pan.tif            at pan_gsd_m (single-band panchromatic)
      pansharp_5band.tif at pan_gsd_m (pansharpened multispectral)
      dem_5m.tif         at 0.5m GSD  (synthetic gradient DEM)
    """
    rng = np.random.default_rng(SEED + seed_offset)
    print(f"\nGenerating dataset: {dataset_name}")
    print(f"  Output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    gsds = [g for g in (pan_gsd_m, ms_gsd_m) if g is not None]
    if not gsds:
        raise ValueError("Must specify pan_gsd_m or ms_gsd_m")
    fine_gsd = min(gsds)

    idx_raster, transform = make_class_index_raster(geom, polygons_gdf, bounds, fine_gsd)

    print(f"  Compositing spectral bands at {fine_gsd*100:.1f}cm GSD...")
    ms_fine = composite_spectra(idx_raster, rng)

    # Pan + pansharp at finest GSD
    if pan_gsd_m is not None and abs(pan_gsd_m - fine_gsd) < 1e-9:
        pan = make_pan(ms_fine, rng)
        print(f"  Writing pan.tif ({pan_gsd_m*100:.1f}cm)...")
        write_geotiff(os.path.join(output_dir, 'pan.tif'),
                      pan, transform, dtype='uint16')
        print(f"  Writing pansharp_5band.tif ({pan_gsd_m*100:.1f}cm)...")
        write_geotiff(os.path.join(output_dir, 'pansharp_5band.tif'),
                      ms_fine, transform, dtype='uint16', count=5)
        with rasterio.open(os.path.join(output_dir, 'pansharp_5band.tif'), 'r+') as ds:
            ds.descriptions = ('Blue', 'Green', 'Red', 'RedEdge', 'NIR')

    # MS at ms_gsd_m (possibly coarser — block-mean downsample)
    if ms_gsd_m is not None:
        if ms_gsd_m > fine_gsd:
            factor = int(round(ms_gsd_m / fine_gsd))
            ms_coarse = downsample_block_mean(ms_fine, factor)
            ms_transform = from_origin(bounds.bounds[0], bounds.bounds[3],
                                       ms_gsd_m, ms_gsd_m)
        else:
            ms_coarse = ms_fine
            ms_transform = transform
        print(f"  Writing ms_5band.tif ({ms_gsd_m*100:.1f}cm)...")
        write_geotiff(os.path.join(output_dir, 'ms_5band.tif'),
                      ms_coarse, ms_transform, dtype='uint16', count=5)
        with rasterio.open(os.path.join(output_dir, 'ms_5band.tif'), 'r+') as ds:
            ds.descriptions = ('Blue', 'Green', 'Red', 'RedEdge', 'NIR')

    imagery_gsd = min(g for g in (pan_gsd_m, ms_gsd_m) if g is not None)
    print(f"  Writing dem_5m.tif at imagery GSD ({imagery_gsd*100:.1f}cm)...")
    write_dem(os.path.join(output_dir, 'dem_5m.tif'),
              bounds, dem_gsd_m=imagery_gsd, geom=geom)

def compute_confusion_matrix(model, loader, num_classes,
                              ignore_index=255, device='cuda'):
    """Pixel-level confusion matrix on a dataloader.
    Returns array cm[true, pred] of pixel counts."""
    import torch, numpy as np
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device)
            y = batch['mask'].to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            mask = (y != ignore_index)
            y_flat = y[mask].cpu().numpy()
            p_flat = pred[mask].cpu().numpy()
            np.add.at(cm, (y_flat, p_flat), 1)
    return cm


def display_confusion_matrix(cm, class_names, normalize='recall', ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(cm, dtype=np.float64)
    if normalize == 'recall':
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt, cbar_label = '.2f', 'P(pred | true) — recall'
    elif normalize == 'precision':
        cm = cm / cm.sum(axis=0, keepdims=True).clip(min=1)
        fmt, cbar_label = '.2f', 'P(true | pred) — precision'
    else:
        fmt, cbar_label = ',.0f', 'Pixel count'

    created_own_fig = ax is None
    if created_own_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', vmin=0,
                   vmax=cm.max() if normalize == 'none' else 1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=9)
    plt.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    if created_own_fig:
        plt.show()        # ← flush this figure before next loop iteration

def rule_argmax_abstain(probs, min_confidence=0.5, abstain_label=255):
    """Argmax, but mark pixel as 'abstain' if no class exceeds min_confidence."""
    import numpy as np
    pred = probs.argmax(axis=1)
    pred[probs.max(axis=1) < min_confidence] = abstain_label
    return pred


def rule_margin_abstain(probs, min_margin=0.15, abstain_label=255):
    """Argmax, but abstain if top1 - top2 < min_margin (top two classes too close)."""
    import numpy as np
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    pred = probs.argmax(axis=1)
    pred[margin < min_margin] = abstain_label
    return pred


def rule_entropy_abstain(probs, max_entropy=1.5, abstain_label=255):
    """Argmax, but abstain if Shannon entropy of probs > max_entropy.
    For 6 classes, max possible entropy = log(6) ≈ 1.79 (uniform distribution)."""
    import numpy as np
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    pred = probs.argmax(axis=1)
    pred[entropy > max_entropy] = abstain_label
    return pred

def rule_margin_with_tie_info(probs, min_margin=0.15, abstain_label=255):
    """Like rule_margin_abstain, but also returns which two classes were tied."""
    import numpy as np
    sorted_idx = np.argsort(probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    pred = probs.argmax(axis=1)
    tied = margin < min_margin
    pred[tied] = abstain_label
    # tied_pairs: (n_pixels, 2) — top-2 class indices for tied pixels
    top1 = sorted_idx[:, -1]
    top2 = sorted_idx[:, -2]
    return pred, np.stack([top1[tied], top2[tied]], axis=1)
