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

