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
