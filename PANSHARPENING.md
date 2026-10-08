# Pansharpening: method

This document describes how the pansharpened 5-band imagery (Blue, Green, Red, Red edge, NIR at 0.93 cm) is produced from the RedEdge-P panchromatic (pan) and multispectral (MS) orthomosaics, and what the resulting data does and does not contain.

The code is in `pansharpen_utils.py` (`build_pansharpen_vrt`, `materialize_pansharpened`) and is run step by step in `pansharpen_pipeline.ipynb`. Co-registration, which must happen first, is in `coreg_diagnostics.py` and `coreg_fit.py`.

## Overview

The pansharpened product is made with GDAL's **weighted Brovey** algorithm, set up as a virtual raster (a `VRTPansharpenedDataset`). `build_pansharpen_vrt` writes a small XML file describing the computation rather than computing anything itself. Pixels are produced only when the virtual raster is read, which here happens once, when `materialize_pansharpened` writes it out as a Cloud-Optimized GeoTIFF (COG).

```python
def build_pansharpen_vrt(pan_path, ms_path, vrt_path,
                         weights=None, resampling="Cubic", ms_bands=(1, 2, 3, 4, 5),
                         nodata=None, num_threads="ALL_CPUS"):
```

## The computation

For each pixel on the pan grid (0.93 cm), with the five MS bands $MS_i$ brought onto the same grid:

1. A pseudo-panchromatic value is formed as a weighted sum of the MS bands:
   $$P' = \sum_i w_i \, MS_i$$
2. The ratio of the real pan to the pseudo-pan is taken:
   $$r = \frac{PAN}{P'}$$
3. Each output band is the MS band scaled by that ratio:
   $$out_i = MS_i \cdot r$$

The ratio $r$ carries the pan's fine spatial detail, meaning everything between the MS resolution (1.76 cm) and the pan resolution (0.93 cm).

### Band ratios and indices are exactly preserved

Every band at a pixel is multiplied by the same $r$, so the ratios between bands are unchanged, and any normalized-difference index equals its value in the MS:

$$NDVI_{out} = \frac{r \cdot NIR - r \cdot Red}{r \cdot NIR + r \cdot Red} = NDVI_{MS}$$

This was verified on the outputs: 99% of pixels differ by less than $2 \times 10^{-4}$ in NDVI, which is the rounding from writing 16-bit integers.

**Consequence for modeling:** NDVI and NDRE computed from the pansharpened bands carry 1.76 cm information. The added 0.93 cm detail is in the band intensities and in the pan itself.

## Parameters

| Parameter | Meaning | Value used |
|---|---|---|
| `weights` | Weights $w_i$ for the pseudo-pan. They affect the absolute values of the output bands, but not band ratios or indices. Matching them to the pan sensor's spectral response would improve absolute radiometric fidelity. | Equal (0.2 each) |
| `resampling` | How the MS is brought onto the pan grid. Effectively unused here: the MS has already been corrected and resampled once (cubic) directly onto the pan grid, so this step does no further resampling. | `Cubic` |
| `ms_bands` | Which bands of the MS input are used, in output order. The corrected MS holds only the five spectral bands. The delivered MS file has six bands, including a downscaled pan, which is excluded upstream. | `(1, 2, 3, 4, 5)` |
| `nodata` | A pixel that is no-data in the pan or in any MS band is no-data in the output, so footprint edges and gaps come out empty rather than as invented values. | `65535` |
| `num_threads` | Parallel computation. | All cores |

The output extent is the intersection of the pan and MS extents.

## Output

- **5-band product:** one 16-bit COG per survey block (Blue, Green, Red, Red edge, NIR), with lossless DEFLATE compression, predictor 2, 512×512 tiles, internal overviews, and band names stored with the file.
  - NDVI = (band 5 − band 3) / (band 5 + band 3)
  - NDRE = (band 5 − band 4) / (band 5 + band 4)
- **Labeling product:** an 8-bit true-color COG (bands 3, 2, 1) with one shared contrast stretch across blocks, for labeling in QGIS.
- **Pan display product:** a COG of the pan for each block, with overviews, for viewing in QGIS.

## Why co-registration comes first

Brovey assumes the pan and MS are perfectly aligned. Any misalignment shows up as color fringing along edges, because $r$ then carries edges that sit in the wrong place relative to the MS colors.

The pan and MS orthos for the 2026-07-24 Wellfleet flight came from separate photogrammetric runs and differed by up to 37 cm: a scale difference of 0.07–0.1% and, in one block, a slight bending. Before pansharpening:

1. Offsets were measured by phase correlation at several dozen test points in each survey block.
2. Shift, affine and quadratic corrections were fitted per block, and the model was chosen by leave-one-out error.
3. The MS was warped, in a single resampling, directly onto the pan grid.

| Block | Model | Leave-one-out error | Residual after correction (independent test points) |
|---|---|---|---|
| 1 (NW) | quadratic | 2.4 cm | 0–4 cm |
| 2 (SE) | quadratic | 1.5 cm | ≤ 2 cm |

## Limitations

- **Modest sharpening.** The pan/MS resolution ratio is only about 1.9, so the visible gain is modest. It's most apparent at stem and burrow scale.
- **Absolute radiometry.** Brovey alters absolute band values wherever the weighted MS sum doesn't match the pan's spectral response, though band ratios are always preserved.
- **Clipping.** Output values are clipped to the 16-bit range. In practice this matters only on rare bright pixels such as specular glint.
- **Footprint edges.** The outermost pixels along footprint boundaries can carry resampling artifacts. Shrink the valid-data mask by a few pixels when cutting training and inference tiles.
