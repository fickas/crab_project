# Assessment: 27e (Unvegetated Bank / Crab Burrowing) Polygon Analysis

## Summary

After reviewing the 27e-tagged polygons against available orthomosaic flights, I recommend pausing this line of analysis until matched season/tide imagery is available. The mismatch between field collection dates and historical flight conditions, combined with the small pixel count, makes a reliable classification model unlikely from current data.

## Background

I identified 15 polygons tagged with Sub&Att 27e — unvegetated banks (27) with Perforated peat, i.e., > 25 crab burrows per square meter (e). These came from two field collection periods:

- 11 polygons from fall 2025
- 4 polygons from fall 2019

Both sets were collected in the fall, and I'm assuming at low tide, given that they target bank features.

The closest available orthomosaic in terms of season and tide state is **fall 2021 at low tide**. Later flights exist (2022, 2023), but none align with both season and tide.

## What I Did

I extracted each polygon's footprint on the 2021 ortho and generated per-polygon chips showing the bank context. Chips are available in the shape_chips folder in this repository.

Two quantitative notes:

- Across all 15 polygons, there are roughly **15,000 pixels** of 27e-labeled area on the ortho.
- This would be the entire training signal available if we used this data for a pixel-based classifier (e.g., random forest).

## Findings and Concerns

**Temporal mismatch.** The 2025 polygons were matched against imagery from 4 years earlier. Visually inspecting those polygons on the 2021 ortho, it's not obvious that 27e-like features (unvegetated banks with crab burrowing signatures) were present at the same locations in 2021. Bank morphology and vegetation cover shift over that timespan, so we can't assume the label applies to the older imagery.

**Insufficient training data.** Fifteen thousand pixels is a small training budget for a random-forest-style classifier, especially for a class as visually variable as bank + burrow texture. Class imbalance against the rest of the scene would be extreme.

**DEM isn't a workaround.** I considered bringing in DEM data as an alternative feature source, but DEM acquisition dates largely mirror the RGB flights, so the same temporal mismatch applies.

## Recommendation

I suggest holding off on modeling and aqnalysis effort until Ryan's upcoming 2026 flights produce labeled data **aligned** with the ortho (and ideally a matching DEM). This new data will specifically target crab activity detrimental to marsh health.

Further work against the existing veg-mapping data will not pay off, in my opinion.
