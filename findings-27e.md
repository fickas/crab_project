# Assessment: 27e (Unvegetated Bank / Crab Burrowing) Polygon Analysis

## Summary

After reviewing the 27e-tagged polygons against available orthomosaic flights, I recommend pausing this line of analysis until matched season/tide imagery is available. The mismatch between field collection dates and historical flight conditions, combined with the small polygon count, makes a reliable classification model unlikely from current data.

## Background

The Vegetation Mapping project is an ambitious undertaking that seeks to accurately measure 33 distinct features of a salt marsh, along with an optional 1-20 additional attributes for each feature. For Wellfleet Marsh, ground observations were conducted in 2019, 2023, and 2025. Flights were conducted in 2019, 2020, 2021, 2022, and 2023.

Looking at the 33 features/classes and attributes, I identified one class and one attribute that were relevant to measuring bank degradation caused by crab activity: class 27 - unvegetated banks; attribute e - perforated peat with >25 crab burrows per square meter.
I identified 15 polygons tagged with 27e across all collection years. These came from two field collection periods:

- 11 polygons from fall 2025
- 4 polygons from fall 2019

All 15 polygons were collected in the fall, and I'm assuming at low tide, given that they target bank features.

The closest available orthomosaic from a drone flight, in terms of season and tide state, is **fall 2021 at low tide**. Later flights exist (2022, 2023), but none align with both season and tide.

## What I Did

I extracted each polygon's footprint on the 2021 ortho and generated per-polygon chips showing the bank context. Chips are available in the shape_chips folder in this repository.

A quantitative note: Across all 15 polygons, there are roughly 27,000 pixels of 27e-labeled area on the ortho. However, it is the actual number of polygons (15) that is critical for machine learning. To avoid spatial autocorrelation, the datasets typically used in machine learning - training, validation, testing - come from the polygons, not individual pixels. In essence, the 15 polygons must be split up among these three datasets.

## Findings and Concerns

**Temporal mismatch.** The 2025 polygons were matched against imagery from 4 years earlier. Visually inspecting those polygons on the 2021 ortho, it's not obvious that 27e-like features (unvegetated banks with crab burrowing signatures) were present at the same locations in 2021. Bank morphology and vegetation cover shift over that timespan, so we can't assume the 2025 label applies to the older imagery.

**Insufficient training data.** Fifteen polygons split among three datasets are not enough to do meaningful analysis using machine learning.

## Recommendation

I suggest holding off on modeling and analysis efforts until Ryan's upcoming 2026 flights produce labeled data **aligned** with the ortho (and, ideally, a matching DEM). This new data will specifically target crab activity detrimental to marsh health. Our goal is to first produce sufficient polygons to feed into a machine learning model, and then train the model on that data. As proposed, our planning allows us to avoid both temporal mismatch and insufficient training data.

Further work against the existing veg-mapping data will not pay off, in my opinion.
