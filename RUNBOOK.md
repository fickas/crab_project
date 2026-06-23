# Marsh Labeler — Ingest & GT Round-Trip Runbook

How to move files between the **production notebook (Google Drive)** and the
**labeler app**, run a labeling round, and get verdicts back into ground truth.

## Mental model

- **Drive is the system of record.** The canonical `gt_labeled.gpkg` (PI-digitized
  polygons, plus any previously exported superpixel verdicts) lives on Drive and is
  what training reads.
- **The app is filesystem-agnostic.** It never reaches into Drive. You *bring files
  in* before a round and *take the updated GT out* after. Both transfers are manual
  copies you do — not something the app or Railway does.
- **Verdicts live in Postgres, not in files.** Labeling writes rows to the DB.
  `export_gt.py` is the only step that turns verdicts into GT polygons, and it is
  also where **reconciliation** (clipping `sp_label` against higher-authority
  `pi_digitized`/`field_survey`) happens.

So one labeling cycle is: **bring in → preflight → ingest → label → bring in GT →
export → take GT out.**

---

## 1. Bring files in (Drive → `data/synthetic/`)

The production notebook writes timestamped files. The app expects fixed canonical
names. Copy and rename:

| From Drive (production notebook)          | To `data/synthetic/`        | Role |
|-------------------------------------------|------------------------------|------|
| `abstain_review_<date>.gpkg`              | `abstain_review.gpkg`        | the labeling queue (required) |
| `pansharp_5band.tif`                      | `pansharp_5band.tif`         | ortho for chips (required) |
| `superpixels.tif`                         | `superpixels.tif`            | canonical labeling grid (required) |
| `abstain_reduced_<date>.tif`              | `abstain.tif`                | composition panel — use the **reduced** raster |
| `full_probs_<date>.tif`                   | `softmax.tif`                | probability bars + composition |
| `pan.tif`                                 | `pan.tif`                    | grayscale chip view |
| `gt_labeled.gpkg`                         | `gt_labeled.gpkg`            | canonical GT (exemplars + export target) |

Notes:

- **Use `abstain_reduced_<date>.tif`** for `abstain.tif`, not the raw abstain — the
  review polygons were built from the reduced raster, so the composition panel must
  read the same one or the confident/unsure counts won't match the questions.
- **One canonical `superpixels.tif`.** It must be the exact segmentation the review
  gpkg was built against — verdicts key on `superpixel_id` against this raster. If
  you re-segment, regenerate the review gpkg too.
- **GT serves two roles from one file.** Point the flight YAML's
  `labeled_polygons: data/synthetic/gt_labeled.gpkg` and `label_class_field: class_id`,
  and let `export_gt` write back into the same file. One canonical GT, no duplicate.

## 2. Preflight (verify everything is present)

```bash
python -m ingest.preflight --config flights/synthetic.yaml
```

Prints each referenced file as `OK` / `MISSING` and exits non-zero if anything is
missing. `ingest` also runs this automatically and refuses to start on a bad config,
so you can't silently ingest against a half-copied set.

## 3. Ingest containers + exemplars

```bash
python -m ingest.ingest_flight   --config flights/synthetic.yaml
python -m ingest.ingest_exemplars --config flights/synthetic.yaml --replace
```

`--replace` swaps the seed placeholder galleries for real rendered exemplars. Ingest
records the canonical raster paths onto the flight row, so `export_gt` can find them.

## 4. Label

Start the app, open the flight, label containers. Class picks settle immediately
(single coverage); skip / needs-split record but do not settle; out-of-scope retires.

## 5. Bring in the canonical GT, then export

`export_gt` **appends and reconciles into whatever GT file already exists**. So the
canonical `gt_labeled.gpkg` (with the PI polygons) must be present locally first —
otherwise you get an island file with only `sp_label` polys and no reconciliation,
and pushing that to Drive would clobber your PI polygons.

```bash
# gt_labeled.gpkg must already be in data/synthetic/ (step 1)
python export_gt.py --flight demo_synthetic
```

This vectorizes settled verdicts, tags them `source=sp_label`, clips them where they
overlap higher-authority polygons, drops redundant ones, logs class disagreements to
`gt_labeled_conflicts.gpkg`, and writes the merged layer. Re-running is safe — it
refreshes only this flight's `sp_label` rows.

## 6. Take the updated GT back out (local → Drive)

Copy `data/synthetic/gt_labeled.gpkg` (and `gt_labeled_conflicts.gpkg`) back up to
Drive, replacing the canonical copy. Training reads it from Drive as before.

---

## Railway (deployed) — what changes

When the app is deployed on Railway, the **verdicts live in Railway's Postgres**, not
on your machine. The file round-trip stays exactly the same and stays **on your
machine** — Railway never touches Drive or any gpkg.

- **Ingest** still runs from your laptop (you have the rasters), but pointed at the
  Railway database: set `DATABASE_URL` to the Railway connection string, then run
  `ingest_flight` as usual. Rasters are read locally; rows are written to Railway.
- **Export** likewise: bring `gt_labeled.gpkg` local, set `DATABASE_URL` to Railway,
  run `export_gt` — it reads verdicts from Railway, reconciles into the local gpkg,
  and you upload that to Drive.
- **The deployed app** only reads Postgres + chip PNGs. It does not need Drive, the
  rasters, or the gpkg.

**Open deployment question (settle when you actually deploy):** the app serves chip
PNGs, so the chips ingest renders must live somewhere the deployed app can read — a
Railway volume or an object store (S3/GCS), not your local disk. Until that's wired,
ingest-against-Railway will write DB rows whose chip URLs point at storage the
deployed app can't serve. This is a storage-backend decision, separate from the GT
workflow above.

---

## Quick reference

```bash
# preflight only
python -m ingest.preflight --config flights/synthetic.yaml

# full local cycle
python -m ingest.ingest_flight   --config flights/synthetic.yaml
python -m ingest.ingest_exemplars --config flights/synthetic.yaml --replace
# ... label in the app ...
python export_gt.py --flight demo_synthetic        # gt_labeled.gpkg must be present

# against Railway: same commands, with DATABASE_URL set to the Railway DB
```
