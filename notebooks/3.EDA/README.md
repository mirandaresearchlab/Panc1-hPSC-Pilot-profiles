# 3. Screen assessment (EDA)

This module looks at the preprocessed profiles to judge the pilot screen: what drives the differences between wells, which technical effects (batch effects) are in the data, and how reproducible, strong and toxic each treatment is. It uses the QCed profiles of `2.preprocessing`, and leaves out the 4 wells that fail well QC (`Metadata_well_qc`, all on the 24h CL1 plate with fewer than 100 cells).

Each notebook answers one question, so a result is shown in one place only:

| Notebook | Question |
|---|---|
| `01.batch-effects-diagnosis.ipynb` | What drives the differences between profiles: biology (cell line, time point, treatment) or technical effects? UMAPs of single cells and all wells (biology), then **batch effects on the DMSO wells only**: a UMAP of the DMSO wells, a plate map of how far each DMSO well is from the typical DMSO well, tests of each technical factor (plate area, cell number, plate edge, row = imaging order), a test for gradients across the plate, and the share of features that follow cell number. |
| `02.explore-metadata.ipynb` | How the screen was set up and run: the plate layout (checked against the layout files), the imaging settings, imaging order and focus of each plate, and the cell counts on the plates and in the DMSO wells. |
| `03.replicate-health-diagnosis.ipynb` | Do replicate wells agree, which treatments differ from DMSO (percent replicating, mAP against DMSO only), which replicate wells are outliers, how strong each treatment is across dose and time, and which treatments look alike. |
| `04.identify-cytotoxic-perts.ipynb` | Which treatments kill cells or stop their growth (cell count per well against DMSO; cytotoxic when the 3 wells are at least 3 robust SDs below DMSO), and do their profile changes look like DMSO wells with fewer cells? |

## Main findings

- **Biology:** cell line and time point drive the largest differences between single cells. Treatment effects are small and grow with time: 40 of 126 treatment tests differ from DMSO (mAP), most of them at 24h.
- **Batch effect 1, plate area:** in CL1 at 1h, all 33 unusual DMSO wells (of 127) lie in the odd rows of columns 1–6; they are 3.7 times further from the typical DMSO well and form their own island in the well UMAP. The pattern is weaker at 4h (11 wells) and points to liquid handling.
- **Batch effect 2, cell number:** DMSO wells with an unusual cell number differ more on every plate (rho 0.30–0.66), and 23–59% of features follow cell number. The 24h plate has low-count areas (CL1 columns 3–4 and rows M–N; CL2 rows L–N), most likely from seeding or handling.
- **Batch effect 3, row gradient:** within each plate, a principal component of the DMSO wells follows the plate row (|rho| 0.58–0.74, 13–19% of the DMSO variation in CL2; smaller in CL1). The plate was imaged row by row, so position and imaging drift cannot be separated.
- **Weak:** distance to the plate edge and the column on their own; the plates as a whole do not differ, because each is normalized to its own DMSO cells. Batch effects are assessed on the DMSO wells only, because treated wells can differ for biological reasons.
- **Confounded by design:** time point with the imaging settings (longer ER and AGP exposures on the 24h plate), and cell line with plate half. All comparisons are made within one plate and cell line.
- **Treatments:** CBK600005 changes the cells at every time point and dose; CBK200566 strongly in CL2 at 24h. CBK041498 and CBK175924 act mainly at 2 mM. TGFβ1 (positive control) is active at 24h in both cell lines. IL6 and CBK600008 show no effect.
- **Cytotoxic** (≥ 3 robust SDs below DMSO): CBK600005 and CBK200566 in CL2 at 24h, at all doses (50–71% of DMSO cells). Nothing at 1h or 4h; in CL1 at 24h the DMSO wells vary too much to call the cell loss of CBK600005 and TGFβ1. The profile changes of the treatments that lose cells look partly like DMSO wells with fewer cells; CBK600005 at 1h and 4h does not.

## Input

- `../2.preprocessing/results/qced_norm_fs_Panc1_hPSC_sc-profiles.parquet` and `qced_agg_norm_fs_Panc1_hPSC_well-profiles.parquet`
- `../2.preprocessing/results/agg_norm_fs_Panc1_hPSC_well-profiles.parquet` (all cells, for cell counts)
- A few metadata columns of `../2.preprocessing/results/norm_fs_Panc1_hPSC_sc-profiles.parquet` (all cells, for the cell counts of notebook 04)
- A few metadata columns of `../1.quality-control/results/qc_Panc1_hPSC_sc-profiles.parquet` (QC pass rates per well)
- The layout files `../../data/metadata/platemap.csv` and `ligands.csv`, and the imaging records of each plate (`parameters.json`, `coords.jsonl` in `../../data/sc-profiles/PALS-screen-profiles-v2/<plate>/input/<plate>/`)

## Results

All results are in `results/`:

| Path | Content |
|---|---|
| `figures/<notebook>/` | Every figure of each notebook, as PNG (300 dpi) and PDF, drawn at print size (full page width 7.2 in, text 6–9 pt) |
| `tables/01.batch-effects-diagnosis/` | `well_distance-and-technical-factors.csv` (each well's distance from the typical DMSO well, cell number, plate area, distance to the edge and imaging time), `dmso_technical-factors.csv` (the tests on DMSO wells), `dmso_unusual-wells.csv`, `dmso_feature-vs-cell-number_spearman.csv` |
| `tables/02.explore-metadata/` | `design_treatments-and-doses.csv`, `imaging_settings.csv`, `qc_pass_summary.csv`, `dmso_cell-count_summary.csv`, `dmso_edge-vs-inner.csv`, `well-qc_failing-wells.csv` |
| `tables/03.replicate-health-diagnosis/` | `treatment_scores.csv` (one row per treatment, plate and cell line: replicate correlation, mAP, strength, flags), `replicating_doses.csv`, `outlier_replicates.csv`, `dose_response.csv` |
| `tables/04.identify-cytotoxic-perts/` | `cytotoxicity_calls.csv`, `dmso_threshold.csv`, `cell-loss-and-density-likeness.csv` |
| `umap_qced_sc-sample.parquet`, `umap_qced_wells.parquet`, `umap_qced_dmso-wells.parquet` | UMAP coordinates with metadata (notebook 01) |

## Run

Run the notebooks in order: notebook 04 reads `treatment_scores.csv` of notebook 03.

```bash
sbatch _run.sbatch                                   # all four, in order
sbatch _run.sbatch 01.batch-effects-diagnosis.ipynb  # one notebook
```

Notebook 01 loads 2 million cells (about 30 GB of memory, 5 minutes). Notebooks 02–04 use the well profiles and metadata columns only and take 1–3 minutes each. Logs go to `logs/`.
