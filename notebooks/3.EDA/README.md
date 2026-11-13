# 3. Screen assessment (EDA)

This module looks at the preprocessed profiles to judge the pilot screen: what drives the differences between wells, how cell number plays a role, and how reproducible and strong each treatment is. It uses the QCed profiles of `2.preprocessing`, and leaves out the 4 wells that fail well QC (`Metadata_well_qc`, all on the 24h CL1 plate with fewer than 100 cells).

| Notebook | Question |
|---|---|
| `01.batch-effects-diagnosis.ipynb` | What drives the differences between profiles: biology (cell line, time point, treatment) or technical effects (cell density, plate position)? UMAPs of single cells and wells, and a test on the DMSO wells. |
| `02.explore-metadata.ipynb` | How the screen was set up and run: the plate layout of controls, ligands and compounds with their doses (checked against the layout files), the imaging settings, imaging order and focus of each plate, and the cell counts: plate maps, DMSO spread, treatment effects on cell number, and how much cell number drives the features. |
| `03.replicate-health-diagnosis.ipynb` | Do replicate wells agree for each compound and dose, which treatments differ from DMSO (percent replicating, mAP), how effects depend on dose and time, and how compounds relate to each other. |

## Main findings

- Cell line and time point drive the largest differences. Treatment effects are small, and grow with time: 32% of the 126 treatment tests differ from DMSO (mAP), most of them at 24h.
- CBK600005 changes the cells at every time point and dose, already at the lowest dose; CBK200566 strongly in CL2 at 24h. CBK041498 and CBK175924 act mainly at 2 mM. TGFβ1 (positive control) is active at 24h in both cell lines. IL6 and CBK600008 show no effect.
- Cell number matters: CBK600005, CBK200566 (CL2) and probably TGFβ1 (CL1) lower it at 24h, and 23–59% of features follow cell number in DMSO wells.
- Technical effects: uneven cell density on the 24h plate (CL1), and a shifted profile in every other row of columns 1–6 in CL1 at 1h and 4h, which points to liquid handling.
- Design and imaging: one layout on all 3 plates, with CL2 repeating CL1; the layout files agree with the profiles. The 24h plate was imaged with longer ER and AGP exposures and a different Mito focus offset. Imaging order and focus do not explain the plate patterns.

## Input

- `../2.preprocessing/results/qced_norm_fs_Panc1_hPSC_sc-profiles.parquet` and `qced_agg_norm_fs_Panc1_hPSC_well-profiles.parquet`
- `../2.preprocessing/results/agg_norm_fs_Panc1_hPSC_well-profiles.parquet` (all cells, for cell counts)
- A few metadata columns of `../1.quality-control/results/qc_Panc1_hPSC_sc-profiles.parquet` (QC pass rates per well)
- The layout files `../../data/metadata/platemap.csv` and `ligands.csv`, and the imaging records of each plate (`parameters.json`, `coords.jsonl` in `../../data/sc-profiles/PALS-screen-profiles-v2/<plate>/input/<plate>/`)

## Results

All results are in `results/`:

| Path | Content |
|---|---|
| `figures/<notebook>/` | Every figure of each notebook, as PNG (the report uses them) |
| `tables/<notebook>/` | Summary tables as CSV, for example `tables/03.replicate-health-diagnosis/treatment_scores.csv` (one row per treatment, plate and cell line, with its 3 wells), `replicating_doses.csv` (which doses replicate) and `tables/02.explore-metadata/treatment_cell-count-and-activity.csv`, `design_treatments-and-doses.csv` and `imaging_settings.csv` |
| `umap_qced_sc-sample.parquet`, `umap_qced_wells.parquet` | UMAP coordinates with metadata (notebook 01) |

Notebook 03 also saves `figures/03.replicate-health-diagnosis/02b_map-negative-vs-positive-reference.png`: two panels (CL1 and CL2) plotting activity mAP against DMSO on x and against all three TGFβ1 doses on y, with colour for agent, shape for time and size for dose. The paired scores, permutation p/q-values and reference counts are in `tables/03.replicate-health-diagnosis/map_control_references.csv`. Higher scores mean distinguishability from the reference, not similarity or anticancer activity; the two axes have different chance baselines because their reference sets differ in size.

## Run

Run one notebook at a time:

```bash
sbatch _run.sbatch 01.batch-effects-diagnosis.ipynb
```

Notebook 01 loads 2 million cells (about 30 GB of memory, 4 minutes). Notebooks 02 and 03 use the well profiles only and take 1–3 minutes. `sbatch _run.sbatch` without a notebook name runs all three in order. Logs go to `logs/`.
