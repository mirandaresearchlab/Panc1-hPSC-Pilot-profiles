# 1. Quality control

Before we compare treatments, we check that the images and the single-cell profiles are reliable. A blurred image or a wrong cell outline gives wrong features, and no later step can repair it.

| Notebook | What it does |
|---|---|
| `01.dataset-qc-report.ipynb` | A report on the whole dataset: are the CellProfiler outlines good enough? It grades data completeness, lighting, cell outlines and features PASS, WARN or FAIL, in one scorecard. Cell counts on the plate, replicate agreement and the TGFβ1 positive control are tested on the QCed profiles in `3.EDA`. |
| `02.qc.ipynb` | Checks every image and every cell, and adds two pass/fail columns: `Metadata_image_qc` (blurred or over-exposed images) and `Metadata_profile_qc` (bad cell outlines, found with coSMicQC). It then joins the 3 plates into one QC profile. |

The two notebooks are independent and can run in either order.

## Main findings

- The CellProfiler settings are good enough: nuclei are not cut into pieces, and cell outlines are mostly good (notebook 01).
- **Image QC:** 193 of 10,257 fields of view fail (blurred or over-exposed). A field of view is one position in a well, imaged in 5 channels (5 image files). They hold 1.5% of cells. Most are over-exposed Mito images of CL1 at 24h.
- **Profile QC:** 19.7–23.4% of CL1 cells and 10.6–11.6% of CL2 cells fail, mostly merged nuclei. CL1 uses all 3 coSMicQC conditions; CL2 leaves out condition 3, because it flags healthy cells in CL2.
- 73.1–88.9% of cells pass both checks.

## Input

- Single-cell profiles: `../0.prepare-profiles/results/single-cell-profiles/`
- Plate layout: `../../data/metadata/` (`platemap.csv`, `ligands.csv`)
- Images and masks, through `../../data/sc-profiles/`, for the example pictures.

## Results

All results are in `results/`.

| Path | Made by | Content |
|---|---|---|
| `qc_Panc1_hPSC_sc-profiles.parquet` | 02 | **The QC profile:** all 3 plates in one table, one row per cell, with all features. It adds the cell line, time point, treatment and both QC columns with their reasons. Keep the cells where `Metadata_image_qc` and `Metadata_profile_qc` are both `pass`. This is the input for preprocessing. |
| `quality-scorecard.csv` | 01 | The value and grade of each dataset check, per plate and cell line. |
| `cache/` | 01 | Saved results of slow steps (a sample of 100 cells per well, and the segmentation measurements on 72 control images). Delete this folder to compute them again. |

## Run

```bash
sbatch _run.sbatch                     # both notebooks, in order
sbatch _run.sbatch 02.qc.ipynb         # one notebook
```

Notebook 01 needs about 200 GB of memory the first time it runs, so use Slurm, not the login node. Notebook 02 takes about 3 minutes. Logs go to `logs/`.
