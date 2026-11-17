# 0. Prepare profiles

CellProfiler measures about 4,000 features for every cell in the three plates (1h, 4h and 24h). This module turns its output into tables that the later modules can read quickly.

| Notebook | What it does |
|---|---|
| `01.convert-profiles-to-parquet.ipynb` | Converts each plate's single-cell table from compressed CSV to parquet, and checks that no cell or column was lost. |

## Input

- CellProfiler output: `../../data/sc-profiles/PALS-screen-profiles-v2/<plate>/`. This is a symlink to the CellProfiler results, which are outside this repository.

## Results

All results are in `results/`.

| Path | Content |
|---|---|
| `single-cell-profiles/<plate>.parquet` | One row per cell, one column per feature. The input of module 1. |
| `single-cell-profiles/<plate>_missing-values.csv` | The number of missing values in each column. |

## Run

```bash
sbatch _run.sbatch
```

The job runs on Slurm (about 40 minutes), because the tables are large. Logs go to `logs/`. Plates that already have a parquet file are skipped.
