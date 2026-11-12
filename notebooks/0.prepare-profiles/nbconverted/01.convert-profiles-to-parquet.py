#!/usr/bin/env python
# coding: utf-8

# # Convert single-cell CellProfiler profiles to parquet
# 
# `headless-cp-wf` already joins the Image, Cells, Cytoplasm and Nuclei tables into one row per cell,
# so CytoTable is not needed here. This notebook only changes the file format:
# one gzip CSV per plate -> one parquet file per plate.
# 
# Memory: DuckDB streams each CSV straight into parquet (`COPY ... TO`), so the table is never held in RAM.
# Memory use is set by `memory_limit` and `row_group_size`, not by the file size.
# 
# Missing values: every missing-value token (`nan`, `inf`, empty) becomes a parquet NULL.

# In[1]:


import csv
import gzip
import json
import pathlib
import shutil

import duckdb
import polars as pl


# ## Settings
# 
# Match `memory_limit` and `n_threads` to the Berzelius allocation.
# Reading a gzip file is single-threaded, so more threads mostly help the parquet writer.

# In[2]:


# headless-cp-wf results: one folder per plate, each with a DONE file
cellprofiler_dir = pathlib.Path("../../data/sc-profiles/PALS-screen-profiles-v2").resolve(strict=True)

# one parquet file per plate is written here
out_dir = pathlib.Path("results/single-cell-profiles").resolve()
out_dir.mkdir(parents=True, exist_ok=True)

# DuckDB resources; data that does not fit in memory_limit spills to tmp_dir
memory_limit = "64GB"
n_threads = 8
tmp_dir = pathlib.Path("tmp").resolve()

# rows per parquet row group; ~4000 columns x 10k rows x 8 bytes = ~320 MB buffered per writer
row_group_size = 10_000

# missing-value tokens in the CSV; all are stored as NULL
# "nan": CellProfiler NaN; "inf"/"-inf": CellProfiler infinities (not usable as features);
# "": Cytoplasm_* of cells without a cytoplasm (left join in headless-cp-wf MERGE)
null_tokens = ["", "nan", "NaN", "-nan", "inf", "-inf", "Inf", "-Inf", "Infinity", "-Infinity"]

# overwrite parquet files that already exist
overwrite = False


# ## Find the plate tables
# 
# Each plate's `DONE` file holds the path of its workflow run. The run has the per-cell table
# (`cp_features/all_plates.csv.gz`) and the MERGE report with the expected number of cells.

# In[3]:


plates = {}
for done_file in sorted(cellprofiler_dir.glob("*/DONE")):
    plate = done_file.parent.name
    run_dir = pathlib.Path(done_file.read_text().strip())
    plates[plate] = {
        "csv": (run_dir / "cp_features" / "all_plates.csv.gz").resolve(strict=True),
        "merge_report": (run_dir / "logs" / plate / "merge.json").resolve(strict=True),
    }

if not plates:
    raise FileNotFoundError(f"No plates with a DONE file in {cellprofiler_dir}")

for plate, paths in plates.items():
    print(f"{plate}: {paths['csv'].stat().st_size / 2**30:.1f} GiB")


# ## Column types
# 
# Types are set explicitly instead of inferred. Type inference only reads a sample of rows, and
# with ~4000 columns a column that is empty in the sample can be typed wrong.
# Explicit types also make the read fail loudly on any value that is not a number or a null token,
# so no unexpected `NaN` can reach the parquet file.
# 
# - text: file names, paths, URLs, MD5 digests, and metadata labels (`Metadata_Column` is `"01"`, so it stays text)
# - integer: image and object indices, site and tile position
# - double: every measurement

# In[4]:


text_prefixes = ("Image_FileName_", "Image_PathName_", "Image_URL_", "Image_MD5Digest_")
integer_metadata = {
    "Metadata_Site",
    "Metadata_X",
    "Metadata_Y",
    "Metadata_PlateImageNumber",
    "Metadata_ImageNumber",
    "Metadata_ObjectNumber",
}


def read_header(csv_path):
    """Read only the header line of a gzip CSV."""
    with gzip.open(csv_path, "rt", newline="") as stream:
        return next(csv.reader(stream))


def column_type(name):
    """DuckDB type for one column of the per-cell table."""
    if name in integer_metadata:
        return "BIGINT"
    if name.startswith("Metadata_") or name.startswith(text_prefixes):
        return "VARCHAR"
    return "DOUBLE"


# all plates must share one schema so they can be read together later
headers = {plate: read_header(paths["csv"]) for plate, paths in plates.items()}
header = next(iter(headers.values()))
for plate, plate_header in headers.items():
    if plate_header != header:
        raise ValueError(f"{plate}: columns differ from the first plate")

schema = {name: column_type(name) for name in header}
print(pl.Series(list(schema.values())).value_counts(sort=True))


# ## Convert
# 
# One streaming pass per plate. The parquet file is first written as `.partial` and renamed when
# complete, so an interrupted run never leaves a file that looks finished.

# In[5]:


def sql_list(values):
    """Format Python strings as a DuckDB list literal."""
    return "[" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + "]"


def convert_plate(csv_path, parquet_path):
    """Stream one gzip CSV into a zstd parquet file with explicit types and NULLs."""
    columns = "{" + ", ".join(f"'{name}': '{dtype}'" for name, dtype in schema.items()) + "}"
    partial = parquet_path.with_name(parquet_path.name + ".partial")
    partial.unlink(missing_ok=True)

    with duckdb.connect() as con:
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET threads = {n_threads}")
        con.execute(f"SET temp_directory = '{tmp_dir}'")
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_csv(
                    '{csv_path}',
                    compression = 'gzip',
                    header = true,
                    auto_detect = false,
                    columns = {columns},
                    nullstr = {sql_list(null_tokens)}
                )
            )
            TO '{partial}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {row_group_size})
        """)
    partial.rename(parquet_path)


# In[6]:


out_dir.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

for plate, paths in plates.items():
    parquet_path = out_dir / f"{plate}.parquet"
    if parquet_path.exists() and not overwrite:
        print(f"{plate}: exists, skipping")
        continue
    print(f"{plate}: converting")
    convert_plate(paths["csv"], parquet_path)
    print(f"{plate}: wrote {parquet_path.stat().st_size / 2**30:.1f} GiB")

shutil.rmtree(tmp_dir, ignore_errors=True)


# ## Validate
# 
# - row count equals the number of cells in the MERGE report
# - the column names and order match the CSV header
# - no `NaN` or infinite value is left in any measurement column (full scan of the parquet file)
# 
# The null counts per column come from the parquet metadata (no scan) and are saved next to each file.

# In[7]:


double_columns = [name for name, dtype in schema.items() if dtype == "DOUBLE"]

# columns per NaN/inf scan query; keeps each query small
scan_batch = 500


def count_nan_inf(con, parquet_path):
    """Count NaN and infinite values per measurement column."""
    counts = {}
    for start in range(0, len(double_columns), scan_batch):
        batch = double_columns[start : start + scan_batch]
        select = ", ".join(f'count(*) FILTER (WHERE isnan("{c}") OR isinf("{c}"))' for c in batch)
        row = con.execute(f"SELECT {select} FROM read_parquet('{parquet_path}')").fetchall()[0]
        counts.update(zip(batch, row))
    return {name: n for name, n in counts.items() if n}


summary = []
with duckdb.connect() as con:
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {n_threads}")

    for plate, paths in plates.items():
        parquet_path = out_dir / f"{plate}.parquet"
        expected_cells = json.loads(paths["merge_report"].read_text())["cells"]

        n_rows = con.execute(
            f"SELECT sum(num_rows) FROM parquet_file_metadata('{parquet_path}')"
        ).fetchall()[0][0]
        parquet_columns = [
            row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").fetchall()
        ]
        if n_rows != expected_cells:
            raise ValueError(f"{plate}: {n_rows} rows, MERGE report has {expected_cells} cells")
        if parquet_columns != header:
            raise ValueError(f"{plate}: parquet columns differ from the CSV header")

        nan_inf = count_nan_inf(con, parquet_path)
        if nan_inf:
            raise ValueError(f"{plate}: NaN/inf left in {len(nan_inf)} columns, e.g. {list(nan_inf.items())[:5]}")

        null_counts = con.sql(f"""
            SELECT path_in_schema AS column, sum(stats_null_count)::BIGINT AS n_null
            FROM parquet_metadata('{parquet_path}')
            GROUP BY path_in_schema
        """).pl().sort("n_null", descending=True)
        null_counts.write_csv(out_dir / f"{plate}_missing-values.csv")

        summary.append({
            "plate": plate,
            "cells": n_rows,
            "columns": len(parquet_columns),
            "columns_with_nulls": null_counts.filter(pl.col("n_null") > 0).height,
            "max_nulls_in_a_column": null_counts["n_null"].max(),
            "parquet_GiB": round(parquet_path.stat().st_size / 2**30, 1),
        })

print(pl.DataFrame(summary))

