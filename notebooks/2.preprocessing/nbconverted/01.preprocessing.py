#!/usr/bin/env python
# coding: utf-8

# # Preprocessing
# 
# **Goal:** turn the single-cell measurements into profiles that we can compare across treatments, plates and cell lines.
# 
# After loading the cells (section 1), we use [mantispy](https://github.com/scverse/mantispy) for 5 steps, in this order:
# 
# 1. **Annotation (section 2):** add what each well received (compound or ligand, and its dose), and mark the DMSO wells as controls.
# 2. **Normalization, MAD (section 3):** express every feature relative to the DMSO cells of the same plate and cell line.
# 3. **Feature selection (section 4):** drop features that carry no information or repeat another feature.
# 4. **Well QC (section 5):** flag wells with too few cells.
# 5. **Aggregation (section 6):** take the median of the cells in each well, to get one profile per well.
# 
# Section 7 saves the results. We run these steps on two sets of cells, and save 4 profiles, each with all 3 plates:
# 
# | Set | Cells | Single-cell profile | Well profile |
# |---|---|---|---|
# | **All cells** | every cell | `norm_fs_Panc1_hPSC_sc-profiles.parquet` | `agg_norm_fs_Panc1_hPSC_well-profiles.parquet` |
# | **QCed** | only cells that pass both image QC and profile QC | `qced_norm_fs_Panc1_hPSC_sc-profiles.parquet` | `qced_agg_norm_fs_Panc1_hPSC_well-profiles.parquet` |
# 
# The prefixes name the steps: **qced** = bad images and bad cells removed, **norm** = MAD-normalized, **fs** = feature-selected, **agg** = aggregated per well.
# 
# - Each set gets its own normalization and feature selection, so bad cells never shape the QCed profiles.
# - The all-cells profiles keep the QC columns, so a later analysis can still filter the cells.
# 
# **Summary of the result:** 4 profiles, each with the 3 plates and 1,140 wells. The all-cells set has 2,647,721 cells with 1,394 features. The QCed set has 2,185,564 cells with 1,399 features. 4 QCed wells on the 24h CL1 plate fail well QC, because they have fewer than 100 cells.
# 
# | Term | Meaning |
# |---|---|
# | Feature | One measurement of a cell, such as nucleus area. |
# | Profile | All features of one cell, or of one well. |
# | MAD | Median absolute deviation: a measure of spread that a few extreme cells cannot move. |
# | CL1, CL2 | The two cell lines, in columns 1–12 and 13–24. |
# 
# ## Setup
# 
# Run with `_run.sbatch`. The job holds the data of 2.6 million cells in memory, so it asks for 400 GB.

# In[1]:


import os
import pathlib

n_threads = 32
os.environ["POLARS_MAX_THREADS"] = str(n_threads)
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["NUMBA_NUM_THREADS"] = str(n_threads)

import mantispy as mp
import numpy as np
import polars as pl


# In[2]:


# input: the QC profile of notebook 1.quality-control/02.qc.ipynb
qc_profile_path = pathlib.Path("../1.quality-control/results/qc_Panc1_hPSC_sc-profiles.parquet").resolve(strict=True)
metadata_dir = pathlib.Path("../../data/metadata").resolve(strict=True)

# output: 4 profiles, each with all 3 plates
# prefixes: qced = only cells that pass image and profile QC, norm = MAD-normalized,
# fs = feature-selected, agg = aggregated to one profile per well
results_dir = pathlib.Path("results").resolve()
results_dir.mkdir(parents=True, exist_ok=True)
output_paths = {
    "all cells": {
        "single cells": results_dir / "norm_fs_Panc1_hPSC_sc-profiles.parquet",
        "wells": results_dir / "agg_norm_fs_Panc1_hPSC_well-profiles.parquet",
    },
    "qced": {
        "single cells": results_dir / "qced_norm_fs_Panc1_hPSC_sc-profiles.parquet",
        "wells": results_dir / "qced_agg_norm_fs_Panc1_hPSC_well-profiles.parquet",
    },
}

# the QC columns: the all-cells profiles keep them; the QCed profiles drop them, since every cell passes
qc_columns = ["Metadata_image_qc", "Metadata_image_qc_reason", "Metadata_profile_qc", "Metadata_profile_qc_reason"]

# CellProfiler compartments whose measurements are features
compartments = ("Cells_", "Cytoplasm_", "Nuclei_")
# columns that hold the ID number of a neighboring cell, not a measurement
non_measurement_columns = ("ObjectNumber",)

# ligand doses: the key in ligands.csv gives L, M and H in ng/ml, without naming the ligand,
# so we assume it applies to both TGFb1 and IL6
ligand_ng_per_ml = {"L": 2.5, "M": 5.0, "H": 10.0}

# normalization: MAD, fitted on the DMSO cells of each plate and cell line
normalize_by = ["Metadata_Plate", "Metadata_cell_line"]

# feature selection: the default operations of pycytominer, plus features normalization could not scale,
# and features with extreme values (above 500 DMSO spreads, from a DMSO spread near zero)
feature_select_operations = ("drop_degenerate", "variance_threshold", "correlation_threshold", "drop_na_columns", "blocklist", "drop_outliers")
correlation_limit = 0.9
missing_limit = 0.05
outlier_limit = 500

# well QC: a well fails if it has fewer cells than this, or more than this share of missing values
min_cells_per_well = 100
max_missing_per_well = 0.1

# aggregation: one profile per well, the median of its cells
aggregate_by = ("Metadata_Plate", "Metadata_Well")

plate_order = ["1h", "4h", "24h"]
set_order = ["all cells", "qced"]


def in_order(table, line="Metadata_cell_line", time="Metadata_time_point"):
    """Sort a table by profile set (if present), cell line, then time point."""
    keys = [pl.col("profiles").replace_strict({s: i for i, s in enumerate(set_order)}, return_dtype=pl.Int64)] if "profiles" in table.columns else []
    return table.sort(*keys, line, pl.col(time).replace_strict({t: i for i, t in enumerate(plate_order)}, return_dtype=pl.Int64))


pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(170)


# ## 1. Load the cells
# 
# We load every cell of the QC profile, with its metadata (including the QC columns) and all compartment measurements. We leave out the 4 columns that hold the ID number of a neighboring cell (`...ClosestObjectNumber...`). These columns are labels, not measurements.

# In[3]:


schema = pl.scan_parquet(qc_profile_path).collect_schema()
metadata_columns = [name for name in schema.names() if name.startswith("Metadata_")]
compartment_columns = [name for name in schema.names() if name.startswith(compartments) and not any(word in name for word in non_measurement_columns)]
passed = (pl.col("Metadata_image_qc") == "pass") & (pl.col("Metadata_profile_qc") == "pass")

qc_counts = in_order(
    pl.scan_parquet(qc_profile_path)
    .group_by("Metadata_cell_line", "Metadata_time_point")
    .agg(pl.len().alias("all cells"), passed.sum().alias("pass QC (qced)"), (100 * passed.mean()).round(1).alias("% pass"))
    .collect()
)

# features are stored as 32-bit numbers, which halves the memory and keeps 7 significant digits
cells = pl.scan_parquet(qc_profile_path).select(*metadata_columns, *[pl.col(name).cast(pl.Float32) for name in compartment_columns]).collect()
print(f"{cells.height:,} cells, {len(metadata_columns)} metadata columns, {len(compartment_columns):,} compartment columns")
qc_counts


# **Result:** the data hold 2,647,721 cells in total. 2,185,564 of them pass both QC checks (73.1–88.9% per plate and cell line).

# ## 2. Annotation
# 
# We add what each well received, from the plate map and the ligand file:
# 
# | Column | Content |
# |---|---|
# | `Metadata_compound`, `Metadata_compound_stock_mM` | The compound (DMSO for none), and its stock concentration in mM |
# | `Metadata_ligand`, `Metadata_ligand_dose`, `Metadata_ligand_ng_per_ml` | The ligand (TGFb1 or IL6), its dose level (L, M or H), and that dose in ng/ml |
# | `Metadata_Perturbation` | One name per condition, for example `CBK041498_0.67`, `TGFb1_L` or `DMSO` |
# 
# mantispy then marks the DMSO wells as negative controls (`Metadata_Control_Type`), which normalization uses. Finally, we split the cells into the two sets: all cells, and the QCed cells.
# 
# **Pass if:** every cell gets a perturbation, and the plate map agrees with the treatment that QC recorded for each well.

# In[4]:


# set the metadata for each well, which is either a compound or a ligand, and its dose
platemap = pl.read_csv(metadata_dir / "platemap.csv", infer_schema_length=0).select(
    pl.col("Well").alias("Metadata_Well"),
    pl.col("Compound ID").alias("Metadata_compound"),
    pl.col("Conc mM").cast(pl.Float64).round(2).alias("Metadata_compound_stock_mM"),
)
ligands = (
    pl.read_csv(metadata_dir / "ligands.csv", infer_schema_length=0)
    .filter(pl.col("destination well").is_not_null())
    .select(
        (pl.col("destination well").str.slice(0, 1) + pl.col("destination well").str.slice(1).str.zfill(2)).alias("Metadata_Well"),
        pl.col("Treatment").alias("Metadata_ligand"),
        pl.col("Concentration final").str.strip_chars().alias("Metadata_ligand_dose"),
    )
)

# annotate the cells with the perturbation, which is either the compound or the ligand, and its dose
annotation = (
    platemap.join(ligands, on="Metadata_Well", how="left")
    .with_columns(pl.col("Metadata_ligand_dose").replace_strict(ligand_ng_per_ml, default=None, return_dtype=pl.Float64).alias("Metadata_ligand_ng_per_ml"))
    .with_columns(
        pl.when(pl.col("Metadata_compound") != "DMSO")
        .then(pl.format("{}_{}", "Metadata_compound", "Metadata_compound_stock_mM"))
        .when(pl.col("Metadata_ligand").is_not_null())
        .then(pl.format("{}_{}", "Metadata_ligand", "Metadata_ligand_dose"))
        .otherwise(pl.lit("DMSO"))
        .alias("Metadata_Perturbation")
    )
)
if annotation["Metadata_Well"].is_duplicated().any():
    raise ValueError("The plate map repeats a well")

cells = cells.join(annotation, on="Metadata_Well", how="left")
if cells["Metadata_Perturbation"].null_count():
    raise ValueError("Some cells have no perturbation")

# the treatment QC recorded must be the compound or the ligand of the plate map
recorded = pl.when(pl.col("Metadata_compound") != "DMSO").then(pl.col("Metadata_compound")).otherwise(pl.col("Metadata_ligand").fill_null("DMSO"))
mismatch = cells.filter(recorded != pl.col("Metadata_treatment")).height
if mismatch:
    raise ValueError(f"The plate map disagrees with the QC treatment in {mismatch:,} cells")

# mantispy holds the cells as an AnnData object: a matrix of features, and a table of metadata
adata = mp.io.from_dataframe(cells.to_pandas(), resolution="object", object_type="cell")
del cells
mp.pp.annotate_controls(adata, negcon=("DMSO",), perturbation_key="Metadata_Perturbation")

# the two sets: all cells, and only the cells that pass both image QC and profile QC
passes_qc = ((adata.obs["Metadata_image_qc"] == "pass") & (adata.obs["Metadata_profile_qc"] == "pass")).to_numpy()
profiles = {"all cells": adata, "qced": adata[passes_qc].copy()}
profiles["qced"].obs = profiles["qced"].obs.drop(columns=qc_columns)
del adata

rows = []
for name, profile in profiles.items():
    obs = pl.from_pandas(profile.obs[["Metadata_cell_line", "Metadata_time_point", "Metadata_Perturbation", "Metadata_Control_Type"]].astype(str).reset_index(drop=True))
    rows.append(
        obs.group_by("Metadata_cell_line", "Metadata_time_point")
        .agg(
            pl.col("Metadata_Perturbation").n_unique().alias("perturbations"),
            (pl.col("Metadata_Control_Type") == "negcon").sum().alias("DMSO cells"),
            (pl.col("Metadata_Control_Type") != "negcon").sum().alias("treated cells"),
        )
        .with_columns(pl.lit(name).alias("profiles"))
    )
    print(f"{name}: {profile.n_obs:,} cells, {profile.n_vars:,} features")
in_order(pl.concat(rows).select("profiles", pl.exclude("profiles")))


# **Result: PASS.**
# 
# - Every cell has a perturbation, and the plate map agrees with the QC treatment in every well.
# - Each plate and cell line has 22 perturbations: DMSO, 5 compounds at 3 doses, and 2 ligands at 3 doses.
# - The DMSO controls hold 232,164–450,511 cells per plate and cell line in the all-cells set, and 181,683–397,451 cells in the QCed set.

# ## 3. Normalization (MAD)
# 
# Raw features differ between plates and cell lines for reasons unrelated to treatment, such as the longer exposures of the 24h plate. So, for each plate and cell line, we rescale every feature against the DMSO cells:
# 
# `normalized value = (value − DMSO median) / (1.4826 × DMSO MAD)`
# 
# - A value of 0 means "like DMSO", and a value of 3 means 3 spreads above DMSO.
# - The factor 1.4826 makes the MAD comparable to a standard deviation.
# - Each set uses its own DMSO cells. The QCed set uses only DMSO cells that pass QC.
# 
# **Pass if:** in every plate and cell line, the DMSO cells end at a median of 0 and a spread of about 1 for each feature.

# In[5]:


rows = []
for name, profile in profiles.items():
    # normalize the features of each plate and cell line, using the DMSO cells as reference
    mp.pp.normalize(profile, method="mad_robustize", by=normalize_by, reference="negcon")

    # check: the DMSO cells of each group should now have median 0 and MAD 1, feature by feature
    is_dmso = (profile.obs["Metadata_Control_Type"] == "negcon").to_numpy()
    for (plate, line), index in profile.obs.groupby(normalize_by, observed=True).indices.items():
        dmso = profile.X[index[is_dmso[index]]]
        medians = np.nanmedian(dmso, axis=0)
        spreads = 1.4826 * np.nanmedian(np.abs(dmso - medians), axis=0)
        rows.append(
            {
                "profiles": name,
                "Metadata_time_point": str(profile.obs["Metadata_time_point"].iloc[index[0]]),
                "Metadata_cell_line": str(line),
                "DMSO cells": int(is_dmso[index].sum()),
                "typical median": round(float(np.nanmedian(medians)), 3),
                "typical spread": round(float(np.nanmedian(spreads)), 3),
            }
        )
    print(f"{name}: features that cannot be scaled in at least one group (no spread in its DMSO cells): {int(profile.var['degenerate_scale'].sum())}")
in_order(pl.DataFrame(rows))


# **Result: PASS.**
# 
# - In every plate and cell line, the DMSO cells end at a median of 0 and a spread of 1.
# - 146 features (all cells) and 150 features (QCed) have no spread in the DMSO cells of at least one plate and cell line. Normalization cannot scale them, so feature selection drops them.

# ## 4. Feature selection
# 
# Many CellProfiler features carry no information, or repeat another feature. We drop a feature if any of these is true:
# 
# | Operation | Drops a feature that |
# |---|---|
# | Cannot be scaled | has no spread in the DMSO cells of some plate and cell line, so normalization could not scale it |
# | Variance | is almost constant across all cells |
# | Correlation | correlates above 0.9 with a feature that is kept (one of each pair is dropped) |
# | Missing values | is missing in more than 5% of cells |
# | Blocklist | is on pycytominer's list of known unreliable features |
# | Extreme values | reaches values above 500 DMSO spreads, which happens when its DMSO spread is close to zero |
# 
# Each set selects its own features, so the two sets can keep slightly different features.
# 
# **Pass if:** the kept features still cover all 3 compartments and all 5 channels.

# In[6]:


dropped, kept_by_group = [], []
for name in set_order:
    profile = profiles[name]
    # select the features that are not degenerate, not too correlated, and not too missing
    mp.pp.feature_select(profile, operations=feature_select_operations, corr_threshold=correlation_limit, na_cutoff=missing_limit, outlier_cutoff=outlier_limit, blocklist="default")
    dropped.append({"profiles": name, "features before": profile.n_vars, **profile.uns["mantispy"]["feature_select"], "features kept": int(profile.var["selected"].sum())})

    # subset the features to those that passed all operations, and check how many features are kept per object and channel
    profiles[name] = mp.pp.subset_features(profile, key="selected")
    del profile
    kept = pl.from_pandas(profiles[name].var[["object", "channel"]].astype(str).reset_index(drop=True)).with_columns(
        pl.when(pl.col("channel").is_in(["nan", "None", ""]))
        .then(pl.lit("none (shape and size)"))
        .when(pl.col("channel").str.contains("|", literal=True))
        .then(pl.lit("two channels (colocalization)"))
        .otherwise(pl.col("channel").str.replace("^Corr", ""))
        .alias("channel")
    )
    kept_by_group.append(
        pl.concat([kept.group_by("object").len().rename({"object": "group"}), kept.group_by("channel").len().rename({"channel": "group"})]).with_columns(pl.lit(name).alias("profiles"))
    )

shared = set(profiles["all cells"].var_names) & set(profiles["qced"].var_names)
print(f"Features kept in both sets: {len(shared):,}")
# features each operation would drop on its own (an operation can drop the same feature as another)
pl.DataFrame(dropped)


# In[7]:


# kept features per compartment and per channel
pl.concat(kept_by_group).pivot(on="profiles", index="group", values="len").sort("group")


# **Result: PASS.**
# 
# - Of 3,052 features, 1,394 are kept for all cells and 1,399 for the QCed cells (1,326 in both).
# - Most dropped features repeat another feature (correlation: 1,421).
# - The extreme-value filter alone would drop 319 and 295 features. Another filter also drops many of these.
# - The kept features cover the 3 compartments and all 5 channels (204–243 features per channel).

# ## 5. Well QC
# 
# Some wells have very few cells left after QC, for example in the low-count areas of the 24h plate. The median of a few cells is noisy.
# 
# - mantispy's well QC marks each well `pass` or `fail` in `Metadata_well_qc`, with the reason in `Metadata_well_qc_reason`.
# - No well is removed. Later notebooks leave out the wells that fail.
# 
# **Pass if:** only a few wells fail, and only for a clear reason.

# In[8]:


rows = []
for name in set_order:
    profile = profiles[name]
    # flag wells with too few cells, or too many missing values, before they are aggregated
    mp.pp.well_qc(profile, min_cells=min_cells_per_well, max_nan_fraction=max_missing_per_well, key=None)
    table = pl.from_pandas(profile.uns["mantispy"]["well_qc"].astype({"Metadata_Plate": str, "Metadata_Well": str}))
    table = table.with_columns(
        pl.concat_str(
            [
                pl.when(pl.col("n_cells") < min_cells_per_well).then(pl.lit(f"fewer than {min_cells_per_well} cells")),
                pl.when(pl.col("nan_fraction") > max_missing_per_well).then(pl.lit("too many missing values")),
            ],
            separator="; ",
            ignore_nulls=True,
        ).alias("Metadata_well_qc_reason")
    )

    # every cell gets the result of its well, so the well profiles carry it too
    reasons = dict(zip(zip(table["Metadata_Plate"], table["Metadata_Well"]), table["Metadata_well_qc_reason"]))
    profile.obs["Metadata_well_qc_reason"] = [reasons[key] for key in zip(profile.obs["Metadata_Plate"].astype(str), profile.obs["Metadata_Well"].astype(str))]
    profile.obs["Metadata_well_qc"] = np.where(profile.obs["qc_well_fail"].astype(bool), "fail", "pass")
    profile.obs = profile.obs.drop(columns="qc_well_fail")

    failed = table.filter(pl.col("qc_well_fail"))
    print(f"{name}: {failed.height} of {table.height} wells fail well QC")
    rows.append(failed.select(pl.lit(name).alias("profiles"), "Metadata_Plate", "Metadata_Well", "n_cells", pl.col("nan_fraction").round(3), "Metadata_well_qc_reason"))
pl.Config.set_fmt_str_lengths(45)
pl.concat(rows)


# **Result: PASS.**
# 
# - No well fails in the all-cells set.
# - In the QCed set, 4 of 1,140 wells fail, all on the 24h plate in CL1, and all because they have fewer than 100 cells: L04 (89), N03 (76), N08 (62) and N10 (39).
# - These wells lie in the low-count area of that plate. They keep their profile, flagged `fail`.

# ## 6. Aggregation
# 
# We take the median of the cells in each well, feature by feature, to get one profile per well. A few unusual cells do not move the median. `Metadata_CellCount` records how many cells each well profile uses.
# 
# **Pass if:** every imaged well of the 3 plates gets a profile, in both sets.

# In[9]:


wells = {name: mp.tl.aggregate(profiles[name], by=aggregate_by, func="median") for name in set_order}

rows = []
for name, well_profile in wells.items():
    obs = pl.from_pandas(well_profile.obs[["Metadata_cell_line", "Metadata_time_point", "Metadata_CellCount"]].astype({"Metadata_cell_line": str, "Metadata_time_point": str}).reset_index(drop=True))
    rows.append(
        obs.group_by("Metadata_cell_line", "Metadata_time_point")
        .agg(pl.len().alias("wells"), pl.col("Metadata_CellCount").min().alias("fewest cells"), pl.col("Metadata_CellCount").median().alias("median cells"))
        .with_columns(pl.lit(name).alias("profiles"))
    )
    print(f"{name}: {well_profile.n_obs:,} well profiles, {well_profile.n_vars:,} features")
in_order(pl.concat(rows).select("profiles", pl.exclude("profiles")))


# **Result: PASS.**
# 
# - Each set has 1,140 well profiles, 190 per plate and cell line.
# - A QCed well holds a median of 1,490–3,186 cells, depending on the plate and cell line. The fewest is 39 (24h, CL1), a well that fails well QC.

# ## 7. Save the profiles
# 
# We save the 4 profiles listed at the top. In each set, the single-cell and well profiles hold the same features, after the `Metadata_` columns.
# 
# **Pass if:** each file holds the 3 plates, the expected rows, and the features of its set.

# In[10]:


def save(adata_to_save, path):
    """Write a profile to parquet; a .partial file first, so a broken run never leaves a file that looks finished."""
    partial = path.with_name(path.name + ".partial")
    mp.get.to_dataframe(adata_to_save).reset_index(drop=True).to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)


rows = []
for name in set_order:
    for level, adata_to_save in [("single cells", profiles[name]), ("wells", wells[name])]:
        path = output_paths[name][level]
        save(adata_to_save, path)
        names = pl.scan_parquet(path).collect_schema().names()
        features = [column for column in names if not column.startswith("Metadata_")]
        counts = pl.scan_parquet(path).select(pl.len().alias("rows"), pl.col("Metadata_Plate").n_unique().alias("plates")).collect().row(0, named=True)
        if counts["rows"] != adata_to_save.n_obs or counts["plates"] != 3 or features != list(profiles[name].var_names):
            raise ValueError(f"{path.name} does not hold the expected rows, plates and features")
        has_missing = pl.scan_parquet(path).select([(pl.col(column).is_null() | pl.col(column).is_nan()).any() for column in features]).collect().row(0)
        rows.append(
            {
                "file": path.name,
                **counts,
                "metadata columns": len(names) - len(features),
                "features": len(features),
                "features with missing values": sum(has_missing),
                "GiB": round(path.stat().st_size / 2**30, 2),
            }
        )
pl.Config.set_fmt_str_lengths(60)
pl.DataFrame(rows)


# **Result: PASS.**
# 
# - Every file holds the 3 plates.
# - The single-cell files hold 2,647,721 and 2,185,564 cells, and the well files hold 1,140 wells each.
# - In the single-cell files, 520 (all cells) and 512 (QCed) features are missing in some cells. The well files have no missing values.

# ## Summary
# 
# - **Annotation:** every cell has its treatment and dose. Each plate and cell line has 22 perturbations: DMSO, 5 compounds and 2 ligands, each at 3 doses.
# - **Normalization:** every feature is expressed against the DMSO cells of the same plate and cell line (0 = like DMSO).
# - **Feature selection:** 3,052 features → 1,394 (all cells) and 1,399 (QCed). It drops repeated features, features that cannot be scaled, and features with extreme values. The 4 ID columns of neighboring cells are not features.
# - **Well QC:** 4 QCed wells on the 24h CL1 plate have fewer than 100 cells, flagged `fail` in `Metadata_well_qc`.
# - **Aggregation:** one profile per well, the median of its cells: 1,140 wells per set.
# - **Use** the `qced_` profiles for analysis, and keep the wells where `Metadata_well_qc` is `pass`.
