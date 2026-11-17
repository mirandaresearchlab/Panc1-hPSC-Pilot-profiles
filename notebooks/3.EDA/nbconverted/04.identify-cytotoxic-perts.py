#!/usr/bin/env python
# coding: utf-8

# # Identify cytotoxic treatments
# 
# **Question:** which treatments kill cells or stop their growth? And do their profile changes come from the cell loss?
# 
# We count the cells in each well from the single-cell profiles. Then we compare every treatment with the DMSO wells of the same plate and cell line. A treatment is **cytotoxic** when its cell count falls more than 3 robust standard deviations (SD) below DMSO.
# 
# Cell number also changes how cells look: in DMSO wells, 23–59% of features follow cell number (notebook 01). So, for the treatments that change the cells, we also ask whether their profile change looks like a DMSO well with fewer cells (section 5).
# 
# Sections: 1. count cells per well, 2. set the threshold from the DMSO wells, 3. call cytotoxic treatments, 4. cell loss per treatment, 5. do the profile changes come from the cell loss?
# 
# ## Setup
# 
# Run notebook 03 first: section 5 reads its activity calls (`treatment_scores.csv`).

# In[1]:


import os

# limit threads, because this can run on a shared login node
os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats


# In[2]:


# inputs: all segmented cells (before QC), with treatment metadata (only a few metadata columns are read),
# the QCed well profiles, and the activity scores of notebook 03
sc_path = pathlib.Path("../2.preprocessing/results/norm_fs_Panc1_hPSC_sc-profiles.parquet").resolve(strict=True)
well_profiles_path = pathlib.Path("../2.preprocessing/results/qced_agg_norm_fs_Panc1_hPSC_well-profiles.parquet").resolve(strict=True)
activity_path = pathlib.Path("results/tables/03.replicate-health-diagnosis/treatment_scores.csv").resolve(strict=True)

# outputs
fig_dir = pathlib.Path("results/figures/04.identify-cytotoxic-perts").resolve()
table_dir = pathlib.Path("results/tables/04.identify-cytotoxic-perts").resolve()
fig_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

# a treatment is cytotoxic when the median z-score of its 3 wells is at or below -n_sd
n_sd = 3
# random sets of 3 DMSO wells, used to check how often the rule calls DMSO cytotoxic
n_null_draws = 100_000
seed = 0
# values beyond this many MADs are set to the limit, so a few extreme features cannot dominate (as in notebooks 01 and 03)
clip_limit = 20

# fixed order and colors, the same in every notebook (checked for color-blind readers)
plate_order = ["1h", "4h", "24h"]
line_order = ["CL1", "CL2"]
line_colors = {"CL1": "#6250d6", "CL2": "#eda100"}
plate_colors = {"1h": "#2a78d6", "4h": "#eb6834", "24h": "#1baf7a"}
compounds = ["CBK041498", "CBK175924", "CBK200566", "CBK600005", "CBK600008"]
treatment_order = ["TGFb1", "IL6", *compounds]
perturbation_order = [f"{t}_{d}" for t in ["TGFb1", "IL6"] for d in ["L", "M", "H"]] + [
    f"{c}_{d}" for c in compounds for d in ["0.2", "0.67", "2.0"]
]
# readable dose labels: compound stock concentration, or final ligand concentration
dose_text = {"0.2": "0.2 mM", "0.67": "0.67 mM", "2.0": "2 mM", "L": "2.5 ng/ml", "M": "5 ng/ml", "H": "10 ng/ml"}
dmso_color, treated_color, called_color = "#bdbdbd", "#525252", "#e34948"
div_cmap = plt.get_cmap("RdBu_r")

# figure style for print: figure sizes are the size on paper (7.2 in = full page width), text is 6-9 pt
sns.set_theme(style="ticks")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6.5, "legend.title_fontsize": 7, "figure.titlesize": 9,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
})
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)
rng = np.random.default_rng(seed)


def save_fig(fig, name):
    """Save a figure as PNG (300 dpi) and PDF, then show it."""
    fig.savefig(fig_dir / f"{name}.png")
    fig.savefig(fig_dir / f"{name}.pdf")
    plt.show()


def time_sort(df):
    """Sort rows by time point (1h, 4h, 24h), then cell line."""
    return df.sort(pl.col("time_point").cast(pl.Enum(plate_order)), "cell_line")


def pretty(perturbation):
    """Readable treatment name, e.g. "CBK041498 0.2 mM"."""
    agent, dose = perturbation.split("_")
    return f"{agent} {dose_text[dose]}"


# ## 1. Count cells per well
# 
# We count every segmented cell per well, before QC, because cells that fail QC are still cells in the well. Wells that fail well QC also keep their counts, because a low count is exactly what we look for here.
# 
# **Pass if:** the data hold 1,140 wells, each with 9 imaged fields of view.

# In[3]:


well_keys = ["Metadata_Plate", "Metadata_Well"]
meta_cols = [*well_keys, "Metadata_cell_line", "Metadata_time_point", "Metadata_treatment", "Metadata_Perturbation"]

# lazy scan: only the metadata columns are read, not the features
wells = (
    pl.scan_parquet(sc_path)
    .select(*meta_cols, "Metadata_Site")
    .with_columns(pl.col(pl.Categorical).cast(pl.String))
    .group_by(meta_cols)
    .agg(n_cells=pl.len(), n_sites=pl.col("Metadata_Site").n_unique())
    .collect()
    .rename({"Metadata_cell_line": "cell_line", "Metadata_time_point": "time_point", "Metadata_treatment": "treatment"})
    .with_columns(is_dmso=pl.col("treatment") == "DMSO")
    .sort(well_keys)
)

assert wells.height == 1140
print(f"{wells.height:,} wells, {wells['n_cells'].sum():,} cells")
print(f"wells with a field of view without cells: {(wells['n_sites'] < 9).sum()}")
time_sort(wells.group_by("time_point", "cell_line").agg(wells=pl.len(), dmso_wells=pl.col("is_dmso").sum()))


# **Result:**
# 
# - The data hold 1,140 wells and 2,647,721 cells. Each plate and cell line has 127 DMSO wells and 21 treatments × 3 wells.
# - 3 wells have one field of view without cells (24h plate, CL1). They sit inside the low-count area of notebook 02.

# ## 2. Set the threshold from the DMSO wells
# 
# 1. For each plate and cell line, we scale every well to the DMSO median (relative cell count).
# 2. We measure the robust SD of the DMSO wells (1.4826 × MAD).
# 3. The threshold is the DMSO median minus 3 robust SDs.
# 
# We use the median of the 3 replicate wells, so one odd well cannot make a call. A median at or below the threshold means that at least 2 of the 3 wells are below it.
# 
# **Negative control:** we apply the same rule to 100,000 random sets of 3 DMSO wells. The share of sets that the rule calls cytotoxic is the false-positive rate per test.
# 
# **Pass if:** the false-positive rate is below about 1%. Then we expect fewer than 1 false call among the 126 treatment tests.

# In[4]:


group_keys = ["Metadata_Plate", "cell_line"]
wells = (
    wells.with_columns(rel_count=pl.col("n_cells") / pl.col("n_cells").filter(pl.col("is_dmso")).median().over(group_keys))
    .with_columns(robust_sd=(1.4826 * (pl.col("rel_count") - 1).abs().filter(pl.col("is_dmso")).median()).over(group_keys))
    .with_columns(z=(pl.col("rel_count") - 1) / pl.col("robust_sd"))
)

dmso_rows = []
for (plate, line), grp in wells.filter(pl.col("is_dmso")).group_by(group_keys, maintain_order=True):
    z = grp["z"].to_numpy()
    # random sets of 3 DMSO wells; the rule calls a set when its median z-score is <= -n_sd
    picks = np.argsort(rng.random((n_null_draws, len(z))), axis=1)[:, :3]
    null_medians = np.median(z[picks], axis=1)
    rsd = grp["robust_sd"][0]
    dmso_rows.append(
        {
            "time_point": grp["time_point"][0], "cell_line": line, "n_dmso_wells": len(z),
            "dmso_median_cells": grp["n_cells"].median(),
            "sd_pct": 100 * grp["rel_count"].std(), "robust_sd_pct": 100 * rsd,
            "threshold_pct_of_dmso": 100 * (1 - n_sd * rsd),
            "false_positive_pct": 100 * np.mean(null_medians <= -n_sd),
            "false_positive_pct_at_2sd": 100 * np.mean(null_medians <= -2),
        }
    )
dmso_table = time_sort(pl.DataFrame(dmso_rows))
dmso_table.write_csv(table_dir / "dmso_threshold.csv")

# 21 treatments are tested per plate and cell line
expected_false = (21 * dmso_table["false_positive_pct"] / 100).sum()
print(f"expected false calls among the 126 tests at {n_sd} SD: {expected_false:.2f}")
dmso_table.with_columns(pl.selectors.float().round(2))


# **Result: PASS.**
# 
# - The robust SD of DMSO is 7–13% on the 1h and 4h plates and for CL2 at 24h. So the threshold lies at 61–78% of DMSO.
# - CL1 at 24h is much noisier (robust SD 22%), because of the low-count area of the plate. Its threshold is 33% of DMSO, so the rule calls only a strong cell loss on this plate.
# - At 3 SD, the rule calls at most 0.22% of random DMSO sets (0.07 false calls expected in 126 tests).
# - At 2 SD, it would call up to 2.7% (about 1.6 false calls expected). So we keep 3 SD.

# ## 3. Call cytotoxic treatments
# 
# For each treatment, plate and cell line, we take the median relative cell count and the median z-score of its 3 wells. We call the treatment cytotoxic when its median z-score is at or below −3.
# 
# **Pass if:** the replicate wells of a called treatment agree (all 3 wells well below DMSO).

# In[5]:


calls = (
    wells.filter(~pl.col("is_dmso"))
    .sort("Metadata_Well")
    .group_by("time_point", "cell_line", "treatment", "Metadata_Perturbation")
    .agg(
        n_wells=pl.len(),
        pct_of_dmso=100 * pl.col("rel_count").median(),
        median_z=pl.col("z").median(),
        well_z=pl.col("z").round(1).cast(pl.String).str.join(", "),
        wells_below=(pl.col("z") <= -n_sd).sum(),
    )
    .with_columns(cytotoxic=pl.col("median_z") <= -n_sd)
    .sort(
        pl.col("Metadata_Perturbation").cast(pl.Enum(perturbation_order)),
        pl.col("cell_line"),
        pl.col("time_point").cast(pl.Enum(plate_order)),
    )
)
assert calls.height == 126 and (calls["n_wells"] == 3).all()
calls.write_csv(table_dir / "cytotoxicity_calls.csv")

print(f"cytotoxic: {calls['cytotoxic'].sum()} of {calls.height} treatment x plate x cell line tests ({n_sd} robust SD)")
# called treatments, and near misses down to 2 SD
calls.filter(pl.col("median_z") <= -2).sort("median_z").drop("n_wells", "treatment").with_columns(pl.col("pct_of_dmso", "median_z").round(2))


# **Result: PASS.**
# 
# - 6 of 126 tests are cytotoxic, all in CL2 at 24h: **CBK600005** (64%, 50%, 51% of DMSO at 0.2, 0.67, 2 mM) and **CBK200566** (71%, 57%, 57%). All 3 wells of each call are at least 2.5 SD below DMSO.
# - The rule calls nothing at 1h or 4h.
# - CL1 at 24h has near misses that roughly halve the cell count:
#   - CBK600005 (33%, 48%, 54%). The 0.2 mM dose just misses, at −2.99 SD.
#   - TGFβ1 5 and 10 ng/ml (50%, 52%).
#   - CBK175924 0.67 mM (49%), but one of its wells is normal.
# - These near misses do not pass, because the DMSO wells of this plate vary so much.

# ## 4. Cell loss per treatment
# 
# Two figures show the cell loss:
# 
# 1. **Heatmap:** the median cell count of the 3 wells of each treatment (% of DMSO). A black frame marks a cytotoxic call.
# 2. **Every well:** each well against the DMSO wells and the threshold of its plate and cell line. Red dots are the wells of a cytotoxic call.

# In[6]:


heat_cols = [(line, tp) for line in line_order for tp in plate_order]


def treatment_matrix(value):
    """Pivot one treatment-level value into a perturbation x (cell line, time point) matrix."""
    mat = np.full((len(perturbation_order), len(heat_cols)), np.nan)
    for r in calls.iter_rows(named=True):
        mat[perturbation_order.index(r["Metadata_Perturbation"]), heat_cols.index((r["cell_line"], r["time_point"]))] = r[value]
    return mat


pct_mat = treatment_matrix("pct_of_dmso")
hit_mat = treatment_matrix("cytotoxic").astype(bool)
norm = TwoSlopeNorm(100, 0, 200)

fig, ax = plt.subplots(figsize=(3.8, 5.0), layout="constrained")
im = ax.imshow(pct_mat, cmap=div_cmap, norm=norm, aspect="auto")
for (r, c), v in np.ndenumerate(pct_mat):
    ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white" if abs(norm(v) - 0.5) > 0.33 else "black")
    if hit_mat[r, c]:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.4))
ax.set_xticks(range(len(heat_cols)), labels=[f"{line}\n{tp}" for line, tp in heat_cols])
ax.set_yticks(range(len(perturbation_order)), labels=[pretty(p) for p in perturbation_order])
ax.axvline(2.5, color="black", lw=1.2)
for y in [2.5, 5.5, 8.5, 11.5, 14.5, 17.5]:
    ax.axhline(y, color="white", lw=1.5)
ax.tick_params(length=0)
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.6, location="bottom", pad=0.02, label="cells, % of DMSO median (median of 3 wells)\nblack frame: ≥ 3 robust SD below DMSO")
ax.set_title("Cell number per treatment")
save_fig(fig, "01_cytotoxicity_heatmap")


# In[7]:


x_labels = ["DMSO", *perturbation_order]
x_pos = {p: i for i, p in enumerate(x_labels)}
called = calls.filter(pl.col("cytotoxic")).select("time_point", "cell_line", "Metadata_Perturbation").with_columns(called=pl.lit(True))
plot_wells = wells.join(called, on=["time_point", "cell_line", "Metadata_Perturbation"], how="left").with_columns(
    x=pl.col("Metadata_Perturbation").replace_strict(x_pos, return_dtype=pl.Float64),
    pct=100 * pl.col("rel_count"),
    called=pl.col("called").fill_null(False),
)
y_max = float(plot_wells["pct"].max()) * 1.05

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), layout="constrained", sharex=True, sharey=True)
for i, line in enumerate(line_order):
    for j, tp in enumerate(plate_order):
        ax = axes[i, j]
        sub = plot_wells.filter((pl.col("cell_line") == line) & (pl.col("time_point") == tp))
        dmso = sub.filter(pl.col("is_dmso"))
        ax.scatter(dmso["x"] + rng.uniform(-0.35, 0.35, dmso.height), dmso["pct"], s=3, color=dmso_color, linewidths=0)
        for is_called, color in [(False, treated_color), (True, called_color)]:
            trt = sub.filter(~pl.col("is_dmso") & (pl.col("called") == is_called))
            ax.scatter(trt["x"], trt["pct"], s=5, color=color, linewidths=0)
        threshold = dmso_table.filter((pl.col("cell_line") == line) & (pl.col("time_point") == tp))["threshold_pct_of_dmso"][0]
        ax.axhline(100, color="black", lw=0.5, ls="--")
        ax.axhline(threshold, color=called_color, lw=0.8)
        ax.set_title(f"{line}, {tp}  (threshold {threshold:.0f}%)")
        ax.set_ylim(0, y_max)
        for x in np.arange(0.5, len(x_labels), 3):
            ax.axvline(x, color="#ececec", lw=0.5, zorder=0)
    axes[i, 0].set_ylabel("cells, % of DMSO median")
for ax in axes[1]:
    ax.set_xticks(range(len(x_labels)), labels=["DMSO", *[pretty(p) for p in perturbation_order]], rotation=90, fontsize=5)
    ax.tick_params(axis="x", length=0)
fig.legend(
    handles=[Line2D([], [], marker="o", ls="", color=c, markersize=3, label=l) for c, l in
             [(dmso_color, "DMSO well"), (treated_color, "treated well"), (called_color, "well of a cytotoxic call")]]
    + [Line2D([], [], color=called_color, lw=0.8, label=f"threshold: DMSO median − {n_sd} robust SD"), Line2D([], [], color="black", lw=0.5, ls="--", label="DMSO median")],
    loc="lower center", bbox_to_anchor=(0.5, -0.06), ncols=5,
)
save_fig(fig, "02_cytotoxicity_replicate-wells")


# **Result:**
# 
# - CL2 at 24h: the CBK200566 and CBK600005 wells sit as a group below or near the threshold, so no single well drives the calls. One CBK041498 0.67 mM well has only 17% of DMSO cells, but its 2 replicates are normal.
# - CL1 at 24h: many DMSO wells have only 40–60% of the median, the same range as TGFβ1, CBK175924 and CBK600005. So on this plate, a cell loss cannot be told apart from the low-count area.

# ## 5. Do the profile changes come from the cell loss?
# 
# In DMSO wells, many features change with cell number (notebook 01). So a treatment that lowers cell number can look "active" partly because its wells are less crowded. We test this per plate and cell line:
# 
# 1. **Density pattern:** from the DMSO wells, we learn how each feature changes when the cell number goes down. This is the slope of a straight line per feature, on the profiles rescaled to the DMSO well-to-well spread (as in notebooks 01 and 03).
# 2. **Low-density likeness:** for each treatment, we take its profile change (median of its wells minus the DMSO median). We correlate it with the density pattern across all features (Spearman).
#    - +1 = the treatment changes the features in the same order as a DMSO well with fewer cells.
#    - 0 = unrelated.
#    - negative = like a well with more cells.
# 
# We compare patterns, not sizes. A straight line learned from DMSO wells (mostly within ±20% of the median count) is not reliable at the 50% cell loss of some treatments. The activity call (mAP q < 0.05) comes from notebook 03.
# 
# **Pass if:** the active treatments that did not lose cells have a low likeness, so their effect is not density.

# In[8]:


profiles_table = pl.read_parquet(well_profiles_path).with_columns(pl.col(pl.Categorical).cast(pl.String))
profiles_table = profiles_table.filter(pl.col("Metadata_well_qc") == "pass").join(
    wells.select(*well_keys, "rel_count"), on=well_keys, how="left"
)
feature_cols = [c for c in profiles_table.columns if not c.startswith("Metadata_") and c != "rel_count"]
X = np.clip(profiles_table.select(feature_cols).to_numpy().astype(np.float64), -clip_limit, clip_limit)
info = profiles_table.select(
    *well_keys, pl.col("Metadata_cell_line").alias("cell_line"), pl.col("Metadata_time_point").alias("time_point"),
    "Metadata_Perturbation", is_dmso=pl.col("Metadata_treatment") == "DMSO", log2_cells=pl.col("rel_count").log(2),
).with_row_index("row")

likeness_rows = []
for (plate, line), grp in info.group_by("Metadata_Plate", "cell_line", maintain_order=True):
    dmso = grp.filter(pl.col("is_dmso"))
    x_dmso = X[dmso["row"].to_numpy()]
    centre = np.median(x_dmso, axis=0)
    spread = 1.4826 * np.median(np.abs(x_dmso - centre), axis=0)
    spread = np.maximum(spread, np.quantile(spread[spread > 0], 0.01))
    rescaled = np.clip((X[grp["row"].to_numpy()] - centre) / spread, -clip_limit, clip_limit)
    position = {r: k for k, r in enumerate(grp["row"].to_list())}
    z_dmso = rescaled[[position[r] for r in dmso["row"].to_list()]]
    cells = dmso["log2_cells"].to_numpy()
    # slope of each feature against log2 cell number, across the DMSO wells
    slope = ((cells - cells.mean())[:, None] * (z_dmso - z_dmso.mean(axis=0))).sum(axis=0) / ((cells - cells.mean()) ** 2).sum()
    for (pert,), reps in grp.filter(~pl.col("is_dmso")).group_by("Metadata_Perturbation", maintain_order=True):
        shift = np.median(rescaled[[position[r] for r in reps["row"].to_list()]], axis=0)
        likeness_rows.append({
            "time_point": grp["time_point"][0], "cell_line": line, "Metadata_Perturbation": pert,
            # a negative slope is the change expected when the cell number goes down
            "low_density_likeness": stats.spearmanr(shift, -slope).statistic,
        })

activity = pl.read_csv(activity_path).select(
    "time_point", "cell_line", pl.col("perturbation").alias("Metadata_Perturbation"), "strength", "mAP_qvalue", "active",
)
density = (
    calls.select("time_point", "cell_line", "Metadata_Perturbation", "pct_of_dmso", "median_z", "cytotoxic")
    .join(pl.DataFrame(likeness_rows), on=["time_point", "cell_line", "Metadata_Perturbation"], how="left")
    .join(activity, on=["time_point", "cell_line", "Metadata_Perturbation"], how="left")
    .with_columns(
        group=pl.when(pl.col("active") & (pl.col("pct_of_dmso") < 80)).then(pl.lit("active, fewer cells (< 80%)"))
        .when(pl.col("active")).then(pl.lit("active, cell number ≥ 80%"))
        .otherwise(pl.lit("not active"))
    )
)
density.write_csv(table_dir / "cell-loss-and-density-likeness.csv")
density.group_by("group").agg(
    n=pl.len(), median_low_density_likeness=pl.col("low_density_likeness").median().round(2),
    min=pl.col("low_density_likeness").min().round(2), max=pl.col("low_density_likeness").max().round(2),
).sort("group")


# In[9]:


# the active treatments and the cytotoxic calls, with their likeness
density.filter(pl.col("active") | pl.col("cytotoxic")).sort("cell_line", pl.col("time_point").cast(pl.Enum(plate_order)), "pct_of_dmso").select(
    "cell_line", "time_point", pl.col("Metadata_Perturbation").map_elements(pretty, return_dtype=pl.String).alias("treatment"),
    pl.col("pct_of_dmso").round(0), "cytotoxic", pl.col("strength").round(1), "active", pl.col("low_density_likeness").round(2),
)


# In[10]:


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained", sharey=True)
for ax, line in zip(axes, line_order):
    sub = density.filter(pl.col("cell_line") == line)
    for tp in plate_order:
        for is_active in [False, True]:
            pts = sub.filter((pl.col("time_point") == tp) & (pl.col("active") == is_active))
            ax.scatter(pts["pct_of_dmso"], pts["low_density_likeness"], s=14 if is_active else 9,
                       facecolor=plate_colors[tp] if is_active else "white", edgecolor=plate_colors[tp], linewidths=0.8, zorder=3 if is_active else 2)
    hits = sub.filter(pl.col("cytotoxic"))
    ax.scatter(hits["pct_of_dmso"], hits["low_density_likeness"], s=40, facecolor="none", edgecolor="black", linewidths=0.6, zorder=4)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(100, color="#bdbdbd", lw=0.5)
    ax.set(title=line, xlabel="cells, % of DMSO median (median of 3 wells)", ylim=(-1, 1))
axes[0].set_ylabel("low-density likeness\n(Spearman rho over features)")
fig.legend(
    handles=[Line2D([], [], marker="o", ls="", color=plate_colors[t], markersize=4, label=t) for t in plate_order]
    + [Line2D([], [], marker="o", ls="", color="black", markersize=4, label="active (mAP q < 0.05)"),
       Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", markersize=4, label="not active"),
       Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black", markersize=7, label="cytotoxic call")],
    loc="lower center", bbox_to_anchor=(0.5, -0.12), ncols=6,
)
fig.suptitle("Does the profile change look like a DMSO well with fewer cells? (one point per treatment)")
save_fig(fig, "03_density-likeness-vs-cell-loss")


# **Result: the 1h and 4h effects are not density. The 24h cell-loss effects are partly density.**
# 
# - **Check of the method (positive control):** across all 126 treatments, treatments with fewer cells have a higher likeness. This is what we expect if the density pattern is real.
# - **Active treatments that lose cells** (12 with < 80% of DMSO cells) all have a positive likeness (0.26–0.72, median 0.45). They are the 6 cytotoxic calls (CBK600005 and CBK200566 in CL2 at 24h, 0.34–0.53), and CBK600005 and TGFβ1 in CL1 at 24h (0.26–0.72).
# - Part of their profile change is the cell loss. So do not read these profiles as morphology alone.
# - **CBK600005 at 1h and 4h** is active in both cell lines without cell loss (89–106% of DMSO), and with a negative likeness (−0.46 to 0.02). This is the clearest effect in the screen that does not come from density.
# - **TGFβ1 in CL2** at 24h (and 2.5 ng/ml at 4h) has a positive likeness (0.38–0.57) without cell loss. TGFβ1 can make cells spread and scatter. This may change the neighbor and texture features in the same way as lower density. This is a possible explanation, not tested here.
# - Three treatments in CL1 at 24h lose cells without being active: CBK175924 0.67 and 2 mM, and CBK200566 0.2 mM (49–79% of DMSO). They look most like low density (0.56–0.75).
# - For treatments with almost no profile change, the likeness is mostly noise (not active: −0.75 to 0.80).

# ## Summary
# 
# **What we did:**
# 
# 1. We counted all cells per well, and scaled each well to the DMSO median of its plate and cell line.
# 2. We called a treatment cytotoxic when the median of its 3 wells is at least 3 robust SDs below DMSO. Random sets of 3 DMSO wells check the false-positive rate (at most 0.22% per test).
# 3. We compared the profile change of each treatment with the pattern of DMSO wells that have fewer cells.
# 
# **Findings:**
# 
# - **Cytotoxic:** CBK600005 and CBK200566 in CL2 at 24h, at all 3 doses (50–71% of DMSO cells). There is no clear dose response: the 0.67 and 2 mM doses give the same loss.
# - No treatment is cytotoxic at 1h or 4h.
# - CL1 at 24h: CBK600005, TGFβ1 (5 and 10 ng/ml) and CBK175924 0.67 mM roughly halve the cell count. The rule does not call them, because the DMSO wells of this plate vary too much (threshold 33% of DMSO).
# - The profile changes of the treatments that lose cells look partly like DMSO wells with fewer cells (likeness 0.26–0.72). CBK600005 at 1h and 4h changes the cells without cell loss and without a density-like pattern.
# 
# **Caveats:**
# 
# - Fewer cells can mean cell death or slower growth. A cell count cannot separate the two.
# - Compound doses are stock concentrations. The final concentrations are unknown.
# - The likeness compares patterns, not sizes, and is a correlation, not a correction. It shows that part of a profile change may be density, but not how much.
# 
# **Outputs:**
# 
# - `tables/04.identify-cytotoxic-perts/cytotoxicity_calls.csv`: one row per treatment, plate and cell line.
# - `dmso_threshold.csv`: the DMSO spread and threshold per plate and cell line.
# - `cell-loss-and-density-likeness.csv`: the cell loss, activity and likeness of each treatment.
