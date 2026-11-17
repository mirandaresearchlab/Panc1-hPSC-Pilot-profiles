#!/usr/bin/env python
# coding: utf-8

# # Replicate health and compound activity
# 
# **Question:** do the compounds and doses capture real biology? We ask four questions:
# 
# 1. Do the replicate wells of a treatment agree with each other?
# 2. Which treatments change the cells, compared with DMSO?
# 3. How does the change depend on dose and time?
# 4. Which treatments look alike?
# 
# Words used in this notebook:
# 
# - **Replicate wells:** the 3 wells of the same treatment and dose, on the same plate and cell line.
# - **Null distribution:** the scores of wells that are *not* replicates. A real result must beat the null.
# - **Replicate correlation and percent replicating:** how similar the 3 replicate profiles are (Pearson correlation across all features), compared with the null. Percent replicating is the share of treatments that beat the 95th percentile of the null.
# - **mAP (mean average precision):** each replicate well ranks its replicates and the DMSO wells by similarity. mAP is 1 when the replicates always come first, and about 0.02 by chance (3 replicates among 127 DMSO wells). A permutation test gives a p-value. A treatment with a significant mAP is **active**: its wells are reproducibly different from DMSO.
# - **Consensus profile:** the median of the 3 replicate profiles.
# 
# DMSO is the only reference. We compare every treatment with the DMSO wells of the same plate (time point) and cell line. The data cannot separate time from the exposure settings, or cell line from plate side, so comparisons never cross plates or cell lines.
# 
# ## Setup
# 
# Run on the login node or with `_run.sbatch` (about 3 minutes, a few GB of memory).

# In[1]:


import os
import pathlib

# setting the number of threads for parallel processing
# few threads are enough for 1,140 wells
n_threads = 8
os.environ["POLARS_MAX_THREADS"] = str(n_threads)
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["NUMBA_NUM_THREADS"] = str(n_threads)

import mantispy as mt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


# In[2]:


# input: the QCed well profiles of the preprocessing module
profiles_path = pathlib.Path("../2.preprocessing/results/qced_agg_norm_fs_Panc1_hPSC_well-profiles.parquet").resolve(strict=True)

# outputs of this notebook
fig_dir = pathlib.Path("results/figures/03.replicate-health-diagnosis").resolve()
table_dir = pathlib.Path("results/tables/03.replicate-health-diagnosis").resolve()
fig_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

seed = 0
rng = np.random.default_rng(seed)
# values beyond this many MADs are set to the limit, so a few extreme features cannot dominate
clip_limit = 20
# random draws for each null distribution
n_null = 10_000
# fewer draws for the DMSO-set distances, because each draw takes a median over ~124 wells
n_null_sets = 2_000
# a score is "above chance" when it beats this quantile of its null
null_quantile = 0.95
# false discovery rate for the mAP activity call
fdr = 0.05
# a DMSO well is "unusual" when its distance from the DMSO centre is above the median + 3 MAD of all DMSO wells
unusual_mads = 3
# a profile "looks like the unusual DMSO wells" when it correlates with their median profile at least this much
artifact_corr = 0.8

# fixed orders and colors, the same as in the other notebooks
plate_order = ["1h", "4h", "24h"]
line_order = ["CL1", "CL2"]
panels = [(time, line) for time in plate_order for line in line_order]
agent_order = ["CBK041498", "CBK175924", "CBK200566", "CBK600005", "CBK600008", "TGFb1", "IL6"]
ligands = ["TGFb1", "IL6"]
dose_order = {agent: (["L", "M", "H"] if agent in ligands else ["0.2", "0.67", "2.0"]) for agent in agent_order}
perturbation_order = [f"{agent}_{dose}" for agent in agent_order for dose in dose_order[agent]]
dose_level = {"0.2": 0, "0.67": 1, "2.0": 2, "L": 0, "M": 1, "H": 2}
# compound doses are stock concentrations; the final well concentration is unknown
dose_text = {"0.2": "0.2 mM", "0.67": "0.67 mM", "2.0": "2 mM", "L": "2.5 ng/ml", "M": "5 ng/ml", "H": "10 ng/ml"}

# setting the same colors for the two cell lines and three time points in all figures
line_colors = {"CL1": "#6250d6", "CL2": "#eda100"}
plate_colors = {"1h": "#2a78d6", "4h": "#eb6834", "24h": "#1baf7a"}
well_type_colors = {"control": "#bdbdbd", "compound": "#525252", "ligand": "#e34948"}
blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#0d366b"])

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
pl.Config.set_tbl_rows(130)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(80)


def pretty(perturbation):
    # readable treatment name, e.g. "CBK041498 0.2 mM"
    agent, dose = perturbation.split("_")
    return f"{agent} {dose_text[dose]}"


def save_figure(fig, name):
    # every figure is saved as PNG (300 dpi) and PDF, then shown
    fig.savefig(fig_dir / f"{name}.png")
    fig.savefig(fig_dir / f"{name}.pdf")
    plt.show()


def panel_filter(time, line):
    # the wells of one plate (time point) and one cell line
    return (pl.col("Metadata_time_point") == time) & (pl.col("Metadata_cell_line") == line)


# ## 1. Load the well profiles
# 
# Each well profile is the median of its QC-passing cells. The preprocessing normalized the features against the DMSO cells of the same plate and cell line, so 0 means "like DMSO".
# 
# The preprocessing also marks each well as passing or failing well QC (`Metadata_well_qc`). A well fails when fewer than 100 of its cells pass cell QC, or when more than 10% of its values are missing. We leave failed wells out of every analysis below.
# 
# **Pass if:** few wells fail, and every treatment keeps at least 2 of its 3 wells per plate and cell line.

# In[3]:


adata = mt.io.read_profiles(profiles_path, resolution="well")
print(f"{adata.n_obs:,} wells x {adata.n_vars:,} features")


def well_table(obs):
    # well metadata as a polars table; "row" is the row of the well in the feature matrix
    return (
        pl.from_pandas(obs.reset_index(drop=True))
    .with_columns(pl.col(pl.Categorical).cast(pl.Utf8))
    .with_row_index("row")
    .with_columns(
        pl.col("Metadata_time_point").cast(pl.Enum(plate_order)),
        pl.col("Metadata_Perturbation").str.split("_").list.first().alias("agent"),
        pl.col("Metadata_Perturbation").str.split("_").list.get(1, null_on_oob=True).alias("dose"),
        pl.col("Metadata_CellCount").cast(pl.Int64),
    )
    .with_columns(pl.col("dose").replace_strict(dose_level, default=None).alias("dose_level"))
    )


# wells that fail well QC are listed, and left out of the analysis
all_wells = well_table(adata.obs)
failed_wells = all_wells.filter(pl.col("Metadata_well_qc") != "pass").sort("Metadata_time_point", "Metadata_cell_line", "Metadata_Well")
failed_wells.select(
    "Metadata_time_point", "Metadata_cell_line", "Metadata_Well", "Metadata_Perturbation", "Metadata_CellCount", "Metadata_well_qc_reason"
).write_csv(table_dir / "failed_qc_wells.csv")
print(f"wells failing well QC: {failed_wells.height} of {all_wells.height}")

adata = adata[(adata.obs["Metadata_well_qc"].astype(str) == "pass").to_numpy()].copy()
features = adata.var_names.to_list()
meta = well_table(adata.obs)
print(f"analysed: {adata.n_obs:,} wells x {adata.n_vars:,} features")
failed_wells.select("Metadata_time_point", "Metadata_cell_line", "Metadata_Well", "Metadata_Perturbation", "Metadata_CellCount", "Metadata_well_qc_reason")


# In[4]:


design = (
    meta.group_by("Metadata_time_point", "Metadata_cell_line")
    .agg(
        (pl.col("agent") == "DMSO").sum().alias("DMSO wells"),
        (pl.col("agent") != "DMSO").sum().alias("treated wells"),
        pl.col("Metadata_Perturbation").filter(pl.col("agent") != "DMSO").n_unique().alias("treatments"),
        pl.col("Metadata_CellCount").median().alias("median cells per well"),
    )
    .sort("Metadata_time_point", "Metadata_cell_line")
)
replicates_per_treatment = (
    meta.filter(pl.col("agent") != "DMSO").group_by("Metadata_time_point", "Metadata_cell_line", "Metadata_Perturbation").len()
)
print("wells per treatment, plate and cell line:", dict(replicates_per_treatment["len"].value_counts().sort("len").iter_rows()))
print("treatments with fewer than 3 wells left:", replicates_per_treatment.filter(pl.col("len") < 3).sort("Metadata_time_point", "Metadata_cell_line", "Metadata_Perturbation").rows())
design


# ### Clip and rescale the features
# 
# 1. **Clip.** Features with almost no spread in the DMSO cells can get huge normalized values. The preprocessing drops the worst of them. We still set any value beyond ±20 to ±20, and report how often that happens.
# 2. **Rescale to the DMSO wells.** The preprocessing scaled the features by the spread between DMSO *cells*. The spread between DMSO *wells* is much smaller, and differs between features. Without a second scaling, a few noisy features would dominate every distance.
#    - For each plate and cell line, we subtract the median DMSO well and divide by the spread of the DMSO wells (1.4826 × MAD).
#    - A value of 3 then means "3 times the normal well-to-well variation of DMSO".
#    - We clip again at ±20.
# 
# All correlations, distances and clusters below use these clipped, rescaled profiles.

# In[5]:


raw = adata.X.astype(np.float64)
extreme = np.abs(raw) > clip_limit
print(f"input: largest |value| {np.abs(raw).max():,.1f}; {extreme.sum():,} values ({100 * extreme.mean():.4f}%) beyond ±{clip_limit}, in {extreme.any(axis=0).sum()} features and {extreme.any(axis=1).sum()} wells")
clipped = np.clip(raw, -clip_limit, clip_limit)

profiles = np.empty_like(clipped)
for time, line in panels:
    rows = meta.filter(panel_filter(time, line))["row"].to_numpy()
    dmso = clipped[meta.filter(panel_filter(time, line) & (pl.col("agent") == "DMSO"))["row"].to_numpy()]
    centre = np.median(dmso, axis=0)
    spread = 1.4826 * np.median(np.abs(dmso - centre), axis=0)
    # a few features hardly vary between DMSO wells; a floor stops them from blowing up
    spread = np.maximum(spread, np.quantile(spread[spread > 0], 0.01))
    profiles[rows] = (clipped[rows] - centre) / spread

extreme = np.abs(profiles) > clip_limit
print(f"rescaled: {100 * extreme.mean():.2f}% of values beyond ±{clip_limit} (clipped), in {extreme.any(axis=0).sum()} features")
profiles = np.clip(profiles, -clip_limit, clip_limit)

# distance of each well from the DMSO centre (the centre is 0 after rescaling)
meta = meta.with_columns(pl.Series("distance", np.linalg.norm(profiles, axis=1)))


# **Result: PASS.**
# 
# - The data hold 1,140 wells × 1,399 features.
# - 4 wells fail well QC, all on the 24h plate in CL1 with fewer than 100 cells: 2 DMSO wells (L04, N10), and one well each of CBK175924 0.67 mM (N03) and 2 mM (N08). We leave them out, so 1,136 wells remain. These two CBK175924 treatments keep 2 wells.
# - Only 48 input values (0.003%, in 14 features, largest 44) were beyond ±20, and we clipped them.
# - After the rescaling, 0.08% of the values were beyond ±20, and we clipped them.

# ## 2. Do the replicate wells agree, and which treatments differ from DMSO?
# 
# We use two standard readouts. Section 2a asks whether the 3 replicate wells of a treatment agree with each other. Section 2b asks whether they also differ from DMSO. Section 2c puts the two readouts side by side, with the DMSO-only checks.
# 
# ### 2a. Percent replicating
# 
# 1. **Replicate correlation:** for each treatment, plate and cell line, the median of the 3 pairwise Pearson correlations between its 3 wells. When well QC left only 2 wells, we use their single correlation, and compare it with a null of 2 wells.
# 2. **Null:** 10,000 times, we draw 3 wells from 3 *different* treatments on the same plate and cell line (DMSO counts as one treatment).
# 3. **Call:** a treatment "replicates" when its replicate correlation is above the 95th percentile of the null.
# 4. **Negative control:** sets of 3 random DMSO wells should beat the threshold only about 5% of the time.
# 
# **How to read the figure:** each panel shows the null on top (grey). Below it, each row is one agent, with one dot per dose (light and small = low dose, dark and large = high dose). A dot right of the dashed line, with a black outline, is a treatment whose wells agree more than chance.
# 
# **Pass if:** DMSO sets pass about 5% of the time, and treatments pass more often than that.

# In[6]:


def median_pair_corr(corr, wells):
    # median of the pairwise correlations between the wells (3 pairs for 3 wells, 1 pair for 2)
    wells = np.asarray(wells)
    a, b = np.triu_indices(len(wells), 1)
    return np.median(corr[wells[a], wells[b]])


replicate_rows, null_scores, dmso_scores, corr_thresholds = [], {}, {}, {}
for time, line in panels:
    panel = meta.filter(panel_filter(time, line))
    corr = np.corrcoef(profiles[panel["row"].to_numpy()])
    labels = panel["Metadata_Perturbation"].to_numpy()
    wells_of = {p: np.flatnonzero(labels == p) for p in np.unique(labels)}
    groups = list(wells_of)

    # null: wells of different perturbations, so they are never replicates; one null per group size
    # (3 wells, or 2 wells for a treatment that lost a well to well QC)
    thresholds = {}
    for size in (2, 3):
        size_null = np.array([
            median_pair_corr(corr, [rng.choice(wells_of[p]) for p in rng.choice(groups, size, replace=False)])
            for _ in range(n_null)
        ])
        thresholds[size] = np.quantile(size_null, null_quantile)
        if size == 3:
            null = size_null
    threshold = thresholds[3]
    # negative control: random sets of 3 DMSO wells
    dmso = np.array([median_pair_corr(corr, rng.choice(wells_of["DMSO"], 3, replace=False)) for _ in range(n_null)])
    null_scores[(time, line)], dmso_scores[(time, line)], corr_thresholds[(time, line)] = null, dmso, threshold

    for perturbation in groups:
        if perturbation == "DMSO":
            continue
        n_wells = len(wells_of[perturbation])
        # a treatment with a single well left cannot be tested for reproducibility
        score = median_pair_corr(corr, wells_of[perturbation]) if n_wells >= 2 else None
        replicate_rows.append({
            "time_point": time, "cell_line": line, "perturbation": perturbation, "n_wells": n_wells,
            "wells": ", ".join(panel["Metadata_Well"].to_numpy()[wells_of[perturbation]]),
            "replicate_corr": score, "corr_threshold": thresholds.get(n_wells),
            "replicating": bool(score > thresholds[n_wells]) if score is not None else None,
        })

scores = pl.DataFrame(replicate_rows).with_columns(
    pl.col("time_point").cast(pl.Enum(plate_order)),
    pl.col("perturbation").str.split("_").list.first().alias("agent"),
    pl.col("perturbation").str.split("_").list.get(1).alias("dose"),
).with_columns(
    pl.when(pl.col("agent").is_in(ligands)).then(pl.lit("ligand")).otherwise(pl.lit("compound")).alias("well_type"),
    pl.col("dose").replace_strict(dose_level).alias("dose_level"),
)

for time, line in panels:
    sub = scores.filter((pl.col("time_point") == time) & (pl.col("cell_line") == line))
    print(
        f"{time:>3} {line}: threshold r = {corr_thresholds[(time, line)]:.2f}, "
        f"{sub['replicating'].sum()}/{sub['replicating'].count()} treatments replicate ({100 * sub['replicating'].mean():.0f}%), "
        f"DMSO sets pass {100 * (dmso_scores[(time, line)] > corr_thresholds[(time, line)]).mean():.1f}%"
    )
print(f"all plates: {scores['replicating'].sum()}/{scores['replicating'].count()} ({100 * scores['replicating'].mean():.0f}%); not testable (1 well left): {scores['replicating'].null_count()}")


# In[10]:


# figure: replicate correlation of each treatment against the null, per plate and cell line
# top of each panel: the null; below it: one row per agent, one dot per dose
dose_sizes = [6, 14, 26]
dose_shades = {"compound": ["#a8e3cc", "#1baf7a", "#0b5e40"], "ligand": ["#f4b6b6", "#d03b3b", "#7a1a1a"]}
# within an agent row, low dose sits on top and high dose at the bottom
dose_offsets = [-0.25, 0.0, 0.25]
bins = np.linspace(-1, 1, 41)
fig = plt.figure(figsize=(7.2, 6.4))
outer = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.08)
first_hist = first_dots = None
for j, time in enumerate(plate_order):
    for i, line in enumerate(line_order):
        inner = outer[i, j].subgridspec(2, 1, height_ratios=[1, 1.35], hspace=0.06)
        ax_hist = fig.add_subplot(inner[0], sharex=first_hist, sharey=first_hist)
        ax_dots = fig.add_subplot(inner[1], sharex=ax_hist, sharey=first_dots)
        first_hist, first_dots = first_hist or ax_hist, first_dots or ax_dots
        threshold = corr_thresholds[(time, line)]
        sub = scores.filter((pl.col("time_point") == time) & (pl.col("cell_line") == line))

        # top: the null, the DMSO check and the threshold
        ax_hist.hist(null_scores[(time, line)], bins=bins, density=True, color="#d9d9d9")
        ax_hist.hist(dmso_scores[(time, line)], bins=bins, density=True, histtype="step", color="#595959", lw=0.8)
        ax_hist.axvline(threshold, ls="--", color="black", lw=0.6)
        ax_hist.text(threshold + 0.03, 0.95, "replicates →", transform=ax_hist.get_xaxis_transform(), va="top", fontsize=5.5)
        ax_hist.set_title(f"{time}, {line}: {sub['replicating'].sum()} of {sub['replicating'].count()} replicate")
        ax_hist.tick_params(labelbottom=False)
        if j == 0:
            ax_hist.set_ylabel("density")
        else:
            ax_hist.tick_params(labelleft=False)

        # bottom: one row per agent, one dot per dose
        ax_dots.axvspan(threshold, 1.05, color="#f2f2f2", zorder=0)
        ax_dots.axvline(threshold, ls="--", color="black", lw=0.6)
        for k in range(len(agent_order) - 1):
            ax_dots.axhline(k + 0.5, color="#bdbdbd" if agent_order[k + 1] in ligands and agent_order[k] not in ligands else "#e6e6e6", lw=0.5, zorder=1)
        for r in sub.filter(pl.col("replicate_corr").is_not_null()).iter_rows(named=True):
            level = r["dose_level"]
            ax_dots.scatter(
                r["replicate_corr"], agent_order.index(r["agent"]) + dose_offsets[level],
                s=dose_sizes[level], color=dose_shades[r["well_type"]][level], marker="o" if r["n_wells"] == 3 else "s",
                edgecolor="black" if r["replicating"] else "white", lw=0.7 if r["replicating"] else 0.3, zorder=3,
            )
        ax_dots.set_yticks(range(len(agent_order)), agent_order)
        ax_dots.set_ylim(len(agent_order) - 0.5, -0.5)
        ax_dots.tick_params(axis="y", length=0, labelleft=j == 0)
        ax_dots.set_xlim(-1.05, 1.05)
        if i == 1 and j == 1:
            ax_dots.set_xlabel("replicate correlation (median Pearson r of the 3 wells)")

line_handles = [
    Patch(color="#d9d9d9", label="null: 3 wells of different treatments"),
    Line2D([], [], color="#595959", lw=0.8, label="3 random DMSO wells"),
    Line2D([], [], color="black", ls="--", lw=0.6, label="95th percentile of the null"),
    Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", markersize=4, label="black outline: replicates"),
]
if (scores["n_wells"] == 2).any():
    line_handles.append(Line2D([], [], marker="s", ls="", color="#9e9e9e", markersize=3.5, label="square: 2 wells left after well QC"))
dose_handles = [
    Line2D([], [], marker="o", ls="", color=dose_shades[kind][level], markersize=np.sqrt(dose_sizes[level]), label=dose_text[dose])
    for kind, doses in (("compound", ["0.2", "0.67", "2.0"]), ("ligand", ["L", "M", "H"]))
    for level, dose in enumerate(doses)
]
fig.legend(handles=line_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.legend(
    handles=dose_handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12),
    title="dose: compounds (stock concentration, green) and ligands (red)",
)
fig.suptitle("Do replicate wells agree more than random wells?", y=0.95)
save_figure(fig, "01_replicate-correlation-vs-null")


# In[11]:


# which doses replicate, per agent, plate and cell line ("–" = no dose)
panel_columns = [f"{time} {line}" for line in line_order for time in plate_order]
replicating_doses = (
    scores.sort("dose_level")
    .with_columns(
        (pl.col("time_point").cast(pl.Utf8) + " " + pl.col("cell_line")).alias("panel"),
        pl.col("dose").replace_strict(dose_text).alias("dose_text"),
    )
    .group_by("agent", "panel", maintain_order=True)
    .agg(pl.col("dose_text").filter(pl.col("replicating").fill_null(False)).str.join(", ").alias("doses"))
    .with_columns(pl.when(pl.col("doses") == "").then(pl.lit("–")).otherwise(pl.col("doses")).alias("doses"))
    .pivot(on="panel", index="agent", values="doses")
    .select("agent", *panel_columns)
    .sort(pl.col("agent").replace_strict({a: i for i, a in enumerate(agent_order)}))
)
replicating_doses.write_csv(table_dir / "replicating_doses.csv")
replicating_doses


# In[12]:


# the wells behind each replicating treatment
scores.filter(pl.col("replicating")).sort(
    "cell_line", "time_point", pl.col("perturbation").replace_strict({p: i for i, p in enumerate(perturbation_order)})
).select(
    "time_point", "cell_line", pl.col("perturbation").map_elements(pretty, return_dtype=pl.Utf8).alias("treatment"),
    pl.col("replicate_corr").round(2), pl.col("corr_threshold").round(2), "wells",
)


# **Result of 2a:**
# 
# - Percent replicating rises with time (see 2c for the numbers).
# - **Doses:** CBK600005 replicates at all 3 doses on every plate. CBK041498, CBK175924 and CBK600008 replicate only at 2 mM. CBK200566 (CL2, 4h and 24h) and TGFβ1 (CL2, 24h) also replicate at lower doses. The table above lists the 3 wells of each replicating set (also in `replicating_doses.csv`).
# - CBK175924 2 mM in CL1 at 24h lost a well to well QC. Its 2 remaining wells (r = 0.55) fall just below the stricter threshold for 2 wells (0.61).
# - TGFβ1 10 ng/ml has the highest score in CL1 at 1h (r = 0.95). But its 3 wells (C02, E01, E06) lie in odd rows of columns 1–6, the shifted area of section 3a. So this score most likely comes from the plate effect, not from TGFβ1.

# ### 2b. mAP against DMSO
# 
# We use `mantispy.tl.map` (activity mode, cosine distance) on the rescaled profiles.
# 
# 1. Each replicate well ranks the other wells of its plate and cell line. Only its replicates and the DMSO wells enter this ranking.
# 2. 100,000 permutations give a p-value. We correct the p-values of all 126 tests together (Benjamini–Hochberg), and call a treatment **active** at q < 0.05.
# 3. A treatment with only one well left after well QC cannot get a score.
# 
# **Negative control:** on each plate and cell line, we treat 20 random sets of 3 DMSO wells as fake treatments. We score them against the remaining DMSO wells. About 5% should reach p < 0.05, and none should pass q < 0.05.
# 
# **How to read the figure:** each row is one treatment and dose, and each panel is one cell line and plate. The x-axis is the mAP against DMSO. Filled dots are active (q < 0.05), open dots are not. The grey band runs from 0 to the highest mAP of the 20 fake DMSO treatments on that plate.
# 
# The grey band is a visual chance level, not the test. The q-values come from the permutation test. Each fake treatment is ranked against the ~67 DMSO wells that remain after we pick the 60 fake wells. A real treatment is ranked against all ~127 DMSO wells. With fewer reference wells, a high mAP comes more easily by chance, so the band sits a little higher than the true chance level of a real treatment.

# In[7]:


# mAP needs an AnnData object: the rescaled profiles, and one label per perturbation x plate x cell line
map_data = adata.copy()
map_data.X = profiles.astype(np.float32)
map_data.obs["Metadata_panel"] = (
    map_data.obs["Metadata_time_point"].astype(str) + "_" + map_data.obs["Metadata_cell_line"].astype(str)
).to_numpy()
map_data.obs["Metadata_query"] = (map_data.obs["Metadata_Perturbation"].astype(str) + "|" + map_data.obs["Metadata_panel"]).to_numpy()
mt.tl.map(map_data, mode="activity", label="Metadata_query", batch="Metadata_panel", null_size=100_000, seed=seed, n_jobs=n_threads)


def read_map(result):
    # split the "perturbation|time_cellline" label back into columns
    return pl.from_pandas(result).select(
        pl.col("Metadata_query").str.split("|").list.first().alias("perturbation"),
        pl.col("Metadata_query").str.split("|").list.last().str.split("_").list.first().alias("time_point"),
        pl.col("Metadata_query").str.split("|").list.last().str.split("_").list.last().alias("cell_line"),
        pl.col("score").alias("mAP"),
        pl.col("pvalue").alias("mAP_pvalue"),
        pl.col("qvalue").alias("mAP_qvalue"),
    )


map_scores = read_map(map_data.uns["mantispy"]["map"]).with_columns(pl.col("time_point").cast(pl.Enum(plate_order)))
scores = scores.join(map_scores, on=["time_point", "cell_line", "perturbation"], how="left").with_columns(
    (pl.col("mAP_qvalue") < fdr).fill_null(False).alias("active")
)
print(f"active (mAP q < {fdr}): {scores['active'].sum()}/{scores.height}")

# negative control: fake treatments made of random DMSO wells
dmso_data = map_data[map_data.obs["Metadata_Control_Type"].astype(str) == "negcon"].copy()
query = dmso_data.obs["Metadata_query"].astype(str).to_numpy().copy()
panel_names = dmso_data.obs["Metadata_panel"].to_numpy()
fake = np.zeros(dmso_data.n_obs, dtype=bool)
for panel_name in np.unique(panel_names):
    picked = rng.permutation(np.flatnonzero(panel_names == panel_name))[:60]
    for k in range(20):
        query[picked[3 * k : 3 * k + 3]] = f"fakeDMSO-{k}|{panel_name}"
    fake[picked] = True
dmso_data.obs["Metadata_query"] = query
dmso_data.obs["Metadata_Control_Type"] = np.where(fake, "treatment", "negcon")
mt.tl.map(dmso_data, mode="activity", label="Metadata_query", batch="Metadata_panel", null_size=100_000, seed=seed, n_jobs=n_threads)
fake_scores = read_map(dmso_data.uns["mantispy"]["map"])
print(
    f"fake DMSO treatments: {(fake_scores['mAP_pvalue'] < 0.05).sum()}/{fake_scores.height} with p < 0.05 "
    f"({100 * (fake_scores['mAP_pvalue'] < 0.05).mean():.1f}%), {(fake_scores['mAP_qvalue'] < fdr).sum()} with q < {fdr}; "
    f"median mAP {fake_scores['mAP'].median():.3f}"
)


# In[8]:


# figure: mAP against DMSO for every treatment and dose, per plate and cell line
# grey band = chance level: the mAP range of the 20 fake treatments made of random DMSO wells (same panel)
y_of = {p: i for i, p in enumerate(perturbation_order)}
fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.6), sharex=True, sharey=True, layout="constrained")
for i, line in enumerate(line_order):
    for j, time in enumerate(plate_order):
        ax = axes[i, j]
        sub = scores.filter((pl.col("time_point") == time) & (pl.col("cell_line") == line))
        fake = fake_scores.filter((pl.col("time_point") == time) & (pl.col("cell_line") == line))["mAP"].to_numpy()
        ax.axvspan(0, fake.max(), color="#e6e6e6", lw=0, zorder=0)
        y = np.array([y_of[p] for p in sub["perturbation"]])
        x = sub["mAP"].to_numpy()
        colors = np.array([well_type_colors[w] for w in sub["well_type"]])
        active = sub["active"].to_numpy()
        # filled = active (q < 0.05), hollow = not active
        ax.scatter(x[active], y[active], s=14, c=colors[active], linewidths=0, zorder=3)
        ax.scatter(x[~active], y[~active], s=14, facecolors="white", edgecolors=colors[~active], linewidths=0.8, zorder=3)
        ax.set_title(f"{line}, {time}  ({active.sum()} of {sub.height} active)")
        ax.grid(axis="y", color="#f0f0f0", lw=0.5, zorder=0)
        if i == 1:
            ax.set_xlabel("mAP against DMSO")
axes[0, 0].set_yticks(range(len(perturbation_order)), [pretty(p) for p in perturbation_order])
axes[0, 0].invert_yaxis()
axes[0, 0].set_xlim(-0.02, 1.02)
for ax in axes.flat:
    ax.tick_params(axis="y", labelsize=5.5)
handles = [
    Patch(color="#e6e6e6", label="chance: mAP of fake DMSO treatments"),
    Line2D([], [], marker="o", ls="", color=well_type_colors["compound"], markersize=4, label="compound, active (q < 0.05)"),
    Line2D([], [], marker="o", ls="", color=well_type_colors["ligand"], markersize=4, label="ligand, active (q < 0.05)"),
    Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", markersize=4, label="not active"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncols=4)
fig.suptitle("Which treatments differ from DMSO? mAP of each treatment and dose")
save_figure(fig, "02b_map-against-dmso")


# **Result of 2b:**
# 
# - CBK600005 scores near 1 at every dose, time and cell line.
# - TGFβ1 becomes active at 4h in CL2, and at 24h in both cell lines. CBK041498 and CBK175924 are active mainly at 2 mM. CBK200566 is active at all doses in CL2 at 24h.
# - IL6 and CBK600008 stay inside or near the grey band, and are never active.
# - The grey band is widest in CL1 at 24h, the plate with uneven cell numbers.

# ### 2c. The two readouts side by side, and the DMSO checks
# 
# The table and figure below give, per plate and cell line, the share of treatments that replicate (2a) and that are active (2b). The right panel shows the DMSO-only checks of both readouts. About 5% should pass by chance.

# In[9]:


# one summary row per plate and cell line
panel_summary = []
for time, line in panels:
    sub = scores.filter((pl.col("time_point") == time) & (pl.col("cell_line") == line))
    fake_sub = fake_scores.filter((pl.col("time_point") == time) & (pl.col("cell_line") == line))
    panel_summary.append({
        "time_point": time, "cell_line": line, "treatments": sub.height, "treatments_with_3_wells": int((sub["n_wells"] == 3).sum()),
        "corr_threshold": corr_thresholds[(time, line)],
        "percent_replicating": 100 * sub["replicating"].mean(),
        "dmso_sets_passing_corr_pct": 100 * (dmso_scores[(time, line)] > corr_thresholds[(time, line)]).mean(),
        "percent_active_mAP": 100 * sub["active"].mean(),
        "fake_dmso_p05_pct": 100 * (fake_sub["mAP_pvalue"] < 0.05).mean(),
    })
panel_summary = pl.DataFrame(panel_summary)
panel_summary.write_csv(table_dir / "panel_summary.csv")
panel_summary.with_columns(pl.selectors.float().round(1))


# In[13]:


# figure: summary of reproducibility and activity, with the DMSO checks
summary_pd = panel_summary.to_pandas()
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained")
x = np.arange(len(plate_order))
width = 0.38
for k, line in enumerate(line_order):
    sub = summary_pd[summary_pd["cell_line"] == line].set_index("time_point").loc[plate_order]
    offset = (k - 0.5) * width
    axes[0].bar(x + offset, sub["percent_replicating"], width, color=line_colors[line], label=line)
    axes[1].bar(x + offset, sub["percent_active_mAP"], width, color=line_colors[line], label=line)
    axes[2].bar(x + offset, sub["dmso_sets_passing_corr_pct"], width, color=line_colors[line], alpha=0.45, label=f"{line}: random DMSO sets, correlation")
    axes[2].scatter(x + offset, sub["fake_dmso_p05_pct"], marker="D", s=14, color=line_colors[line], edgecolor="black", linewidths=0.5, zorder=3, label=f"{line}: fake DMSO treatments, mAP p < 0.05")
    for ax, column in zip(axes[:2], ["percent_replicating", "percent_active_mAP"]):
        for xi, value in zip(x + offset, sub[column]):
            ax.text(xi, value + 1, f"{value:.0f}", ha="center", va="bottom", fontsize=6)
titles = [
    "Percent replicating\n(replicate correlation above the null)",
    "Percent active\n(mAP against DMSO, q < 0.05)",
    "Negative control check\n(DMSO wells only)",
]
for ax, title in zip(axes, titles):
    ax.set_xticks(x, plate_order)
    ax.set_xlabel("time after treatment")
    ax.set_title(title)
    ax.axhline(5, ls="--", color="black", lw=0.6)
axes[0].set_ylabel("% of treatments")
axes[1].set_ylabel("% of treatments")
axes[2].set_ylabel("% passing (expected 5%, dashed)")
axes[0].set_ylim(0, 100)
axes[1].set_ylim(0, 100)
axes[2].set_ylim(0, 20)
axes[0].legend(title="cell line", loc="upper left")
axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncols=1, fontsize=5.5)
save_figure(fig, "02_reproducibility-summary")


# **Result:**
# 
# - **Percent replicating (2a)** rises with time: 19% (4 of 21) in both cell lines at 1h, 33% (CL1) and 43% (CL2) at 4h, and 33% (CL1) and 57% (CL2) at 24h. Overall, 43 of 126 tests (34%) replicate.
# - **mAP (2b)** calls 40 of 126 tests active: 14% / 19% at 1h, 19% / 43% at 4h, and 43% / 52% at 24h (CL1 / CL2).
# - The two readouts mostly agree. 6 tests replicate without being active, and 3 tests are active without replicating.
# - **DMSO checks:** random DMSO sets pass the correlation threshold 4.6–4.9% of the time (8.4% in CL2 at 4h). 4 of 120 fake DMSO treatments (3.3%) reach mAP p < 0.05, and none reaches q < 0.05. 3 of these 4 are in CL1 at 24h, the plate with uneven cell numbers.
# - So both readouts behave as expected on DMSO wells, and the activity calls are not chance.

# ## 3. Replicate health
# 
# ### 3a. Are the DMSO wells alike across the plate?
# 
# DMSO wells should all look alike, wherever they sit on the plate.
# 
# - For each well, we measure its **distance from the DMSO centre**: the Euclidean length of its rescaled profile. 0 is the typical DMSO well.
# - A DMSO well is **unusual** when its distance is above the median + 3 MAD of all DMSO wells of that plate and cell line.
# 
# Notebook 01 (`03_plate-maps_dmso-distance.png`) draws these wells on the plate, and tests which technical factors explain them. Here, we use them to flag treatments that may look active only because of where their wells sit.
# 
# **Pass if:** unusual DMSO wells are few and scattered, not grouped in one area of the plate.

# In[14]:


dmso_rows, unusual_profile, distance_limits = [], {}, {}
for time, line in panels:
    dmso = meta.filter(panel_filter(time, line) & (pl.col("agent") == "DMSO"))
    d = dmso["distance"].to_numpy()
    limit = np.median(d) + unusual_mads * 1.4826 * np.median(np.abs(d - np.median(d)))
    distance_limits[(time, line)] = limit
    unusual = dmso.filter(pl.col("distance") > limit)
    normal = dmso.filter(pl.col("distance") <= limit)
    # the shared profile of the unusual DMSO wells, used to spot treatments that look like them
    unusual_profile[(time, line)] = np.median(profiles[unusual["row"].to_numpy()], axis=0) if unusual.height >= 3 else None
    in_odd_rows = unusual.filter(pl.col("Metadata_Row").is_in(list("ACEGIKMO"))).height
    corr_unusual = np.corrcoef(profiles[unusual["row"].to_numpy()]) if unusual.height >= 3 else None
    dmso_rows.append({
        "time_point": time, "cell_line": line, "dmso_wells": dmso.height, "unusual_wells": unusual.height,
        "unusual_in_odd_rows": in_odd_rows,
        "unusual_columns": ", ".join(sorted(unusual["Metadata_Column"].unique().to_list())),
        "distance_limit": limit,
        "median_cells_unusual": unusual["Metadata_CellCount"].median(),
        "median_cells_normal": normal["Metadata_CellCount"].median(),
        "median_corr_among_unusual": np.median(corr_unusual[np.triu_indices(unusual.height, 1)]) if corr_unusual is not None else None,
    })
dmso_health = pl.DataFrame(dmso_rows)
dmso_health.write_csv(table_dir / "dmso_well_health.csv")
dmso_health.with_columns(pl.selectors.float().round(2))


# **Result: FAIL. The unusual DMSO wells group in two kinds of plate area** (plate maps and tests in notebook 01).
# 
# - In CL1 at 1h, 33 of 127 DMSO wells (26%) are unusual, all in the odd rows (A, C, …, O) of columns 1–6. They share one profile (median r = 0.91 between them), so they shifted together.
# - The same pattern is weaker at 4h (11 wells, odd rows, columns 1–5), and gone at 24h. A pattern that follows every other row is technical (most likely liquid handling), not biological.
# - At 24h, the unusual DMSO wells are the low-count wells: rows M–N in both cell lines, columns 3–4 (rows F–M) in CL1, and columns 20–22 (rows J–N) in CL2. These are 19 wells in CL1 and 22 wells in CL2.
# - These 24h wells have fewer QC-passing cells: median 319 vs 1,670 in CL1, and 2,546 vs 3,251 in CL2. The 2 emptiest DMSO wells of CL1 (L04, N10) already failed well QC.
# - CL2 at 1h and 4h is nearly clean (2 and 7 unusual DMSO wells). At 4h, most of them sit at the right edge (columns 23–24).
# - Treated wells in these areas take on the same shift, so a treatment can look active only because of where its wells sit. Below, we flag treatments and wells that correlate r ≥ 0.8 with the shared profile of the unusual DMSO wells.

# ### 3b. Outlier replicates
# 
# For each treatment with 3 wells, we check whether one well disagrees with the other two. A well is an **outlier replicate** in two cases:
# 
# - **direction:** its two sibling wells correlate with each other above chance, but the well correlates with neither of them above chance. Chance is the 95th percentile of the correlation between two wells of different treatments.
# - **strength:** the well lies beyond the unusual-well limit of the DMSO wells (3a, dashed line in the figure) while both siblings lie below it, or the reverse.
# 
# For each well, we also give its cell count, its plate position, and its correlation with the unusual DMSO wells of 3a ("looks like unusual DMSO" when r ≥ 0.8). These help to tell a technical problem from biology.
# 
# **How to read the figure:** the grey band holds the middle 90% of the usual DMSO wells, and the grey line marks the median DMSO well.

# In[15]:


health_rows = []
for time, line in panels:
    panel = meta.filter(panel_filter(time, line))
    rows = panel["row"].to_numpy()
    corr = np.corrcoef(profiles[rows])
    labels = panel["Metadata_Perturbation"].to_numpy()
    wells_of = {p: np.flatnonzero(labels == p) for p in np.unique(labels)}
    groups = list(wells_of)
    # chance level for one pair: two wells of two different perturbations
    pair_null = np.array([corr[rng.choice(wells_of[a]), rng.choice(wells_of[b])] for a, b in (rng.choice(groups, 2, replace=False) for _ in range(n_null))])
    pair_limit = np.quantile(pair_null, null_quantile)
    distance = panel["distance"].to_numpy()
    far = distance > distance_limits[(time, line)]
    cells = panel["Metadata_CellCount"].to_numpy()
    artifact = unusual_profile[(time, line)]
    like_unusual = [float(np.corrcoef(profiles[r], artifact)[0, 1]) for r in rows] if artifact is not None else [None] * len(rows)
    dmso_cells = np.median(cells[wells_of["DMSO"]])

    for perturbation in groups:
        if perturbation == "DMSO":
            continue
        trio = wells_of[perturbation]
        # the outlier rules need 3 wells
        if len(trio) != 3:
            continue
        for k, w in enumerate(trio):
            s1, s2 = np.delete(trio, k)
            own = (corr[w, s1], corr[w, s2])
            disagrees = corr[s1, s2] > pair_limit and max(own) < pair_limit
            odd_strength = far[w] != far[s1] and far[s1] == far[s2]
            reason = "; ".join(
                text for flag, text in [
                    (disagrees, "direction: siblings agree, this well does not"),
                    (odd_strength and far[w], "strength: only this well is outside the DMSO range"),
                    (odd_strength and not far[w], "strength: only this well is inside the DMSO range"),
                ] if flag
            )
            health_rows.append({
                "time_point": time, "cell_line": line, "perturbation": perturbation,
                "well": panel["Metadata_Well"][int(w)],
                "sibling_wells": ", ".join(panel["Metadata_Well"][int(s)] for s in (s1, s2)),
                "corr_with_siblings": float(np.mean(own)), "corr_between_siblings": float(corr[s1, s2]), "pair_chance_limit": float(pair_limit),
                "distance": float(distance[w]), "sibling_distances": f"{distance[s1]:.0f}, {distance[s2]:.0f}",
                "dmso_distance_limit": float(distance_limits[(time, line)]),
                "cells": int(cells[w]), "sibling_cells": f"{cells[s1]}, {cells[s2]}", "dmso_median_cells": float(dmso_cells),
                "corr_with_unusual_dmso": like_unusual[w],
                "outlier": bool(disagrees or odd_strength), "reason": reason,
            })

health = pl.DataFrame(health_rows).with_columns(pl.col("time_point").cast(pl.Enum(plate_order)))
outliers = health.filter(pl.col("outlier")).sort("time_point", "cell_line", "perturbation")
outliers.write_csv(table_dir / "outlier_replicates.csv")
health.write_csv(table_dir / "replicate_well_health.csv")

sets_with_outlier = outliers.select("time_point", "cell_line", "perturbation").unique().height
print(f"outlier replicate wells: {outliers.height} of {health.height} treated wells in sets of 3; replicate sets with an outlier: {sets_with_outlier} of {scores.filter(pl.col('n_wells') == 3).height}")
print("outliers per plate and cell line:", outliers.group_by("time_point", "cell_line").len().sort("time_point", "cell_line").rows())
print(f"outliers that look like unusual DMSO wells (r >= {artifact_corr}): {(outliers['corr_with_unusual_dmso'] >= artifact_corr).fill_null(False).sum()}")
outliers.select(
    "time_point", "cell_line", "perturbation", "well", "sibling_wells",
    pl.col("corr_with_siblings").round(2), pl.col("corr_between_siblings").round(2),
    pl.col("distance").round(0), "sibling_distances", "cells", "sibling_cells",
    pl.col("corr_with_unusual_dmso").round(2), "reason",
)


# In[16]:


# figure: the 3 replicate wells of each treatment, by distance from the DMSO centre
fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.6), sharey=True, sharex=True)
y_of = {p: i for i, p in enumerate(perturbation_order)}
for j, time in enumerate(plate_order):
    for i, line in enumerate(line_order):
        ax = axes[i, j]
        sub = health.filter((pl.col("time_point") == time) & (pl.col("cell_line") == line))
        # where the usual DMSO wells sit: the middle 90% of the DMSO wells below the unusual-well limit
        dmso_distance = meta.filter(panel_filter(time, line) & (pl.col("agent") == "DMSO"))["distance"].to_numpy()
        usual = dmso_distance[dmso_distance <= distance_limits[(time, line)]]
        ax.axvspan(*np.quantile(usual, [0.05, 0.95]), color="#e6e6e6", lw=0, zorder=0)
        ax.axvline(np.median(dmso_distance), color="#8c8c8c", lw=0.8, zorder=1)
        ax.axvline(distance_limits[(time, line)], color="black", lw=0.6, ls="--", zorder=1)
        y = np.array([y_of[p] for p in sub["perturbation"]]) + rng.uniform(-0.15, 0.15, sub.height)
        types = ["ligand" if p.split("_")[0] in ligands else "compound" for p in sub["perturbation"]]
        like = (sub["corr_with_unusual_dmso"] >= artifact_corr).fill_null(False).to_numpy()
        ax.scatter(
            sub["distance"], y, s=9, c=[well_type_colors[t] for t in types],
            edgecolor=np.where(like, "black", "white"), lw=np.where(like, 0.8, 0.3), zorder=2,
        )
        flagged = sub["outlier"].to_numpy()
        ax.scatter(sub["distance"].to_numpy()[flagged], y[flagged], marker="x", s=16, color="black", lw=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_title(f"{line}, {time}")
        ax.grid(axis="y", color="#f0f0f0", lw=0.5, zorder=0)
        if i == 1:
            ax.set_xlabel("distance from DMSO centre (log)")
ax.set_xlim(12, 600)
for ax in axes.flat:
    ax.set_xticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
    ax.minorticks_off()
    ax.tick_params(axis="y", labelsize=5.5)
axes[0, 0].set_yticks(range(len(perturbation_order)), [pretty(p) for p in perturbation_order])
axes[0, 0].invert_yaxis()
handles = [
    Patch(color="#e6e6e6", label="middle 90% of the usual DMSO wells"),
    Line2D([], [], color="#8c8c8c", lw=0.8, label="median DMSO well"),
    Line2D([], [], color="black", lw=0.6, ls="--", label="limit for an unusual DMSO well (median + 3 MAD)"),
    Line2D([], [], marker="o", ls="", color=well_type_colors["compound"], label="compound well"),
    Line2D([], [], marker="o", ls="", color=well_type_colors["ligand"], label="ligand well"),
    Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.8, label=f"looks like unusual DMSO wells (r ≥ {artifact_corr})"),
    Line2D([], [], marker="x", ls="", color="black", markeredgewidth=0.8, label="outlier replicate"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Do the 3 replicate wells of each treatment agree?")
fig.tight_layout(rect=(0, 0.0, 1, 1))
save_figure(fig, "04_replicate-spread-and-outliers")


# **Result:**
# 
# - 48 of 372 treated wells (13%) are outlier replicates, in 46 of 124 replicate sets with 3 wells. 33 are outliers because of strength, and 21 because of direction (6 for both). 17 of the 48 look like the unusual DMSO wells (r ≥ 0.8).
# - CL1 at 1h has the most outliers (15). In 9 of them, one well lies in the shifted area of 3a while its two siblings do not. Examples are CBK041498 0.2 mM K02 and IL6 2.5 ng/ml K04 (r = 0.80–0.98 with the unusual DMSO wells).
# - For TGFβ1 2.5 ng/ml at 1h, it is the reverse. The two "agreeing" wells (A04, A07) sit in the shifted area, and the flagged well (N12) is the normal one. Plate position, not the treatment, explains these outliers.
# - At 24h, several outliers are wells with few cells in rows L–N, the band of unusual DMSO wells. In CL1 these are CBK200566 0.67 mM N04 (172 cells, siblings 1,311 and 1,137) and TGFβ1 2.5 ng/ml N12 (115 cells).
# - In CL2 these are CBK041498 0.67 mM N13 (274 cells, siblings 3,547 and 3,197) and CBK600005 0.67 mM L22 (644 cells). The emptiest well (CBK175924 2 mM N08, 62 cells) failed well QC, so it is already left out.
# - `outlier_replicates.csv` gives the full list with positions, cell counts and correlations. Consider leaving out these wells (at least the low-cell wells) in later analyses.

# ## 4. Phenotypic strength: how far is each treatment from DMSO?
# 
# **Strength** = the distance of the consensus profile (median of the replicate wells) from the DMSO centre, divided by the typical distance of a set of 3 DMSO wells from the other DMSO wells. The typical distance is the median of 2,000 random DMSO sets.
# 
# - A strength of 1 means "as far as 3 random DMSO wells". A strength of 3 means three times further.
# - The heatmap marks each **activity call** (mAP q < 0.05, section 2b) with *.
# - It marks with † each treatment whose consensus looks like the unusual DMSO wells of the same plate (r ≥ 0.8, section 3a).
# 
# **Pass if:** every active treatment (mAP) is further from DMSO than a typical set of 3 random DMSO wells (strength above 1).

# In[17]:


strength_rows, set_typical = [], {}
for time, line in panels:
    panel = meta.filter(panel_filter(time, line))
    dmso_rows = panel.filter(pl.col("agent") == "DMSO")["row"].to_numpy()
    # null: distance of the median of 3 DMSO wells from the median of the other DMSO wells
    set_distances = np.empty(n_null_sets)
    for k in range(n_null_sets):
        pick = rng.choice(len(dmso_rows), 3, replace=False)
        rest = np.delete(dmso_rows, pick)
        set_distances[k] = np.linalg.norm(np.median(profiles[dmso_rows[pick]], axis=0) - np.median(profiles[rest], axis=0))
    typical = np.median(set_distances)
    set_typical[(time, line)] = typical
    dmso_cells = panel.filter(pl.col("agent") == "DMSO")["Metadata_CellCount"].median()
    artifact = unusual_profile[(time, line)]
    for perturbation, group in panel.filter(pl.col("agent") != "DMSO").group_by("Metadata_Perturbation"):
        consensus = np.median(profiles[group["row"].to_numpy()], axis=0)
        strength_rows.append({
            "time_point": time, "cell_line": line, "perturbation": perturbation[0],
            "consensus_distance": float(np.linalg.norm(consensus)),
            "strength": float(np.linalg.norm(consensus) / typical),
            "beyond_95pct_dmso_sets": bool(np.linalg.norm(consensus) > np.quantile(set_distances, 0.95)),
            "cell_ratio": float(group["Metadata_CellCount"].median() / dmso_cells),
            "corr_with_unusual_dmso": float(np.corrcoef(consensus, artifact)[0, 1]) if artifact is not None else None,
        })
    print(f"{time:>3} {line}: typical distance of a DMSO set {typical:.1f}, 95th percentile {np.quantile(set_distances, 0.95):.1f}")

scores = scores.join(
    pl.DataFrame(strength_rows).with_columns(pl.col("time_point").cast(pl.Enum(plate_order))),
    on=["time_point", "cell_line", "perturbation"], how="left",
).with_columns((pl.col("corr_with_unusual_dmso") >= artifact_corr).fill_null(False).alias("looks_like_unusual_dmso"))


# In[18]:


# table: strength and activity of every treatment
strength_table = scores.with_columns(pl.col("perturbation").map_elements(pretty, return_dtype=pl.Utf8).alias("treatment")).sort(
    "cell_line", "time_point", pl.col("perturbation").replace_strict({p: i for i, p in enumerate(perturbation_order)})
)
active_not_artifact = scores.filter(pl.col("active") & ~pl.col("looks_like_unusual_dmso"))
print(f"active: {scores['active'].sum()} of {scores.height}; active and not like unusual DMSO: {active_not_artifact.height}")
print(f"look like unusual DMSO wells: {scores['looks_like_unusual_dmso'].sum()}, of which active: {scores.filter(pl.col('looks_like_unusual_dmso'))['active'].sum()}")
print(f"replicating but not active: {scores.filter(pl.col('replicating') & ~pl.col('active')).height}; active but not replicating: {scores.filter(~pl.col('replicating') & pl.col('active')).height}")
print("active per agent:", dict(scores.filter(pl.col("active")).group_by("agent").len().sort("agent").iter_rows()))
strength_table.filter(pl.col("active") | pl.col("looks_like_unusual_dmso")).select(
    "cell_line", "time_point", "treatment", pl.col("strength").round(1), pl.col("mAP").round(2), pl.col("mAP_qvalue").round(4),
    "replicating", pl.col("cell_ratio").round(2), pl.col("corr_with_unusual_dmso").round(2), "looks_like_unusual_dmso",
)


# In[19]:


# figure: strength heatmap per cell line (rows: treatment x dose, columns: time point)
fig, axes = plt.subplots(1, 2, figsize=(5.2, 5.4), sharey=True, layout="constrained")
strength_norm = LogNorm(vmin=0.5, vmax=float(scores["strength"].max()))
for ax, line in zip(axes, line_order):
    lookup = {(r["perturbation"], r["time_point"]): r for r in scores.filter(pl.col("cell_line") == line).iter_rows(named=True)}
    strength = np.array([[lookup[(p, t)]["strength"] for t in plate_order] for p in perturbation_order])
    # number, * when active, † when the treatment looks like the unusual DMSO wells, ° when well QC left fewer than 3 wells
    marks = np.array([
        [f"{lookup[(p, t)]['strength']:.1f}" + ("*" if lookup[(p, t)]["active"] else "") + ("†" if lookup[(p, t)]["looks_like_unusual_dmso"] else "")
         + ("°" if lookup[(p, t)]["n_wells"] < 3 else "") for t in plate_order]
        for p in perturbation_order
    ])
    sns.heatmap(strength, ax=ax, cmap=blues, norm=strength_norm, annot=marks, fmt="", cbar=False, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 6})
    ax.set_title(line)
    ax.set_xticklabels(plate_order, rotation=0)
    ax.set_xlabel("time point")
    ax.tick_params(length=0)
    # separate the agents with white lines
    for boundary in range(3, len(perturbation_order), 3):
        ax.axhline(boundary, color="white", lw=2)
axes[0].set_yticks(np.arange(len(perturbation_order)) + 0.5, [pretty(p) for p in perturbation_order], rotation=0)
strength_bar = fig.colorbar(plt.cm.ScalarMappable(norm=strength_norm, cmap=blues), ax=axes, location="bottom", shrink=0.7, aspect=30)
strength_bar.set_ticks([0.5, 1, 2, 4, 8], labels=["0.5", "1", "2", "4", "8"])
strength_bar.minorticks_off()
strength_bar.set_label("strength: distance from DMSO / typical distance of 3 DMSO wells (log scale)\n* active (mAP q < 0.05)   † looks like unusual DMSO wells   ° 2 wells after well QC")
save_figure(fig, "05_phenotypic-strength-heatmap")


# **Result: PASS.** All 40 active treatments are further from DMSO than a typical set of 3 random DMSO wells (strength 1.2–9.7). 33 of them lie beyond the 95th percentile of the random DMSO sets.
# 
# - 40 of 126 tests are active (mAP q < 0.05): CBK600005 18 of 18, TGFβ1 9, CBK200566 6, CBK041498 4, CBK175924 3, CBK600008 0 and IL6 0 of 18.
# - **CBK600005** is the strongest treatment. It is active at every dose, time and cell line (strength 3.3–9.7), and strongest in CL2 at 24h (8.5–9.7).
# - At 24h, CBK600005 also lowers the number of QC-passing cells to 35–54% (CL1) and 38–47% (CL2) of DMSO (`cell_ratio` in the table above). So part of its effect may be toxicity. Notebook 04 tests cell loss.
# - **TGFβ1** is active at all 3 doses in both cell lines at 24h (strength 2.3–3.4), and in CL2 already at 4h (2.5 and 10 ng/ml). In CL1 at 24h, it also lowers the number of QC-passing cells to 45–51% of DMSO.
# - The high TGFβ1 strength in CL1 at 1h (3.6 and 5.1) and 4h (4.4) comes with † (r = 0.91–0.98 with the unusual DMSO wells). So it is mostly the plate-position effect. The only active test with † is TGFβ1 10 ng/ml at 4h in CL1.
# - **IL6** is never active (strength 0.7–1.6). **CBK600008** is never active either (0.7–1.3).
# - **CBK200566** is active at all doses in CL2 at 24h (3.5–6.5, cells 54–73% of DMSO), and at 2 mM in CL1 at 24h (3.2).
# - **CBK041498** and **CBK175924** are active mainly at 2 mM at 24h, and in CL2 also at 4h. CBK175924 2 mM in CL2 is active already at 1h.

# ## 5. Dose and time
# 
# The strength heatmap above shows each agent across the 3 doses and 3 time points. To test for a **dose response**, we do three steps per plate and cell line:
# 
# 1. Take the distance from the DMSO centre of each of the 9 wells of an agent (3 doses × 3 replicates, fewer when a well failed QC).
# 2. Correlate these distances with the dose rank (Spearman ρ).
# 3. Correct for the 42 tests (Benjamini–Hochberg).
# 
# The data cannot separate time from the exposure settings. So a trend in time can also come from the longer ER and AGP exposures of the 24h plate.

# In[20]:


dose_rows = []
for time, line in panels:
    panel = meta.filter(panel_filter(time, line) & (pl.col("agent") != "DMSO"))
    for agent in agent_order:
        sub = panel.filter(pl.col("agent") == agent)
        rho, p_value = stats.spearmanr(sub["dose_level"], sub["distance"])
        medians = dict(sub.group_by("dose_level").agg(pl.col("distance").median()).iter_rows())
        dose_rows.append({"time_point": time, "cell_line": line, "agent": agent, "spearman_rho": rho, "pvalue": p_value,
                          "n_wells": sub.height, "median_distance_low": medians.get(0), "median_distance_mid": medians.get(1), "median_distance_high": medians.get(2)})
dose_response = pl.DataFrame(dose_rows).with_columns(pl.col("time_point").cast(pl.Enum(plate_order)))
dose_response = dose_response.with_columns(pl.Series("qvalue", stats.false_discovery_control(dose_response["pvalue"].to_numpy())))
dose_response = dose_response.with_columns(((pl.col("qvalue") < fdr) & (pl.col("spearman_rho") > 0)).alias("increases_with_dose"))
dose_response.write_csv(table_dir / "dose_response.csv")
print(f"increases with dose (q < {fdr}): {dose_response['increases_with_dose'].sum()} of {dose_response.height}")
dose_response.filter(pl.col("pvalue") < 0.05).sort("agent", "cell_line", "time_point").with_columns(pl.selectors.float().round(3))


# **Result:**
# 
# - Only 1 of 42 dose tests is significant after correction: CBK600005 in CL1 at 4h (ρ = 0.95, q = 0.004). With 3 doses × 3 wells, the test has little power.
# - CBK041498 (both cell lines) and CBK175924 (CL2) at 24h rise with dose (ρ = 0.79–0.84, p < 0.05, q = 0.06–0.09).
# - CBK600005 has full strength already at the lowest stock dose (0.2 mM). Its strength is about the same at all 3 doses (heatmap above), so the tested doses are probably all above saturation.
# - Most active agents are strongest at 24h. TGFβ1 in CL2 grows with time (1.0–1.1 at 1h, 1.2–2.2 at 4h, 2.3–3.4 at 24h).
# - The medium TGFβ1 dose (5 ng/ml) is furthest from DMSO at 4h and 24h, so TGFβ1 shows no clean dose response. In CL1, the TGFβ1 dose order is clean only at 24h (2.4, 3.1, 3.3 from low to high dose).
# - IL6 stays near DMSO (0.7–1.6) at all doses and times. The 24h plate also had longer ER and AGP exposures.

# ## 6. Which treatments look alike?
# 
# ### 6a. Similarity between active treatments
# 
# We take the consensus profiles of the active treatments (mAP q < 0.05), and correlate every pair (Pearson r over all features). Treatments with a similar effect on the cells correlate strongly, and the clustering puts them next to each other. A † marks treatments that look like the unusual DMSO wells of their plate (section 3a).

# In[21]:


active = scores.filter(pl.col("active")).sort("agent", "cell_line", "time_point", "dose_level")
consensus = np.vstack([
    np.median(profiles[meta.filter(panel_filter(r["time_point"], r["cell_line"]) & (pl.col("Metadata_Perturbation") == r["perturbation"]))["row"].to_numpy()], axis=0)
    for r in active.iter_rows(named=True)
])
names = [
    f"{pretty(r['perturbation'])} · {r['time_point']} · {r['cell_line']}" + (" †" if r["looks_like_unusual_dmso"] else "")
    for r in active.iter_rows(named=True)
]
similarity = np.corrcoef(consensus)
pl.DataFrame(similarity, schema=names).insert_column(0, pl.Series("treatment", names)).write_csv(table_dir / "active_consensus_correlation.csv")

# how alike are treatments of the same agent, compared with different agents?
agents = active["agent"].to_numpy()
upper = np.triu_indices(len(names), 1)
same = agents[upper[0]] == agents[upper[1]]
print(f"{len(names)} active treatments; median r between them: same agent {np.median(similarity[upper][same]):.2f} ({same.sum()} pairs), different agents {np.median(similarity[upper][~same]):.2f} ({(~same).sum()} pairs)")
for agent in agent_order:
    pick = agents == agent
    if pick.sum() >= 2:
        block = similarity[np.ix_(pick, pick)][np.triu_indices(pick.sum(), 1)]
        print(f"  {agent}: {pick.sum()} active, median r within {np.median(block):.2f}")
# most similar pairs of different agents
pairs = sorted(((similarity[a, b], names[a], names[b]) for a, b in zip(*upper) if agents[a] != agents[b]), reverse=True)[:8]
print("most similar pairs of different agents:")
for r, a, b in pairs:
    print(f"  r = {r:.2f}: {a}  vs  {b}")


# In[22]:


# figure: clustered correlation heatmap of the active consensus profiles
linkage = hierarchy.linkage(squareform(1 - similarity, checks=False), method="average")
row_colors = [
    [plate_colors[t] for t in active["time_point"].cast(pl.Utf8)],
    [line_colors[c] for c in active["cell_line"]],
    [well_type_colors[w] for w in active["well_type"]],
]
grid = sns.clustermap(
    similarity, row_linkage=linkage, col_linkage=linkage, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
    row_colors=row_colors, xticklabels=False, yticklabels=names, figsize=(7.2, 6.6),
    dendrogram_ratio=(0.1, 0.06), colors_ratio=0.02, cbar_pos=(0.02, 0.86, 0.015, 0.1), tree_kws={"linewidths": 0.5},
)
grid.ax_cbar.set_title("Pearson r", fontsize=6.5)
grid.ax_cbar.tick_params(labelsize=6)
grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), fontsize=5.5)
grid.ax_heatmap.tick_params(length=0)
grid.ax_row_colors.set_xticks([0.5, 1.5, 2.5], ["time", "cell line", "type"], rotation=90, fontsize=6)
grid.ax_row_colors.tick_params(length=0)
handles = (
    [Patch(color=plate_colors[t], label=t) for t in plate_order]
    + [Patch(color=line_colors[c], label=c) for c in line_order]
    + [Patch(color=well_type_colors[w], label=w) for w in ["compound", "ligand"]]
)
grid.ax_heatmap.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=7)
grid.figure.suptitle("Which active treatments change the cells in the same way?  († looks like unusual DMSO wells)", y=1.01)
save_figure(grid.figure, "07_active-treatment-similarity")


# ### 6b. Which feature groups change most?
# 
# We take the strongest active treatment of each agent and cell line, and leave out the † treatments. For each, we average the absolute rescaled values of the consensus profile within each compartment × channel group (for example, all cytoplasm ER features). Shape features have no channel. Colocalization features use two channels.

# In[23]:


var = adata.var.reset_index()
channel_names = {"CorrDNA": "DNA", "CorrRNA": "RNA", "CorrER": "ER", "CorrAGP": "AGP", "CorrMito": "Mito"}
channel_of = [
    "Shape" if not isinstance(c, str) else ("Coloc." if "|" in c else channel_names[c])
    for c in var["channel"]
]
group_of = np.array([f"{o} · {c}" for o, c in zip(var["object"], channel_of)])
group_order = [f"{o} · {c}" for o in ["Cells", "Cytoplasm", "Nuclei"] for c in ["DNA", "RNA", "ER", "AGP", "Mito", "Shape", "Coloc."]]
group_order = [g for g in group_order if g in set(group_of)]

# the strongest active treatment of each agent and cell line, leaving out the ones that look like unusual DMSO wells
top = (
    active.filter(~pl.col("looks_like_unusual_dmso"))
    .sort("strength", descending=True)
    .group_by("agent", "cell_line", maintain_order=True)
    .first()
    .sort("cell_line", pl.col("agent").replace_strict({a: i for i, a in enumerate(agent_order)}))
)
index_of = {(r["perturbation"], r["time_point"], r["cell_line"]): i for i, r in enumerate(active.iter_rows(named=True))}
top_rows = [index_of[(r["perturbation"], r["time_point"], r["cell_line"])] for r in top.iter_rows(named=True)]
group_change = np.array([[np.abs(consensus[i, group_of == g]).mean() for g in group_order] for i in top_rows])
top_names = [names[i] for i in top_rows]
feature_groups = pl.DataFrame(group_change, schema=group_order).insert_column(0, pl.Series("treatment", top_names))
feature_groups.write_csv(table_dir / "feature_group_change_top_active.csv")
print("features per group:", {g: int((group_of == g).sum()) for g in group_order})


# In[24]:


# figure: mean absolute change per compartment x channel for the strongest active treatments
fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
sns.heatmap(
    group_change, ax=ax, cmap=blues, annot=True, fmt=".1f", annot_kws={"fontsize": 5.5},
    linewidths=0.4, linecolor="white", cbar_kws={"label": "mean |change|\n(× DMSO well-to-well spread)", "shrink": 0.8},
)
ax.set_yticks(np.arange(len(top_names)) + 0.5, top_names, rotation=0, fontsize=6)
ax.set_xticks(np.arange(len(group_order)) + 0.5, [g.split(" · ")[1] for g in group_order], rotation=90, fontsize=6)
ax.tick_params(length=0)
# compartment labels above the columns, separated by white lines
compartments = [g.split(" · ")[0] for g in group_order]
for compartment in ["Cells", "Cytoplasm", "Nuclei"]:
    cols = [k for k, c in enumerate(compartments) if c == compartment]
    ax.text(np.mean(cols) + 0.5, -0.3, compartment, ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.axvline(cols[0], color="white", lw=2.5)
ax.set_xlabel("channel (Shape: no channel; Coloc.: two-channel colocalization)")
ax.set_title("Which feature groups change most? (strongest active treatment of each agent and cell line)", pad=20)
save_figure(fig, "08_feature-groups-strongest-treatments")


# **Result:**
# 
# - Among the 40 active treatments, treatments of the same agent correlate more (median r = 0.36) than treatments of different agents (median r = −0.06).
# - The doses of one compound at one time point are nearly identical: CBK600005 median r = 0.93–0.95 within each time and cell line, CBK200566 in CL2 at 24h 0.89, TGFβ1 at 24h 0.80–0.82. So dose changes the size of the effect, not its direction.
# - The two cell lines respond differently. CBK600005 correlates r = 0.46 between CL1 and CL2 at 1h, 0.64 at 4h, but only 0.13 at 24h. For TGFβ1 at 24h, r = 0.19.
# - No two different compounds look alike. The most similar pair of different agents reaches only r = 0.50 (CBK175924 2 mM 1h CL2 vs TGFβ1 10 ng/ml 24h CL2). TGFβ1 in CL2 shares part of its profile with CBK600005 at 24h in CL2 (median r = 0.42).
# - **Feature groups:** CBK600005 at 24h changes nuclear DNA and AGP features most in CL2 (mean |change| 9.0 and 8.7), and cell and nuclear shape in CL1 (5.1).
# - CBK200566 changes mainly DNA features (2.7 in CL1, nuclear DNA 5.6 and shape 5.0 in CL2). TGFβ1 changes mostly ER and AGP (2.7–3.2). CBK041498 and CBK175924 in CL2 change mostly Mito (2.8–3.3).
# - The ER and AGP changes at 24h may partly come from the longer exposures of that plate, although every comparison is against DMSO of the same plate.

# ## 7. Save the results
# 
# We save the main table: one row per treatment, plate and cell line, with the replicate correlation, mAP, strength and flags.

# In[25]:


# the main table: one row per treatment x plate x cell line
scores.sort("cell_line", "time_point", pl.col("perturbation").replace_strict({p: i for i, p in enumerate(perturbation_order)})).write_csv(
    table_dir / "treatment_scores.csv"
)


# ## Summary
# 
# **Data:** 1,136 QCed well profiles (1,399 features, 4 wells left out by well QC) from 3 plates (1h, 4h, 24h) × 2 cell lines. Each plate and cell line has 21 treatments × 3 replicate wells, and 127 DMSO wells.
# 
# **What we did:**
# 
# 1. We clipped extreme values at ±20, and rescaled each feature to the DMSO well-to-well spread.
# 2. We measured replicate agreement (percent replicating against a non-replicate null) and activity (mAP against DMSO, q < 0.05), with DMSO-only negative controls.
# 3. We checked the DMSO wells across the plate, and flagged outlier replicate wells.
# 4. We measured the strength of each treatment, the dose and time trends, and the similarity between active treatments.
# 
# **Findings:**
# 
# - Reproducibility grows with time. Percent replicating is 19% / 19% at 1h, 33% / 43% at 4h, and 33% / 57% at 24h (CL1 / CL2). 40 of 126 tests are active by mAP. The DMSO checks are at chance (about 5%, and 3.3% for mAP).
# - Doses: CBK600005 works at all 3 doses, already at 0.2 mM. CBK041498, CBK175924 and CBK600008 replicate only at 2 mM. CBK200566 (CL2) and TGFβ1 (CL2, 24h) also replicate at lower doses.
# - CL1 at 1h and 4h has a strong plate-position effect: 33 (1h) and 11 (4h) of 127 DMSO wells are shifted, all in odd rows of columns 1–6. At 24h, low-cell wells are shifted: rows M–N (both lines), columns 3–4 (CL1) and columns 20–22 (CL2).
# - These shifted areas explain most outlier replicates, and the apparent TGFβ1 effect in CL1 at 1h and 4h.
# - 48 of 372 treated wells (13%) are outlier replicates. Many of them sit in the shifted areas or have few cells.
# - CBK600005 is active in all 18 tests (strength up to 9.7), flat across doses, and halves the cell count at 24h (35–54% of DMSO). It has a strong effect, possibly with toxicity.
# - TGFβ1 is active at all doses at 24h in both lines, and grows with time in CL2. It shows no clean dose response (5 ng/ml strongest in CL2). IL6 and CBK600008 are never active.
# - CBK200566 is strongly active in CL2 at 24h (all doses). CBK041498 and CBK175924 are active mainly at 2 mM.
# - The doses of one compound share one profile. Different compounds do not look alike (r ≤ 0.50). The two cell lines respond differently to the same treatment (for example, r = 0.13 for CBK600005 at 24h).
# 
# **Caveats:**
# 
# - Time changes together with exposure (the 24h plate used longer ER and AGP exposures), and cell line changes together with plate side.
# - Compound doses are stock concentrations. The final concentrations are unknown.
# - Only 3 replicates and 3 doses: the dose tests have little power, and one bad well changes a result.
# - We do not correct the plate-position effects here. The "looks like unusual DMSO" flag (r ≥ 0.8) is a simple screen, not a correction.
# - Cell number drives many features (notebook 01). Do not read the profiles of treatments that lower cell number at 24h (notebook 04) as pure morphology.
