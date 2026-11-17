# Pilot morphological profiling analysis

This repository contains the analysis of an imaging-based pilot screen across treatments, doses and three time points (1, 4 and 24 hours). We processed CellProfiler measurements into single-cell and well-level profiles, checked data quality, and assessed treatment responses and technical variation.

The analysis followed four stages:

1. **[Prepare profiles](notebooks/0.prepare-profiles/):** converted single-cell measurements from compressed CSV files to Parquet and checked data integrity.
2. **[Quality control](notebooks/1.quality-control/):** assessed image quality, segmentation and measurement completeness, and flagged unreliable images and cell profiles.
3. **[Preprocessing](notebooks/2.preprocessing/):** added treatment metadata, normalized features relative to DMSO controls, removed uninformative and redundant features, checked well quality, and aggregated single-cell measurements into median profiles per well. We processed both all cells and cells passing quality control for comparison.
4. **[Exploratory analysis](notebooks/3.EDA/):** examined the experimental layout and imaging settings, visualized profile variation, investigated batch and plate-position effects, assessed replicate agreement and treatment responses across dose and time, and identified treatments associated with reduced cell counts and possible cytotoxicity.

The outputs include processed profiles, quality-control reports, figures and summary tables, saved in each stage's `results/` directory. Each stage has its own README with inputs, outputs and instructions for running the notebooks.
