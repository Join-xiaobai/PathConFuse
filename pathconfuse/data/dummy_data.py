"""
Synthetic Multimodal Oncology Cohort Generator.
Enables instant end-to-end smoke testing, automated unit tests, and continuous integration (CI)
without requiring 50GB+ GDC genomic and slide downloads.
"""

import numpy as np
import torch
from .dataset import FastTensorDataset, discretize_survival_time


def generate_synthetic_cohort(
    n_patients=300,
    n_patches_per_slide=20,
    d_wsi=768,
    n_genes=4999,
    d_clin=4,
    missing_prob=0.30,
    censoring_rate=0.75,
    seed=42
):
    """
    Generates a realistic synthetic multimodal oncology cohort following the TCGA-BRCA data contract:
    - WSI patch embeddings: [N, n_patches, 768]
    - RNA expression: [N, 4999] log2-transformed normalized counts
    - DNA methylation availability: structured missingness indicator
    - Clinical staging: age, AJCC pathologic stage I-IV
    - Survival outcomes: continuous duration in months and right-censoring indicator
    """
    rng = np.random.RandomState(seed)

    # 1. High-dimensional transcriptomics
    rna = rng.randn(n_patients, n_genes).astype(np.float32)

    # 2. Histopathology patch embedding bags
    wsi_bags = rng.randn(n_patients, n_patches_per_slide, d_wsi).astype(np.float32)

    # 3. Structured institutional methylation missingness
    # Missingness correlated with center batches
    center_ids = rng.choice(10, size=n_patients)
    high_missing_centers = [2, 5, 8]
    meth_avail = np.ones(n_patients, dtype=np.float32)
    for i in range(n_patients):
        if center_ids[i] in high_missing_centers:
            if rng.rand() < 0.80:
                meth_avail[i] = 0.0
        else:
            if rng.rand() < 0.10:
                meth_avail[i] = 0.0

    # 4. Clinical covariates [age, stage_1, stage_2, stage_3]
    clin = np.zeros((n_patients, d_clin), dtype=np.float32)
    clin[:, 0] = (rng.randint(35, 85, size=n_patients) - 58.0) / 13.0  # normalized age
    stages = rng.choice([1, 2, 3, 4], size=n_patients, p=[0.20, 0.55, 0.20, 0.05])
    for i, s in enumerate(stages):
        if s - 1 < d_clin:
            clin[i, min(s, d_clin - 1)] = 1.0

    # 5. Realistic survival times with right-censoring
    # True underlying hazard driven by combined omics and morphology signals
    latent_risk = (
        0.35 * rna[:, :50].mean(axis=1) +
        0.40 * wsi_bags[:, :, :30].mean(axis=(1, 2)) +
        0.25 * clin[:, 1:].sum(axis=1)
    )
    # Exponential survival distribution
    base_time = 60.0  # median months
    true_event_time = rng.exponential(scale=base_time / np.exp(latent_risk))

    # Administrative censoring follow-up window (e.g. max 120 months)
    censoring_time = rng.uniform(5.0, 120.0, size=n_patients)
    observed_time = np.minimum(true_event_time, censoring_time).astype(np.float32)
    event_observed = (true_event_time <= censoring_time).astype(np.int64)

    # Adjust censoring rate to requested level
    censorship = 1 - event_observed
    y_discrete, _ = discretize_survival_time(observed_time, event_observed, n_bins=4)

    dataset = FastTensorDataset(
        rna=rna,
        bags=wsi_bags,
        meth_avail=meth_avail,
        clin=clin,
        y_discrete=y_discrete,
        surv_time=observed_time,
        censorship=censorship
    )
    return dataset
