# Data Acquisition and Preprocessing Specification

This directory describes how to acquire and structure multi-modal oncology cohorts from the NCI Genomic Data Commons (GDC) for training and evaluating **PathConFuse**.

---

## 1. Public Data Sources (Open-Access)

All clinical and omics files used in PathConFuse belong to the **GDC Open-Access Tier** and require no dbGaP authorization:
- **Diagnostic Whole-Slide Images (WSI)**: SVS files at 20× / 40× magnification.
- **Transcriptomics (RNA-seq)**: `STAR - Counts` gene expression quantification (unstranded TPM and count matrices).
- **DNA Methylation**: Illumina Human Methylation 450 platform Beta values.
- **Clinical Endpoints**: Overall survival (OS) time in months, right-censoring indicator, age, histologic subtype, and AJCC pathologic tumor stage.

---

## 2. GDC Manifest Queries

To query and download cohorts using the official `gdc-client`, use the parameters defined in `data/gdc_manifest_filters.json`:
- **TCGA-BRCA**: 1,062 diagnostic slide cases, 1,095 STAR counts, 791 Methylation-450k cases, 1,098 clinical cases.
- **TCGA-LUAD**: 478 diagnostic slide cases, 518 STAR counts, 461 Methylation-450k cases, 585 clinical cases.

---

## 3. Directory Layout for Real Cohort Data

When working with local data, arrange preprocessed features according to the following layout:

```text
data/
├── gdc_manifest_filters.json
├── DATA_CONTRACT.yaml
├── brca_metadata.csv            # Patient IDs, survival_months, censorship, clinical covariates
├── brca_rna_counts.csv          # 4,999 normalized gene expressions (log2(TPM + 1))
├── brca_wsi_features/           # Directory of pre-extracted patch feature bags
│   ├── TCGA-A8-A06O.pt          # Tensor shape: [N_patches, 768] (CTransPath / UNI / ResNet)
│   ├── TCGA-A8-A06P.pt
│   └── ...
└── luad_wsi_features/           # Same format for TCGA-LUAD
```

---

## 4. Zero-Data Synthetic Testing

If you do not have raw GDC files downloaded, you can run all models, ablations, and conformal interval calibrations instantly via the built-in synthetic generator:

```bash
python scripts/demo_smoke_test.py
```
This generates realistic synthetic cohorts (100 training, 50 testing) matching the exact dimensions of WSI bags (100×768), 4,999 genes, and institutional availability masks.
