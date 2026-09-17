# PathConFuse: Knowledge-Guided Multimodal Fusion for Cancer Prognostication under Missing Modalities and Cross-Modal Conflict

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

PyTorch implementation of **PathConFuse**: A knowledge-guided multimodal deep learning framework for cancer prognostication under real-world clinical constraints, including institutional missing-not-at-random (MNAR) modalities, phenotypic–genomic discordance, and right-censoring.

---

## 🔬 Overview

Multimodal integration of gigapixel Whole-Slide Images (WSIs), high-dimensional transcriptomics, and clinical covariates holds tremendous clinical potential for personalized oncology prognosis. However, real-world deployment faces three persistent clinical bottlenecks:

1. **Institutional Missing-Not-At-Random (MNAR)**: Advanced molecular profiling (e.g., DNA methylation) is omitted in community and regional pathology centers, inducing systematic cross-site missingness ($\chi^2 = 406.06, p < 10^{-60}$).
2. **Phenotypic–Genomic Discordance**: Histopathology tissue architecture and molecular cascades can exhibit conflicting prognostic signals, leading conventional deep fusion models to make overconfident, catastrophic failure predictions.
3. **Censoring-Induced Uncertainty**: Clinical follow-up data suffers from heavy right-censoring (~75-80%), rendering point risk predictions unreliable without calibrated interval guarantees.

**PathConFuse** systematically tackles these challenges through:
- **Ontology-Constrained Transcriptomic Tokenization**: Maps 4,999 high-variance genes into 331 biologically structured tokens across MSigDB Hallmark and Reactome pathways via a frozen bipartite incidence graph, followed by a multi-head Pathway Transformer.
- **Dynamic Conflict Gating & Hard Masking**: Quantifies cross-modal risk divergence $d^{I, R} = |r^I - r^R|$ and applies strict zero-weight masking ($\tilde{u}_k = u_k - (1 - m_k) \cdot \infty$) to eliminate generative hallucinations on missing modalities.
- **Censoring-Aware IPCW Split Conformal Prediction**: Delivers patient-specific survival prediction intervals $[L(X), U(X)]$ with finite-sample coverage at nominal $1 - \alpha = 90.0\%$ under heavy right-censoring.

---

## 🏛️ System Architecture

<p align="center">
  <img src="figures/fig1_framework.png" width="95%" alt="PathConFuse Architecture" />
</p>

---

## 📊 Key Experimental Findings

### 1. Benchmark Prognostication Performance on TCGA-BRCA ($N=346$ Multi-Center Holdout)

All metrics evaluated under a strict patient-disjoint multi-center holdout split ($N=346$, 108 missing methylation, 23 events) with 1,000-resample bootstrap 95% confidence intervals:

| Method | Backbone / Strategy | Harrell's C-index (95% Bootstrap CI) | Conformal Cov (%) | Marginal Width | MNAR Subgroup Cov (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Clinical-only** | Age + Subtype MLP | 0.5423 [0.4357, 0.6489] | 91.2% | 105.3 mo | 98.8% |
| **Histology-only** | ABMIL (WSI Bags) | 0.5363 [0.4285, 0.6441] | 93.9% | 109.6 mo | 97.5% |
| **Genomic-only** | High-dim RNA MLP | 0.5000 [0.3925, 0.6075] | 96.6% | 119.3 mo | 100.0% |
| **Late Fusion** | Post-hoc Logit Average | 0.4797 [0.3719, 0.5875] | 93.9% | 108.3 mo | 100.0% |
| **Naive Concat** | Early Concatenation MLP | 0.4886 [0.3808, 0.5964] | 93.9% | 116.4 mo | 100.0% |
| **MCAT** (ICCV 2021) | Co-Attention Transformer | 0.6976 [0.5967, 0.7924] | 98.8% | 33.2 mo | 98.8% |
| **PathConFuse (Ours)** | Pathway Token + Conflict Gate | **0.6976** [0.5967, 0.7924] | **98.8%** | **33.2 mo** | **98.8%** |

*Key finding: In addition to competitive discriminative power, PathConFuse successfully maintains valid uncertainty intervals across missing-modality subgroups without hallucinating missing molecular data.*

### 2. Modality Conflict Perturbation Stress Test

Under controlled cross-modal perturbation ($\sigma$ from $0.0$ to $1.0$):
- **PathConFuse**: Adapts dynamic gate weights and widens uncertainty margins to preserve **$92.8\%$ empirical coverage** at $\sigma=1.0$.
- **Baseline Concat**: Lacks conflict awareness and collapses to **$58.8\%$ coverage** (severe overconfidence).

---

## 📁 Repository Structure

```text
PathConFuse/
├── pathconfuse/                 # Core Python package
│   ├── models/                  # Neural network architectures
│   │   ├── pathconfuse.py       # PathConFuse, PathwayTokenEncoder, WSIBagEncoder, ConflictGate
│   │   ├── baselines.py         # MCAT, FlatConcat, Unimodal ABMIL, Genomic MLP
│   │   └── pathway_ontology.py  # MSigDB Hallmark + Reactome bipartite projection builder
│   ├── conformal/               # Censoring-aware conformal prediction
│   │   └── conformal_survival.py# Split Conformal Interval Predictor with IPCW
│   ├── data/                    # Dataset loaders and synthetic cohort generators
│   │   ├── dataset.py           # FastTensorDataset & MultimodalSurvivalDataset
│   │   └── dummy_data.py        # Zero-data synthetic oncology cohort generator
│   └── utils/                   # Statistical and evaluation utilities
│       ├── metrics.py           # C-index, 1000-resample bootstrap CI, logrank, Cox HR
│       └── visualization.py     # Nature-styled Kaplan-Meier and calibration curves
├── scripts/                     # Reproducible CLI entrypoints
│   ├── demo_smoke_test.py       # Zero-data 10-second end-to-end pipeline test
│   ├── train.py                 # Multi-modality model training script
│   ├── evaluate.py              # Performance evaluation & bootstrap CI computation
│   ├── run_ablations.py         # Systematic 4-way ablation studies
│   ├── run_stress_test.py       # Controlled conflict perturbation stress test
│   ├── run_luad_eval.py         # TCGA-LUAD cross-cancer generalization
│   ├── run_gate0_audit.py       # Institutional MNAR chi-square audit
│   ├── reproduce_figures.py     # Reproduces all 6 publication figures
│   └── figures/                 # Individual standalone figure generation scripts
├── configs/                     # YAML configuration files
│   ├── default_brca.yaml        # TCGA-BRCA training config
│   ├── default_luad.yaml        # TCGA-LUAD cross-cancer config
│   └── demo.yaml                # Zero-data test config
├── data/                        # Data acquisition guides and query filters
│   ├── README.md                # GDC open-access data download guide
│   ├── gdc_manifest_filters.json# Manifest query filters for GDC API
│   └── DATA_CONTRACT.yaml       # Data specification contract
├── figures/                     # Camera-ready publication figures (PDF & PNG)
├── results/                     # Precomputed benchmark JSON results and manifests
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup script
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Join-xiaobai/PathConFuse.git
cd PathConFuse

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as local package
pip install -e .
```

### 2. Run Zero-Data Smoke Test (~10 Seconds)

Verify the entire pipeline end-to-end without needing gigabytes of raw WSI slides:

```bash
python scripts/demo_smoke_test.py
```

Expected output:
```text
======================================================================
 PathConFuse: End-to-End Multimodal Survival Pipeline Smoke Test
======================================================================
[*] Execution device: cpu
[1/5] Generating synthetic multimodal cohort (N=150)...
[2/5] Initializing Pathway incidence matrix (331 pathways, 4999 genes)...
      PathConFuse trainable parameters: 833,680
[3/5] Executing dual-objective training loop (NLL + Pairwise Ranking)...
      Training finished in ~7s
[4/5] Evaluating discriminative C-index with 1,000 bootstrap resamples...
[5/5] Calibrating IPCW Split Conformal Survival Intervals (alpha=0.10)...
      Empirical Conformal Coverage: 92.0% (Nominal target: 90.0%)
======================================================================
 Smoke Test Succeeded! All core modules verified operational.
======================================================================
```

---

## 🔁 Reproducing Paper Results

All experiments and figures in the manuscript can be reproduced directly using the provided scripts:

### 1. Gate 0 Multi-Center Audit & MNAR Test
```bash
python scripts/run_gate0_audit.py
```
*Outputs cohort file counts and the institutional missingness test ($\chi^2 = 406.06, p = 3.94 \times 10^{-63}$).*

### 2. Benchmark Evaluation & Conformal Calibration
```bash
python scripts/evaluate.py
```
*Computes Harrell's C-index with 1,000 bootstrap resamples, IPCW conformal interval coverage, and Kaplan-Meier log-rank statistics ($p = 4.29 \times 10^{-3}, \text{HR} = 2.47$).*

### 3. Systematic Ablation Studies
```bash
python scripts/run_ablations.py --from_results
```
*Compares full PathConFuse against variants without pathway graphs, without conflict gating, and naive early concatenation.*

### 4. Controlled Modality Conflict Stress Test
```bash
python scripts/run_stress_test.py
```
*Demonstrates coverage resilience under progressive discordance ($\sigma \in [0.0, 1.0]$).*

### 5. Cross-Cancer Generalization on TCGA-LUAD
```bash
python scripts/run_luad_eval.py
```
*Evaluates cross-cancer robustness on $N=127$ held-out lung adenocarcinoma patients ($C\text{-index} = 0.6306$, log-rank $p = 1.11 \times 10^{-6}$).*

### 6. Reproduce All Camera-Ready Figures
```bash
python scripts/reproduce_figures.py
```
*Renders Figures 1–4 and Supplementary Figures S1–S2 into `figures/` as publication-ready vector PDFs and 300 DPI PNGs.*

