"""
PathConFuse: Pathway-Constrained Conflict-aware Fusion of Whole-Slide Images
and Incomplete Multi-Omics for Trustworthy Cancer Survival Intervals.
"""

__version__ = "1.0.0"
__author__ = "PathConFuse Research Team"

from .models.pathconfuse import (
    PathConFuse,
    PathwayTokenEncoder,
    WSIBagEncoder,
    ConflictGate,
    NLLSurvLoss,
    PairwiseSurvLoss
)
from .models.baselines import (
    MCATSurv,
    FlatConcatSurv,
    HistologyOnlySurv,
    GenomicOnlySurv,
    ClinicalOnlySurv,
    LateFusionSurv
)
from .models.pathway_ontology import load_or_generate_pathway_mask
from .conformal.conformal_survival import CensoredConformalSurvival
from .data.dataset import FastTensorDataset, MultimodalSurvivalDataset, discretize_survival_time
from .data.dummy_data import generate_synthetic_cohort
from .utils.metrics import compute_c_index, bootstrap_c_index_ci, logrank_analysis

__all__ = [
    "PathConFuse",
    "PathwayTokenEncoder",
    "WSIBagEncoder",
    "ConflictGate",
    "NLLSurvLoss",
    "PairwiseSurvLoss",
    "MCATSurv",
    "FlatConcatSurv",
    "HistologyOnlySurv",
    "GenomicOnlySurv",
    "ClinicalOnlySurv",
    "LateFusionSurv",
    "load_or_generate_pathway_mask",
    "CensoredConformalSurvival",
    "FastTensorDataset",
    "MultimodalSurvivalDataset",
    "discretize_survival_time",
    "generate_synthetic_cohort",
    "compute_c_index",
    "bootstrap_c_index_ci",
    "logrank_analysis",
]
