"""
Publication-Grade Plotting Utilities for Survival Analysis and Conformal Calibration.
"""

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter


def set_nature_style():
    """Configures clean Nature-inspired matplotlib styling."""
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#EBEBEB"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5


def plot_kaplan_meier_curves(y_time, event_observed, risk_scores, save_path=None, title="Patient Risk Stratification"):
    """
    Plots high-risk vs low-risk patient Kaplan-Meier curves with 95% confidence bands.
    """
    set_nature_style()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    median_risk = np.median(risk_scores)
    high_mask = (risk_scores >= median_risk)
    low_mask = ~high_mask

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    kmf_low.fit(y_time[low_mask], event_observed[low_mask], label="Low Risk (Predicted)")
    kmf_high.fit(y_time[high_mask], event_observed[high_mask], label="High Risk (Predicted)")

    kmf_low.plot_survival_function(ax=ax, color="#2B5C8F", ci_show=True, ci_alpha=0.15, lw=2.0)
    kmf_high.plot_survival_function(ax=ax, color="#D95F02", ci_show=True, ci_alpha=0.15, lw=2.0)

    ax.set_xlabel("Survival Time (Months)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overall Survival Probability", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
