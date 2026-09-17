#!/usr/bin/env python3
"""
Supplementary Figure S2: Controlled Modality Conflict Stress Test
Dynamic uncertainty margin expansion and attention redistribution under escalating discordance.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def find_results_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.environ.get("FIG_OUTPUT_DIR", os.path.abspath(os.path.join(base_dir, "../../figures")))
    os.makedirs(fig_dir, exist_ok=True)
    candidates = [
        os.path.join(fig_dir, "results"),
        os.path.join(fig_dir, "../results"),
        os.path.join(fig_dir, "../../07-engineering/results"),
        fig_dir
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "results_manifest.json")):
            return c
    raise FileNotFoundError("Could not find results_manifest.json in candidate paths.")

def generate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.environ.get("FIG_OUTPUT_DIR", os.path.abspath(os.path.join(base_dir, "../../figures")))
    os.makedirs(fig_dir, exist_ok=True)
    res_dir = find_results_dir()
    
    res_path = os.path.join(res_dir, "results_manifest.json")
    with open(res_path, "r") as f:
        master = json.load(f)

    stress_results = master["controlled_conflict_stress_test"]
    nl = stress_results["noise_levels"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10.5), dpi=300)

    # (a) Conformal Coverage Resilience
    ax1.plot(nl, np.array(stress_results["pathconfuse_cov"])*100, "s-", color="#2E7D32", linewidth=2.5, markersize=8, label="PathConFuse (Conflict Gate Preserves Coverage)")
    ax1.plot(nl, np.array(stress_results["concat_cov"])*100, "o--", color="#C62828", linewidth=2.3, markersize=7.5, label="Baseline Concat (Severe Coverage Collapse)")
    ax1.axhline(90.0, color="black", linestyle=":", linewidth=1.8, label="Nominal 90% Target")
    ax1.fill_between(nl, 90.0, 100, color="#E8F5E9", alpha=0.35)
    ax1.text(0.32, 92.0, "Calibrated Safety Zone", fontsize=10.0, color="#2E7D32", fontweight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#E8F5E9", edgecolor="#A5D6A7", alpha=0.90))
    ax1.text(0.80, 78.0, "Overconfident\nUndercoverage\n(Hazardous)", fontsize=10.0, color="#C62828", fontweight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFEBEE", edgecolor="#EF9A9A", alpha=0.92))
    ax1.set_xlabel("Discordant Perturbation Intensity (σ)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Empirical Conformal Coverage (%)", fontsize=13, fontweight="bold")
    ax1.set_title("(a) Conformal Coverage Resilience Under Modality Discordance", fontsize=13.5, fontweight="bold", pad=10)
    ax1.set_ylim(50, 102)
    ax1.tick_params(axis="both", labelsize=11.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=10, frameon=True, facecolor="white", edgecolor="#B0BEC5")

    # (b) Interval Width Response
    ax2.plot(nl, stress_results["pathconfuse_width"], "s-", color="#2E7D32", linewidth=2.4, markersize=7.5, label="PathConFuse (Adaptive Expansion)")
    ax2.plot(nl, stress_results["concat_width"], "^--", color="#7F7F7F", linewidth=2.2, markersize=6.5, label="Baseline Concat (Rigid / Overconfident)")
    ax2.set_xlabel("Discordant Perturbation Intensity (σ)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Total Conformal Interval Width (Months)", fontsize=13, fontweight="bold")
    ax2.set_title("(b) Conformal Interval Width Response to Conflict", fontsize=13.5, fontweight="bold", pad=10)
    ax2.tick_params(axis="both", labelsize=11.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor="#B0BEC5")

    # (c) Modality Gating Weights
    ax3.plot(nl, stress_results["pathconfuse_alpha_wsi"], "o-", color="#55A868", linewidth=2.4, markersize=7.5, label="Histopathology Weight (α_I)")
    ax3.plot(nl, stress_results["pathconfuse_alpha_rna"], "s-", color="#4C72B0", linewidth=2.4, markersize=7.5, label="Transcriptomics Weight (α_R)")
    ax3.set_xlabel("Discordant Perturbation Intensity (σ)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Modality Gate Attention Weight", fontsize=13, fontweight="bold")
    ax3.set_title("(c) Dynamic Modality Reliability Redistribution", fontsize=13.5, fontweight="bold", pad=10)
    ax3.tick_params(axis="both", labelsize=11.5)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="center right", fontsize=10, frameon=True, facecolor="white", edgecolor="#B0BEC5")

    # (d) Conflict Metric
    ax4.plot(nl, stress_results["conflict_scores"], "d-", color="#D95F02", linewidth=2.4, markersize=7.5, label="Mean Modality Discordance Metric (d^{I, R})")
    ax4.set_xlabel("Discordant Perturbation Intensity (σ)", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Absolute Risk Divergence |r^I - r^R|", fontsize=13, fontweight="bold")
    ax4.set_title("(d) Detected Inter-Modality Discordance Score", fontsize=13.5, fontweight="bold", pad=10)
    ax4.tick_params(axis="both", labelsize=11.5)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor="#B0BEC5")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_s2_conflict_stress_test.pdf"))
    plt.savefig(os.path.join(fig_dir, "fig_s2_conflict_stress_test.png"), dpi=300)
    plt.close()
    print("Generated fig_s2_conflict_stress_test successfully!")

if __name__ == "__main__":
    generate()
