#!/usr/bin/env python3
"""
Supplementary Figure S1: TCGA-LUAD Cross-Cancer Protocol Replication
Empirical patient-level Kaplan-Meier curves and conformal calibration benchmarks.
Aligned with Table 3 of the main manuscript.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

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
        if os.path.exists(os.path.join(c, "luad_patient_predictions.json")):
            return c
    raise FileNotFoundError("Could not find luad_patient_predictions.json in candidate paths.")

def generate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.environ.get("FIG_OUTPUT_DIR", os.path.abspath(os.path.join(base_dir, "../../figures")))
    os.makedirs(fig_dir, exist_ok=True)
    res_dir = find_results_dir()
    
    res_path = os.path.join(res_dir, "pathconfuse_master_revision_results.json")
    if not os.path.exists(res_path):
        res_path = os.path.join(res_dir, "results_manifest.json")
        
    with open(res_path, "r") as f:
        master = json.load(f)
        
    luad_pred_path = os.path.join(res_dir, "luad_patient_predictions.json")
    with open(luad_pred_path, "r") as f:
        luad_data = json.load(f)

    luad_results = master["tcga_luad_replication"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.6, 11.4), dpi=300)

    # (a) True empirical patient-level Kaplan-Meier plot (N=127)
    times = np.array(luad_data["times"])
    events = np.array(luad_data["events"])
    predicted_risk = np.array(luad_data["predicted_risk"])
    threshold = float(luad_data["risk_threshold"])
    high_mask = predicted_risk >= threshold
    
    kmf_l = KaplanMeierFitter()
    kmf_h = KaplanMeierFitter()
    
    kmf_l.fit(times[~high_mask], event_observed=events[~high_mask], label="Low Risk (< Median)")
    kmf_h.fit(times[high_mask], event_observed=events[high_mask], label="High Risk (≥ Median)")
    
    kmf_l.plot_survival_function(ax=ax1, color="#2E7D32", linewidth=2.6, ci_show=True)
    kmf_h.plot_survival_function(ax=ax1, color="#C62828", linewidth=2.6, ci_show=True)
    
    ax1.set_xlabel("Follow-up Time (Months)", fontsize=14.5, fontweight="bold")
    ax1.set_ylabel("Overall Survival Probability", fontsize=14.5, fontweight="bold")
    ax1.set_title(f"(a) Kaplan-Meier Risk Stratification (TCGA-LUAD, N={len(times)})", fontsize=15.0, fontweight="bold", pad=14)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlim(0, 100)
    ax1.tick_params(axis="both", labelsize=12.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=12.5, frameon=True, facecolor="white", edgecolor="#B0BEC5")
    
    hr = float(luad_data["hazard_ratio"])
    hr_ci = luad_data["hazard_ratio_ci"]
    p_val = float(luad_data["logrank_p"])
    ax1.text(0.45, 0.77, f"Log-rank Test: p = {p_val:.2e}\nHR = {hr:.2f} (95% CI: {hr_ci[0]:.2f}–{hr_ci[1]:.2f})",
             transform=ax1.transAxes, fontsize=13.0, fontweight="bold", color="#1A237E",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8EAF6", edgecolor="#7986CB", linewidth=1.3))

    # (b) Calibration curve
    nom_l = np.array(luad_results["PathConFuse (Full Proposed)"]["calibration_curve"]["nominal_levels"]) * 100
    pcf_l_ov = np.array(luad_results["PathConFuse (Full Proposed)"]["calibration_curve"]["overall_coverages"]) * 100
    cat_l_ov = np.array(luad_results["Baseline (Concat Fusion)"]["calibration_curve"]["overall_coverages"]) * 100

    ax2.plot(nom_l, nom_l, "k--", linewidth=2.2, label="Ideal 1:1 Diagonal")
    ax2.plot(nom_l, pcf_l_ov, "s-", color="#4C72B0", linewidth=2.6, markersize=8.5, label="PathConFuse (LUAD Holdout)")
    ax2.plot(nom_l, cat_l_ov, "^-.", color="#C62828", linewidth=2.2, markersize=7.5, label="Concat Fusion (LUAD)")
    ax2.set_xlabel("Nominal Confidence Level 1 - α (%)", fontsize=14.5, fontweight="bold")
    ax2.set_ylabel("Empirical Conformal Coverage (%)", fontsize=14.5, fontweight="bold")
    ax2.set_title("(b) Conformal Calibration Curve Across Nominal Levels", fontsize=15.0, fontweight="bold", pad=14)
    ax2.tick_params(axis="both", labelsize=12.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=12.5, frameon=True, facecolor="white", edgecolor="#B0BEC5")

    # (c) Comparative discrimination bar plot (aligned strictly with Table 3 of the main manuscript)
    luad_names = ["Clinical Only", "Histology (WSI)", "Genomics (RNA)", "Late Fusion", "MCAT (SOTA)", "Concat Fusion", "PathConFuse"]
    c_luad_vals = [0.4958, 0.6478, 0.5712, 0.6339, 0.6784, 0.6587, 0.6558]
    c_luad_cis = [
        [0.4230, 0.5696],
        [0.5844, 0.7136],
        [0.5009, 0.6429],
        [0.5656, 0.7039],
        [0.6123, 0.7466],
        [0.5886, 0.7263],
        [0.5904, 0.7212]
    ]
    c_luad_low = [c_luad_vals[i] - c_luad_cis[i][0] for i in range(len(c_luad_vals))]
    c_luad_high = [c_luad_cis[i][1] - c_luad_vals[i] for i in range(len(c_luad_vals))]

    colors_c = ["#999999", "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#2B8CBE"]
    bars3 = ax3.bar(np.arange(len(luad_names)), c_luad_vals, yerr=[c_luad_low, c_luad_high], capsize=5.5,
                    color=colors_c, edgecolor="black", alpha=0.88, linewidth=1.2,
                    error_kw=dict(lw=1.4, capthick=1.4))

    for idx, bar in enumerate(bars3):
        val = c_luad_vals[idx]
        upper = c_luad_cis[idx][1]
        ax3.text(bar.get_x() + bar.get_width()/2, upper + 0.009, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=12.0, fontweight="bold", color="#111111")

    ax3.set_ylabel("Concordance Index (C-Index)", fontsize=14.5, fontweight="bold")
    ax3.set_title("(c) Comparative Discrimination on TCGA-LUAD Holdout", fontsize=15.0, fontweight="bold", pad=14)
    ax3.set_xticks(np.arange(len(luad_names)))
    ax3.set_xticklabels(luad_names, fontsize=12.0, fontweight="bold", rotation=25, ha="right")
    ax3.set_ylim(0.40, 0.82)
    ax3.tick_params(axis="y", labelsize=12.5)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)

    # (d) Coverage bar plot (aligned strictly with Table 3 of the main manuscript)
    cov_luad_vals = [85.9, 85.9, 85.9, 85.9, 85.9, 82.8, 85.9]
    colors_d = ["#4A7BB0"] * 5 + ["#C0392B"] + ["#1E8449"]
    bars4 = ax4.bar(np.arange(len(luad_names)), cov_luad_vals, color=colors_d, edgecolor="black", alpha=0.88, linewidth=1.2)

    for bar, val in zip(bars4, cov_luad_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=12.0, fontweight="bold", color="#111111")

    ax4.axhline(90.0, color="#D32F2F", linestyle="--", linewidth=2.4, label="90% Nominal Calibration Target")
    ax4.set_ylabel("Empirical Conformal Coverage (%)", fontsize=14.5, fontweight="bold")
    ax4.set_title("(d) Conformal Coverage on TCGA-LUAD Holdout", fontsize=15.0, fontweight="bold", pad=14)
    ax4.set_xticks(np.arange(len(luad_names)))
    ax4.set_xticklabels(luad_names, fontsize=12.0, fontweight="bold", rotation=25, ha="right")
    ax4.set_ylim(58, 105)
    ax4.tick_params(axis="y", labelsize=12.5)
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=12.0, frameon=True, facecolor="white", edgecolor="#B0BEC5")

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])
    plt.subplots_adjust(hspace=0.32, wspace=0.20)

    pdf_out = os.path.join(fig_dir, "fig_s1_luad_replication.pdf")
    png_out = os.path.join(fig_dir, "fig_s1_luad_replication.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out, dpi=300)
    plt.close()
    print(f"Generated {pdf_out} and {png_out} successfully!")

if __name__ == "__main__":
    generate()
