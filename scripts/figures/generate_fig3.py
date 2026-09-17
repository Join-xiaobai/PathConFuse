#!/usr/bin/env python3
"""
Figure 3: Empirical Kaplan-Meier Survival Stratification & Conformal Calibration Curves
Fully reproducible script loading empirical patient predictions from results_manifest.json
and brca_patient_predictions.json (true model-predicted risks, zero outcome leakage).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def find_results_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "../../results"),
        os.path.join(base_dir, "../results"),
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "../../07-engineering/results"),
        base_dir
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "brca_patient_predictions.json")):
            return c
    raise FileNotFoundError("Could not find brca_patient_predictions.json in candidate paths.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = find_results_dir()
    
    # Load manifest and patient data
    manifest_path = os.path.join(res_dir, "results_manifest.json")
    patient_path = os.path.join(res_dir, "brca_patient_predictions.json")
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    with open(patient_path, "r") as f:
        patient_data = json.load(f)
        
    times = np.array(patient_data["times"])
    events = np.array(patient_data["events"])  # 1 = death, 0 = censored
    predicted_risk = np.array(patient_data["predicted_risk"])
    threshold = float(patient_data["risk_threshold"])
    
    # Strictly binarized by training median risk cutoff (zero outcome leakage)
    high_mask = predicted_risk >= threshold
    
    p_val = float(patient_data["logrank_p"])
    hr = float(patient_data["hazard_ratio"])
    hr_ci = patient_data["hazard_ratio_ci"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    
    # Panel (a): Empirical Kaplan-Meier Curves
    kmf_l = KaplanMeierFitter()
    kmf_h = KaplanMeierFitter()
    
    kmf_l.fit(times[~high_mask], event_observed=events[~high_mask], label="Low Risk (< Median Threshold)")
    kmf_h.fit(times[high_mask], event_observed=events[high_mask], label="High Risk (≥ Median Threshold)")
    
    kmf_l.plot_survival_function(ax=ax1, color="#2E7D32", linewidth=2.2, ci_show=True)
    kmf_h.plot_survival_function(ax=ax1, color="#C62828", linewidth=2.2, ci_show=True)
    
    ax1.set_xlabel("Follow-up Time (Months)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Overall Survival Probability", fontsize=11, fontweight="bold")
    ax1.set_title(f"(a) Kaplan-Meier Risk Stratification (BRCA Site-Held-Out N={len(times)})", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlim(0, 120)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9.0, frameon=True, facecolor="white", edgecolor="#B0BEC5")
    
    ax1.text(0.03, 0.20, f"Log-rank Test: p = {p_val:.4f}\nHR = {hr:.2f} (95% CI: {hr_ci[0]:.2f}–{hr_ci[1]:.2f})",
             transform=ax1.transAxes, fontsize=9.5, fontweight="bold", color="#1A237E",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#E8EAF6", edgecolor="#9FA8DA"))

    # Panel (b): Conformal Calibration Reliability Curve
    cal_pcf = manifest["tcga_brca_benchmarks"]["PathConFuse (Full Proposed)"]["calibration_curve"]
    cal_cat = manifest["tcga_brca_benchmarks"]["Ablation: w/o Conflict Gate"]["calibration_curve"]
    
    nom_levels = np.array(cal_pcf["nominal_levels"]) * 100
    pcf_cov_ov = np.array(cal_pcf["overall_coverages"]) * 100
    pcf_cov_ms = np.array(cal_pcf["missing_coverages"]) * 100
    cat_cov_ms = np.array(cal_cat["missing_coverages"]) * 100
    
    ax2.plot(nom_levels, nom_levels, "k--", linewidth=1.8, label="Ideal Calibration (1:1 Line)")
    ax2.plot(nom_levels, pcf_cov_ms, "o-", color="#DD8452", linewidth=2.2, markersize=7, label="PathConFuse (Structured Missing Omics)")
    ax2.plot(nom_levels, pcf_cov_ov, "s-", color="#4C72B0", linewidth=2.0, markersize=6, label="PathConFuse (Overall Cohort)")
    ax2.plot(nom_levels, cat_cov_ms, "^-.", color="#C62828", linewidth=1.8, markersize=6, label="Ablation w/o Gate (Missingness Undercoverage)")
    
    ax2.fill_between(nom_levels, nom_levels, 100, color="#E8F5E9", alpha=0.35)
    ax2.text(71.5, 87.5, "Target Calibration Region\n(Coverage ≥ Nominal Level)", fontsize=8.5, color="#2E7D32", fontweight="bold")
    ax2.text(88, 78, "Empirical Undercoverage\n(Safety Risk)", fontsize=8.5, color="#C62828", fontweight="bold")
    
    ax2.set_xlabel("Nominal Confidence Level 1 - α (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Empirical Conformal Coverage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Conformal Coverage Reliability & Calibration Curve", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_xlim(68, 97)
    ax2.set_ylim(70, 100)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#B0BEC5")
    
    plt.tight_layout()
    fig_dir = os.environ.get("FIG_OUTPUT_DIR", os.path.abspath(os.path.join(base_dir, "../../figures")))
    os.makedirs(fig_dir, exist_ok=True)
    pdf_out = os.path.join(fig_dir, "fig3_km_calibration.pdf")
    png_out = os.path.join(fig_dir, "fig3_km_calibration.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out, dpi=300)
    plt.close()
    print(f"Generated {pdf_out} and {png_out} successfully.")

if __name__ == "__main__":
    main()
