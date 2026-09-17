#!/usr/bin/env python3
"""Figure 2: Ablation Study & Subgroup Conformal Coverage."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

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
        if os.path.exists(os.path.join(c, "results_manifest.json")):
            return c
    raise FileNotFoundError("Could not find results_manifest.json in candidate paths.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = find_results_dir()
    manifest_path = os.path.join(res_dir, "results_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    brca_results = manifest["tcga_brca_benchmarks"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.6), dpi=300)
    ablation_names = ["PathConFuse\n(Full Proposed)", "w/o Conflict\nGate", "w/o Pathway\nGraph", "Baseline\nConcat"]
    ablation_keys = ["PathConFuse (Full Proposed)", "Ablation: w/o Conflict Gate", "Ablation: w/o Pathway Graph", "Concat (Flat MLP w/o Gate)"]
    
    c_indices = [brca_results[k]["c_index"] for k in ablation_keys]
    c_ci_lo = [brca_results[k]["c_index_ci"][0] for k in ablation_keys]
    c_ci_hi = [brca_results[k]["c_index_ci"][1] for k in ablation_keys]
    c_err_lo = [c_indices[i] - c_ci_lo[i] for i in range(4)]
    c_err_hi = [c_ci_hi[i] - c_indices[i] for i in range(4)]
    
    # Subplot (a) - Discriminative Performance
    bars1 = ax1.bar(np.arange(4), c_indices, yerr=[c_err_lo, c_err_hi], capsize=5,
                    color=["#2E7D32", "#4C72B0", "#E65100", "#C62828"], edgecolor="black", alpha=0.88, width=0.52)
    ax1.set_ylabel("Harrell's C-Index (95% CI)", fontsize=11.5, fontweight="bold")
    ax1.set_title("(a) Discriminative Performance Across Architectural Ablations", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels(ablation_names, fontsize=10, fontweight="bold")
    ax1.set_ylim(0.48, 0.88)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    # Text labels above the top error bar cap to avoid any line collisions
    for i, (b, v, top_err) in enumerate(zip(bars1, c_indices, c_ci_hi)):
        ax1.text(b.get_x() + b.get_width()/2, top_err + 0.012, f"{v:.4f}\n[{c_ci_lo[i]:.4f}, {c_ci_hi[i]:.4f}]",
                 ha="center", va="bottom", fontsize=8.6, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FDFEFE", edgecolor="#BDC3C7", alpha=0.95))

    # Subplot (b) - Conformal Coverage
    x_pos = np.arange(4)
    w = 0.32
    cov_ov = [brca_results[k]["overall_coverage"] * 100 for k in ablation_keys]
    cov_ms = [brca_results[k]["meth_missing_coverage"] * 100 for k in ablation_keys]
    
    b1 = ax2.bar(x_pos - w/2, cov_ov, width=w, label="Overall Cohort Coverage", color="#3470A3", edgecolor="black", alpha=0.88)
    b2 = ax2.bar(x_pos + w/2, cov_ms, width=w, label="Missing Subgroup Coverage", color="#E67E22", edgecolor="black", alpha=0.88)
    line_target = ax2.axhline(90.0, color="#D32F2F", linestyle="--", linewidth=1.8, label="90% Nominal Target Level")
    
    # Value annotations on top of bars
    for bar, val in zip(b1, cov_ov):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#1B4F72")
    for bar, val in zip(b2, cov_ms):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#935116")
        
    ax2.set_ylabel("Empirical Conformal Coverage (%)", fontsize=11.5, fontweight="bold")
    ax2.set_title("(b) Conformal Coverage Under Institutional Missingness", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ablation_names, fontsize=10, fontweight="bold")
    ax2.set_ylim(75, 114)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    # Elegant single-row legend in the clean upper whitespace
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=8.2, frameon=True,
               facecolor="#F8F9FA", edgecolor="#BDC3C7", framealpha=0.95, columnspacing=0.8)
    
    plt.tight_layout()
    fig_dir = os.environ.get("FIG_OUTPUT_DIR", os.path.abspath(os.path.join(base_dir, "../../figures")))
    os.makedirs(fig_dir, exist_ok=True)
    pdf_out = os.path.join(fig_dir, "fig2_ablation_coverage.pdf")
    png_out = os.path.join(fig_dir, "fig2_ablation_coverage.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out, dpi=300)
    plt.close()
    print(f"Generated {pdf_out} and {png_out} successfully.")

if __name__ == "__main__":
    main()
