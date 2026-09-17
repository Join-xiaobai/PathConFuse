import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_framework():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Colors
    c_wsi = "#2E86AB"      # Blue
    c_omics = "#A23B72"    # Purple/Burgundy
    c_pathway = "#F18F01"  # Amber/Orange
    c_gate = "#C73E1D"     # Coral Red
    c_conformal = "#3B7A57"# Forest Green
    c_box = "#F4F5F7"      # Light Gray
    c_border = "#4A4A4A"

    # Title
    ax.text(7, 7.6, "PathConFuse: Pathway-Constrained Conflict-Aware Multimodal Survival Architecture", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1D2A44")

    # 1. Modality 1: Gigapixel Pathology (WSI)
    rect_wsi = patches.FancyBboxPatch((0.5, 4.8), 3.2, 2.3, boxstyle="round,pad=0.15", 
                                      facecolor="#EBF4F9", edgecolor=c_wsi, linewidth=1.8)
    ax.add_patch(rect_wsi)
    ax.text(2.1, 6.8, "Diagnostic Pathology (WSI)", ha="center", va="center", fontsize=12, fontweight="bold", color=c_wsi)
    ax.text(2.1, 6.3, "• Patient Whole-Slide Image\n• Tissue Patch Grid ($N_i \\times 768$)\n• Gated Attention MIL (ABMIL)", 
            ha="center", va="center", fontsize=9.5, color="#333333")
    
    # Sub-box for WSI representation
    rect_zwsi = patches.FancyBboxPatch((0.8, 5.0), 2.6, 0.6, boxstyle="round,pad=0.08", 
                                       facecolor=c_wsi, edgecolor="none")
    ax.add_patch(rect_zwsi)
    ax.text(2.1, 5.3, "Histology Latent $z_i^I \\in \\mathbb{R}^{256}$ & Logit $h_i^I$", ha="center", va="center", 
            fontsize=9.5, fontweight="bold", color="white")

    # 2. Modality 2: Omics + Reactome Knowledge Graph
    rect_omics = patches.FancyBboxPatch((0.5, 0.8), 3.2, 3.5, boxstyle="round,pad=0.15", 
                                        facecolor="#FBF0F5", edgecolor=c_omics, linewidth=1.8)
    ax.add_patch(rect_omics)
    ax.text(2.1, 4.0, "Multi-Omics & Biology Knowledge", ha="center", va="center", fontsize=12, fontweight="bold", color=c_omics)
    ax.text(2.1, 3.2, "• 4,999 Bulk RNA Expressions $X_i^R$\n• DNA Methylation Indicator $m_i^M$\n• Reactome Bipartite Graph $\\tilde{M}$\n  (331 Curated Biological Pathways)\n• 2-layer Pathway Transformer", 
            ha="center", va="center", fontsize=9.5, color="#333333")
    
    # Sub-box for Omics representation
    rect_zomics = patches.FancyBboxPatch((0.8, 1.1), 2.6, 0.6, boxstyle="round,pad=0.08", 
                                         facecolor=c_omics, edgecolor="none")
    ax.add_patch(rect_zomics)
    ax.text(2.1, 1.4, "Pathway Latent $z_i^R \\in \\mathbb{R}^{256}$ & Logit $h_i^R$", ha="center", va="center", 
            fontsize=9.5, fontweight="bold", color="white")

    # Arrows to Conflict Detector
    ax.annotate("", xy=(4.4, 5.3), xytext=(3.7, 5.3),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=c_wsi))
    ax.annotate("", xy=(4.4, 1.4), xytext=(3.7, 1.4),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=c_omics))

    # 3. Middle Block: Modality Conflict Gate
    rect_gate = patches.FancyBboxPatch((4.5, 2.0), 3.8, 4.0, boxstyle="round,pad=0.2", 
                                       facecolor="#FFF5F2", edgecolor=c_gate, linewidth=2.0)
    ax.add_patch(rect_gate)
    ax.text(6.4, 5.6, "Modality Conflict Gate", ha="center", va="center", fontsize=13, fontweight="bold", color=c_gate)
    
    # Details in Gate
    ax.text(6.4, 4.7, "Disagreement Metric:\n$d_i^{I, R} = |r_i^I - r_i^R|$", 
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#7A1C06",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDE8E1", edgecolor="none"))
    
    ax.text(6.4, 3.5, "Availability-Constrained Gating:\n$\\alpha_i^k = \\frac{\\tilde{\\alpha}_i^k \\cdot m_i^k}{\\sum_j \\tilde{\\alpha}_i^j m_i^j + \\epsilon}$\nStrict Zero-Weight on Missing Assays\n(Guards Against Missingness Bias)", 
            ha="center", va="center", fontsize=9.2, color="#4A4A4A")

    rect_fused_vec = patches.FancyBboxPatch((4.9, 2.3), 3.0, 0.55, boxstyle="round,pad=0.08", 
                                           facecolor=c_gate, edgecolor="none")
    ax.add_patch(rect_fused_vec)
    ax.text(6.4, 2.57, "Adaptive Fusion: $z_i^{\\text{bio}} = \\sum_k \\alpha_i^k z_i^k$", 
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")

    # Arrow to Survival Head
    ax.annotate("", xy=(9.0, 4.0), xytext=(8.3, 4.0),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#333333"))

    # 4. Joint Survival Head
    rect_head = patches.FancyBboxPatch((9.1, 2.6), 4.4, 2.8, boxstyle="round,pad=0.18", 
                                       facecolor="#F4FAF6", edgecolor=c_conformal, linewidth=1.8)
    ax.add_patch(rect_head)
    ax.text(11.3, 5.0, "Trustworthy Survival Head & Conformal Bounds", ha="center", va="center", 
            fontsize=12, fontweight="bold", color=c_conformal)
    
    ax.text(11.3, 4.1, "• Fused Discrete Hazards: $h_i^{\\text{fused}} = \\text{Head}([z_i^{\\text{bio}}, z_i^C])$\n• IPCW Weighting via Censoring KM $\\hat{G}(t)$\n• Calibration Score: $S_i = \\frac{1}{\\hat{G}(T_i)}[\\delta_i |T_i - \\hat{T}_i| + ...]$",
            ha="center", va="center", fontsize=9.2, color="#333333")

    rect_interval = patches.FancyBboxPatch((9.4, 2.8), 3.8, 0.6, boxstyle="round,pad=0.08", 
                                           facecolor=c_conformal, edgecolor="none")
    ax.add_patch(rect_interval)
    ax.text(11.3, 3.1, "Calibrated Interval: $[\\hat{T}_i - \\hat{q}, \\, \\hat{T}_i + \\hat{q}]$ (90% Nominal)", 
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")

    # 5. Clinical Safety Annotation at bottom
    ax.text(7.0, 0.35, "★ Key Feature: Under missing omics assays (structured missingness) or high morphology-genotype conflict,\nPathConFuse broadens interval margin $\\hat{q}$ and avoids overconfident risk extrapolation.",
            ha="center", va="center", fontsize=10, style="italic", color="#2C3E50",
            bbox=dict(boxstyle="square,pad=0.4", facecolor="#EAECEE", edgecolor="#BDC3C7", lw=1))

    plt.tight_layout()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.environ.get("FIG_OUTPUT_DIR", os.path.abspath(os.path.join(base_dir, "../../figures")))
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "fig1_architecture.png"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig1_architecture.pdf"))
    plt.close()
    print("Figure 1 generated successfully!")

if __name__ == "__main__":
    draw_framework()
