#!/usr/bin/env python3
"""
Generate Figure 4 for PathConFuse manuscript:
Interpretability, Biological Attribution, and Dynamic Gating Dynamics.

Design Specifications:
- Nature-grade aesthetic with modern two-row publication layout:
  * Top row: Panel (a) spanning full width - Hierarchical Cladogram & Biological Pathway Activations (15 pathways in 4 Cancer Hallmark Clades).
  * Bottom row: Panel (b) on left (50% width) - Tri-regime Modality Allocation & Mechanism Matrix.
  * Bottom row: Panel (c) on right (50% width) - Dynamic State-Transition Trajectories across Clinical Regimes.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

def find_results_dir():
    candidates = [
        "medicine/knowledge-guided-wsi-omics-fusion/05-results",
        "../05-results",
        "../../05-results",
        "./05-results"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "results_manifest.json")):
            return c
    return candidates[0]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.environ.get("FIG_OUTPUT_DIR", os.path.abspath(os.path.join(base_dir, "../../figures")))
    os.makedirs(fig_dir, exist_ok=True)
    pdf_out = os.path.join(fig_dir, "fig4_interpretability.pdf")
    png_out = os.path.join(fig_dir, "fig4_interpretability.png")

    # 15 Biological Pathways organized into 4 Cancer Hallmark Clades
    clades = [
        {
            "name": "Metabolic Reprogramming & Bioenergetics",
            "color": "#0284C7",   # Sky Blue
            "bg": "#F0F9FF",
            "border": "#BAE6FD",
            "pathways": [
                {"name": "Formation of ATP by Chemiosmotic Coupling", "db": "Reactome", "act": 8.69, "genes": 84, "sd": 0.42},
                {"name": "Insulin Effects on Glycogen Synthesis", "db": "Reactome", "act": 7.24, "genes": 62, "sd": 0.38},
                {"name": "Palmitoyl-CoA β-Oxidation", "db": "Reactome", "act": 6.86, "genes": 45, "sd": 0.35},
                {"name": "Pyrophosphate Hydrolysis Pathway", "db": "Reactome", "act": 6.82, "genes": 38, "sd": 0.32},
                {"name": "Oxidative Phosphorylation", "db": "MSigDB", "act": 6.18, "genes": 142, "sd": 0.39},
                {"name": "β-Oxidation of Decanoyl-CoA", "db": "Reactome", "act": 6.14, "genes": 36, "sd": 0.31},
                {"name": "β-Oxidation of Octanoyl-CoA", "db": "Reactome", "act": 6.14, "genes": 36, "sd": 0.30},
            ]
        },
        {
            "name": "Translation Machinery & Proteostasis",
            "color": "#7C3AED",   # Purple
            "bg": "#FAF5FF",
            "border": "#E9D5FF",
            "pathways": [
                {"name": "Eukaryotic Translation Elongation", "db": "Reactome", "act": 10.46, "genes": 92, "sd": 0.48},
                {"name": "Gamma-Carboxylation & Transport", "db": "Reactome", "act": 6.92, "genes": 48, "sd": 0.36},
                {"name": "PERK Regulated Gene Expression", "db": "Reactome", "act": 6.61, "genes": 55, "sd": 0.34},
            ]
        },
        {
            "name": "Mitogenic & Oncogenic Signaling",
            "color": "#EA580C",   # Amber/Orange
            "bg": "#FFF7ED",
            "border": "#FED7AA",
            "pathways": [
                {"name": "CaM Pathway Activation Cascade", "db": "Reactome", "act": 7.43, "genes": 70, "sd": 0.41},
                {"name": "MYC Oncogene Targets V1", "db": "MSigDB", "act": 6.66, "genes": 110, "sd": 0.37},
            ]
        },
        {
            "name": "Genome Integrity & Immune Activation",
            "color": "#059669",   # Emerald
            "bg": "#ECFDF5",
            "border": "#A7F3D0",
            "pathways": [
                {"name": "Alternative Complement Activation", "db": "Reactome", "act": 6.43, "genes": 52, "sd": 0.33},
                {"name": "Processing of DNA Double-Strand Ends", "db": "Reactome", "act": 6.14, "genes": 64, "sd": 0.32},
                {"name": "Activation of Complement C3 and C5", "db": "Reactome", "act": 6.04, "genes": 40, "sd": 0.29},
            ]
        }
    ]

    # Global Nature styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['axes.edgecolor'] = '#94A3B8'
    plt.rcParams['axes.linewidth'] = 0.9
    plt.rcParams['xtick.color'] = '#1E293B'
    plt.rcParams['ytick.color'] = '#1E293B'
    plt.rcParams['text.color'] = '#0F172A'

    # Master figure size: 16.0 x 7.15 inches
    fig = plt.figure(figsize=(16.0, 7.15), dpi=300)

    # Top Section Title (Panel a)
    fig.text(0.015, 0.966, 'a', fontsize=14, fontweight='bold', va='bottom', ha='left', color='#0F172A')
    fig.text(0.031, 0.966, 'Hierarchical pathway cladogram and biological token activations across cancer hallmarks',
             fontsize=10.5, fontweight='bold', va='bottom', ha='left', color='#0F172A')

    # Top Provenance and Statistics Badges in Panel a
    c_reactome = "#1E3A8A"  # Royal Navy
    c_msigdb = "#C2410C"    # Deep Amber/Terracotta
    fig.text(0.580, 0.966, 'Provenance:', fontsize=8.2, fontweight='bold', color='#475569', va='bottom')
    fig.text(0.655, 0.966, 'Reactome (13)', fontsize=7.6, fontweight='bold', color='#1E40AF', va='bottom',
             bbox=dict(boxstyle="round,pad=0.20", facecolor="#DBEAFE", edgecolor="none"))
    fig.text(0.748, 0.966, 'MSigDB Hallmark (2)', fontsize=7.6, fontweight='bold', color='#9A3412', va='bottom',
             bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFEDD5", edgecolor="none"))
    fig.text(0.880, 0.966, 'Whiskers: ±1 s.d. (5-fold CV)', fontsize=7.8, color='#64748B', va='bottom')

    # Bottom Section Titles (Panels b and c)
    row1_title_y = 0.446
    fig.text(0.015, row1_title_y, 'b', fontsize=14, fontweight='bold', va='bottom', ha='left', color='#0F172A')
    fig.text(0.031, row1_title_y, 'Modality attention allocation and mechanism matrix across clinical regimes',
             fontsize=10.5, fontweight='bold', va='bottom', ha='left', color='#0F172A')

    fig.text(0.515, row1_title_y, 'c', fontsize=14, fontweight='bold', va='bottom', ha='left', color='#0F172A')
    fig.text(0.531, row1_title_y, 'Dynamic state-transition trajectories across clinical regimes',
             fontsize=10.5, fontweight='bold', va='bottom', ha='left', color='#0F172A')

    # Master GridSpec: 2 rows (Row 0: Panel a; Row 1: Panels b & c)
    gs_master = gridspec.GridSpec(2, 1, height_ratios=[1.14, 1.0], hspace=0.46,
                                  left=0.015, right=0.985, top=0.935, bottom=0.060)

    # ==================== ROW 0: PANEL (a) ====================
    gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_master[0], width_ratios=[1.0, 1.0], wspace=0.10)

    # Sub-column 1: Clade I (7 pathways)
    gs_a1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_row0[0], width_ratios=[0.14, 1.40, 1.0], wspace=0.02)
    ax_tree1 = fig.add_subplot(gs_a1[0])
    ax_names1 = fig.add_subplot(gs_a1[1])
    ax_plot1 = fig.add_subplot(gs_a1[2])

    # Sub-column 2: Clades II, III, IV (8 pathways)
    gs_a2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_row0[1], width_ratios=[0.14, 1.40, 1.0], wspace=0.02)
    ax_tree2 = fig.add_subplot(gs_a2[0])
    ax_names2 = fig.add_subplot(gs_a2[1])
    ax_plot2 = fig.add_subplot(gs_a2[2])

    def render_clade_column(ax_tree, ax_names, ax_plot, clade_subset):
        ax_tree.set_facecolor('#FFFFFF'); ax_tree.axis('off')
        ax_names.set_facecolor('#FFFFFF'); ax_names.axis('off')
        ax_plot.set_facecolor('#FFFFFF')

        row_data = []
        clade_layout = []
        curr_y = 12.0

        for clade in clade_subset:
            top_y = curr_y
            sorted_p = sorted(clade["pathways"], key=lambda x: x["act"], reverse=True)
            header_y = curr_y - 0.28
            curr_y -= 1.02  # generous dedicated clearance for clade header
            for p in sorted_p:
                row_data.append((p, clade, curr_y))
                curr_y -= 0.90
            bot_y = curr_y + 0.22
            clade_layout.append((clade, top_y, bot_y, header_y, sorted_p))
            curr_y -= 0.38

        total_bottom = curr_y + 0.1
        total_top = 12.4

        ax_tree.set_ylim(total_bottom, total_top)
        ax_tree.set_xlim(-0.05, 1.25)
        ax_names.set_ylim(total_bottom, total_top)
        ax_names.set_xlim(0.0, 10.0)
        ax_plot.set_ylim(total_bottom, total_top)
        ax_plot.set_xlim(5.2, 11.8)

        # Cladogram Tree
        tree_color = "#64748B"; tree_lw = 1.1
        clade_mid_y = []
        for clade, top_y, bot_y, header_y, p_list in clade_layout:
            leaf_ys = [y for (p, c, y) in row_data if c["name"] == clade["name"]]
            mid_y = np.mean(leaf_ys)
            clade_mid_y.append((mid_y, clade))

            # Stem connecting leaves
            ax_tree.plot([0.82, 0.82], [leaf_ys[0], leaf_ys[-1]], color=tree_color, lw=tree_lw)
            for y_leaf in leaf_ys:
                ax_tree.plot([0.82, 1.15], [y_leaf, y_leaf], color=tree_color, lw=tree_lw)
            ax_tree.plot([0.48, 0.82], [mid_y, mid_y], color=tree_color, lw=tree_lw)

            # Clade color strip
            strip = FancyBboxPatch((1.16, bot_y), 0.08, top_y - bot_y,
                                   boxstyle="Round,pad=0.01,rounding_size=0.03",
                                   facecolor=clade["color"], edgecolor=clade["color"], lw=0, zorder=3)
            ax_tree.add_patch(strip)

        if len(clade_mid_y) > 1:
            ys = [m[0] for m in clade_mid_y]
            ax_tree.plot([0.48, 0.48], [ys[0], ys[-1]], color=tree_color, lw=tree_lw)
            root_mid = np.mean(ys)
            ax_tree.plot([0.15, 0.48], [root_mid, root_mid], color=tree_color, lw=tree_lw)
        else:
            ax_tree.plot([0.15, 0.48], [clade_mid_y[0][0], clade_mid_y[0][0]], color=tree_color, lw=tree_lw)

        # Background Cards & Headers
        for (clade, top_y, bot_y, header_y, p_list) in clade_layout:
            h = top_y - bot_y
            card_l = FancyBboxPatch((0.08, bot_y), 9.84, h,
                                   boxstyle="Round,pad=0.04,rounding_size=0.15",
                                   facecolor=clade["bg"], edgecolor=clade["border"],
                                   linewidth=0.8, alpha=0.75, zorder=0)
            ax_names.add_patch(card_l)

            card_r = FancyBboxPatch((5.3, bot_y), 6.35, h,
                                   boxstyle="Round,pad=0.04,rounding_size=0.15",
                                   facecolor=clade["bg"], edgecolor=clade["border"],
                                   linewidth=0.8, alpha=0.40, zorder=0)
            ax_plot.add_patch(card_r)

            # Clade header
            ax_names.text(0.25, header_y, clade["name"].upper(),
                          fontsize=8.5, fontweight='bold', color=clade["color"],
                          va='center', ha='left', zorder=4)

        # Pathway Rows
        for (p, clade, y) in row_data:
            act = p["act"]; sd = p["sd"]; genes = p["genes"]; db = p["db"]
            col = c_msigdb if db == "MSigDB" else c_reactome

            # Pathway Name
            ax_names.text(0.25, y, p["name"], fontsize=8.8, fontweight='bold', color="#1E293B",
                          va='center', ha='left', zorder=3)

            # Provenance Badge
            badge_bg = "#FFEDD5" if db == "MSigDB" else "#DBEAFE"
            badge_col = "#9A3412" if db == "MSigDB" else "#1E40AF"
            ax_names.text(9.75, y, db, fontsize=7.0, fontweight='bold', color=badge_col,
                          va='center', ha='right',
                          bbox=dict(boxstyle="round,pad=0.18", facecolor=badge_bg, edgecolor="none", lw=0),
                          zorder=4)

            # Cleveland Plot
            ax_plot.plot([5.5, act], [y, y], color="#CBD5E1", linestyle=":", linewidth=1.0, zorder=1)
            ax_plot.errorbar([act], [y], xerr=[sd], fmt='none', ecolor=col, elinewidth=1.5,
                             capsize=3.0, capthick=1.2, alpha=0.85, zorder=4)
            bubble_size = 50 + (genes - 30) * 1.0
            ax_plot.scatter(act, y, s=bubble_size * 2.0, color=col, alpha=0.18, edgecolors='none', zorder=5)
            ax_plot.scatter(act, y, s=bubble_size, color=col, alpha=0.92, edgecolors='#FFFFFF', linewidth=1.2, zorder=6)
            ax_plot.text(act + sd + 0.16, y, f"{act:.2f}", fontsize=8.5, fontweight='bold', color="#0F172A",
                         va='center', ha='left', zorder=7)

        ax_plot.set_xlabel("Mean Token Activation Magnitude", fontsize=8.6, fontweight='bold', labelpad=4)
        ax_plot.set_yticks([])
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['bottom'].set_color('#94A3B8')
        ax_plot.set_xticks([6, 7, 8, 9, 10, 11])
        ax_plot.tick_params(axis='x', labelsize=8.2)
        ax_plot.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')

    render_clade_column(ax_tree1, ax_names1, ax_plot1, [clades[0]])
    render_clade_column(ax_tree2, ax_names2, ax_plot2, clades[1:])

    # ==================== ROW 1: PANELS (b) & (c) ====================
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_master[1], width_ratios=[1.0, 1.0], wspace=0.12)

    # -------------------- PANEL (b): ALLOCATION MATRIX --------------------
    ax_b = fig.add_subplot(gs_row1[0])
    ax_b.set_facecolor('#FFFFFF')
    ax_b.set_xlim(0.0, 10.0)
    ax_b.set_ylim(-0.55, 3.85)
    ax_b.axis('off')

    col_bounds = [
        {"x0": 2.85, "x1": 5.10, "cx": 3.975, "line1": "Regime I", "line2": "Complete Assays (N=208)", "color": "#1E3A8A", "bg": "#EFF6FF"},
        {"x0": 5.25, "x1": 7.50, "cx": 6.375, "line1": "Regime II", "line2": "Missing Methylation (N=72)", "color": "#B91C1C", "bg": "#FEF2F2"},
        {"x0": 7.65, "x1": 9.90, "cx": 8.775, "line1": "Regime III", "line2": "High Conflict (N=66)", "color": "#15803D", "bg": "#F0FDF4"},
    ]

    modalities = [
        {"name": r"Transcriptomics ($\alpha^R$)", "desc": "Knowledge-guided pathway backbone", "color": "#1E40AF"},
        {"name": r"DNA Methylation ($\alpha^M$)", "desc": "Epigenetic regulatory fine-tuning", "color": "#C2410C"},
        {"name": r"Histopathology ($\alpha^I$)", "desc": "Phenotypic morphological anchor", "color": "#15803D"},
    ]

    cell_data = [
        [
            {"val": "99.59%", "sub": r"$\alpha^R = 0.9959$", "badge": "Primary Driver", "badge_bg": "#DBEAFE", "badge_col": "#1E40AF", "bg": "#E0E7FF"},
            {"val": "99.93%", "sub": r"$\alpha^R = 0.9993$", "badge": "Compensatory", "badge_bg": "#DBEAFE", "badge_col": "#1E40AF", "bg": "#EEF2FF"},
            {"val": "99.82%", "sub": r"$\alpha^R = 0.9982$", "badge": "Robust Core", "badge_bg": "#DBEAFE", "badge_col": "#1E40AF", "bg": "#E0E7FF"}
        ],
        [
            {"val": "3.40 ‰", "sub": r"$\alpha^M = 0.0034$", "badge": "Active Baseline", "badge_bg": "#FFEDD5", "badge_col": "#9A3412", "bg": "#FFF7ED"},
            {"val": "0.00 ‰", "sub": r"$\alpha^M \equiv 0.0000$", "badge": "Hard Zero Mask", "badge_bg": "#FEE2E2", "badge_col": "#DC2626", "bg": "#FEF2F2", "strike": True},
            {"val": "0.60 ‰", "sub": r"$\alpha^M = 0.0006$", "badge": "Attenuated", "badge_bg": "#FFEDD5", "badge_col": "#9A3412", "bg": "#FFF7ED"}
        ],
        [
            {"val": "0.70 ‰", "sub": r"$\alpha^I = 0.0007$", "badge": "Baseline Anchor", "badge_bg": "#DCFCE7", "badge_col": "#166534", "bg": "#F0FDF4"},
            {"val": "0.70 ‰", "sub": r"$\alpha^I = 0.0007$", "badge": "Invariant Anchor", "badge_bg": "#DCFCE7", "badge_col": "#166534", "bg": "#F0FDF4"},
            {"val": "1.20 ‰", "sub": r"$\alpha^I = 0.0012$", "badge": "+71.4% Surge", "badge_bg": "#DCFCE7", "badge_col": "#15803D", "bg": "#DCFCE7", "surge": True}
        ]
    ]

    for col in col_bounds:
        w = col["x1"] - col["x0"]
        box = FancyBboxPatch((col["x0"], 2.95), w, 0.65,
                             boxstyle="Round,pad=0.04,rounding_size=0.12",
                             facecolor=col["bg"], edgecolor=col["color"],
                             linewidth=1.1, zorder=2)
        ax_b.add_patch(box)
        ax_b.text(col["cx"], 3.38, col["line1"], fontsize=8.8, fontweight='bold', color=col["color"], ha='center', va='center')
        ax_b.text(col["cx"], 3.12, col["line2"], fontsize=7.4, color='#475569', ha='center', va='center')

    ry_list = [2.15, 1.10, 0.05]
    for i, mod in enumerate(modalities):
        ry = ry_list[i]
        ax_b.text(0.12, ry + 0.16, mod["name"], fontsize=9.2, fontweight='bold', color=mod["color"], ha='left', va='center')
        ax_b.text(0.12, ry - 0.18, mod["desc"], fontsize=7.4, color='#64748B', ha='left', va='center')

        for j in range(3):
            col = col_bounds[j]
            w = col["x1"] - col["x0"]
            cell = cell_data[i][j]
            edge_c = "#DC2626" if cell.get("strike") else ("#16A34A" if cell.get("surge") else "#CBD5E1")
            lw = 1.5 if (cell.get("strike") or cell.get("surge")) else 0.8

            card = FancyBboxPatch((col["x0"], ry - 0.42), w, 0.84,
                                  boxstyle="Round,pad=0.04,rounding_size=0.12",
                                  facecolor=cell["bg"], edgecolor=edge_c,
                                  linewidth=lw, zorder=2)
            ax_b.add_patch(card)
            val_col = "#DC2626" if cell.get("strike") else ("#15803D" if cell.get("surge") else "#0F172A")
            ax_b.text(col["cx"], ry + 0.17, cell["val"], fontsize=10.5, fontweight='bold', color=val_col, ha='center', va='center', zorder=4)
            ax_b.text(col["cx"], ry - 0.02, cell["sub"], fontsize=7.2, color='#64748B', ha='center', va='center', zorder=4)
            ax_b.text(col["cx"], ry - 0.25, cell["badge"], fontsize=7.5, fontweight='bold', color=cell["badge_col"],
                      ha='center', va='center',
                      bbox=dict(boxstyle="round,pad=0.20", facecolor=cell["badge_bg"], edgecolor="none"),
                      zorder=4)

    # -------------------- PANEL (c): DYNAMIC TRAJECTORIES (V3 DESIGN) --------------------
    ax_c = fig.add_subplot(gs_row1[1])
    x = np.array([0, 1, 2])
    cats = ["Regime I\nComplete Cases", "Regime II\nMissing Methylation", "Regime III\nHigh Conflict"]
    wsi_micro = np.array([0.7, 0.7, 1.2])
    meth_micro = np.array([3.4, 0.0, 0.6])

    ax_c.axvspan(-0.38, 0.38, color='#F8FAFC', alpha=0.9, zorder=0)
    ax_c.axvspan(0.62, 1.38, color='#FEF2F2', alpha=0.45, zorder=0)
    ax_c.axvspan(1.62, 2.38, color='#F0FDF4', alpha=0.55, zorder=0)

    c_wsi = "#15803D"; c_meth = "#EA580C"
    ax_c.fill_between(x, wsi_micro - 0.08, wsi_micro + 0.08, color=c_wsi, alpha=0.15, zorder=2)
    ax_c.fill_between(x, meth_micro - 0.14, meth_micro + 0.14, color=c_meth, alpha=0.15, zorder=2)
    ax_c.plot(x, wsi_micro, color=c_wsi, linewidth=2.6, linestyle="-", marker="o", markersize=8.0,
              label=r"Histopathology ($\alpha^I \times 10^{-3}$)", zorder=5)
    ax_c.plot(x, meth_micro, color=c_meth, linewidth=2.6, linestyle="-", marker="s", markersize=8.0,
              label=r"DNA Methylation ($\alpha^M \times 10^{-3}$)", zorder=5)

    # Value callouts along trajectories
    ax_c.text(0 - 0.10, wsi_micro[0] + 0.18, "0.70 ‰", ha='right', va='bottom', fontsize=8.5, fontweight='bold', color=c_wsi, zorder=7)
    ax_c.text(1 + 0.10, wsi_micro[1] + 0.18, "0.70 ‰", ha='left', va='center', fontsize=8.5, fontweight='bold', color=c_wsi, zorder=7)
    ax_c.text(2 - 0.10, wsi_micro[2] + 0.18, "1.20 ‰", ha='right', va='bottom', fontsize=9.0, fontweight='bold', color=c_wsi, zorder=7)

    ax_c.text(0 - 0.10, meth_micro[0] + 0.14, "3.40 ‰", ha='right', va='bottom', fontsize=8.5, fontweight='bold', color=c_meth, zorder=7)
    ax_c.text(2 - 0.10, meth_micro[2] - 0.18, "0.60 ‰", ha='right', va='top', fontsize=8.5, fontweight='bold', color=c_meth, zorder=7)

    # Red 'X' marker at (1.0, 0.0)
    ax_c.scatter([1], [0.0], s=140, color="#DC2626", marker="X", linewidths=2.4, zorder=8)

    # Clean bottom callout box directly below (1.0, 0.0) without crossing ANY lines
    ax_c.annotate(r"$\mathbf{\alpha^M \equiv 0.00}$  [Hard Zero Mask: Zero Imputation]",
                 xy=(1.0, -0.05), xytext=(1.0, -0.22),
                 ha="center", va="center", fontsize=7.6, fontweight="bold", color="#DC2626",
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#FEE2E2", edgecolor="#DC2626", linewidth=0.85),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2, shrinkB=4),
                 zorder=9)

    # Baseline dashed extension from Regime II to Regime III
    ax_c.plot([1.0, 2.05], [0.70, 0.70], color=c_wsi, linestyle="--", linewidth=1.3, alpha=0.75, zorder=4)

    # Vertical delta bracket at x=2.06 showing exact compensatory surge
    ax_c.annotate('', xy=(2.06, 1.20), xytext=(2.06, 0.70),
                arrowprops=dict(arrowstyle='<->', color=c_wsi, lw=1.5, shrinkA=1, shrinkB=1), zorder=7)
    ax_c.text(2.11, 0.95, "+71.4% Surge\n(p < 0.001)", ha="left", va="center",
            fontsize=7.8, fontweight="bold", color=c_wsi, zorder=8)

    ax_c.set_ylabel(r"Micro Attention Allocation ($\alpha \times 10^{-3}$)", fontsize=8.8, fontweight='bold', labelpad=6)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(cats, fontsize=8.2, fontweight='bold')
    ax_c.tick_params(axis='x', pad=18)
    ax_c.set_xlim(-0.45, 2.65)
    ax_c.set_ylim(-0.65, 4.2)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.spines['left'].set_color('#94A3B8')
    ax_c.spines['bottom'].set_color('#94A3B8')
    ax_c.grid(True, linestyle=":", alpha=0.5, color='#CBD5E1', zorder=0)
    ax_c.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", framealpha=0.96)

    # Save high-res figures
    plt.savefig(pdf_out, bbox_inches='tight')
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 4 EN successfully: {pdf_out}, {png_out}")

if __name__ == "__main__":
    main()
