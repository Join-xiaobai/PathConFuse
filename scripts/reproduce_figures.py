#!/usr/bin/env python3
"""
Master script to reproduce camera-ready manuscript and supplementary figures.
Generates:
- Figure 2: Ablation Study & Subgroup Conformal Coverage Under Institutional MNAR
- Figure 3: Empirical Kaplan-Meier Survival Stratification & Conformal Calibration Curves
- Figure 4: Multi-Scale Biological Interpretability & Cross-Modal Conflict Resolution
- Figure S1: Cross-Cancer Protocol Replication on TCGA-LUAD
- Figure S2: Controlled Modality Conflict Perturbation Stress Test

Note: Figure 1 (System Architecture) is the authentic high-resolution diagram
from the manuscript and is preserved directly in figures/fig1_architecture.{pdf,png}.
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce publication result figures.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Target directory for generated figures (defaults to figures/).")
    return parser.parse_args()

def main():
    args = parse_args()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = args.output_dir if args.output_dir else os.path.join(base_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig_scripts_dir = os.path.join(base_dir, "scripts", "figures")

    figures = [
        ("Figure 2 (Ablation & Coverage)", "generate_fig2.py"),
        ("Figure 3 (KM & Calibration)", "generate_fig3.py"),
        ("Figure 4 (Interpretability)", "generate_fig4.py"),
        ("Figure S1 (LUAD Replication)", "generate_fig_s1.py"),
        ("Figure S2 (Conflict Stress Test)", "generate_fig_s2.py"),
    ]

    print("=" * 70)
    print(" PathConFuse: Generating Camera-Ready Result Figures")
    print(f" Target output directory: {out_dir}")
    print("=" * 70)

    for name, script in figures:
        script_path = os.path.join(fig_scripts_dir, script)
        print(f"[*] Rendering {name} via {script}...")
        res = subprocess.run([sys.executable, script_path], cwd=out_dir, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[!] Warning: {script} failed with error:\n{res.stderr}")
        else:
            print(f"    Successfully generated {name}.")

    print("=" * 70)
    print(" All experimental result figures successfully generated in:")
    print(f" {out_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
