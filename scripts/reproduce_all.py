#!/usr/bin/env python3
"""
Master End-to-End Replication Script for PathConFuse.
Executes all experimental evaluation modules, statistical audits,
and publication figure reproductions sequentially.
"""

import os
import sys
import subprocess

def run_script(script_name, args=None):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(scripts_dir, script_name)]
    if args:
        cmd.extend(args)
    print(f"\n{'='*75}")
    print(f"[*] Running: {' '.join(cmd)}")
    print(f"{'='*75}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[!] Warning: {script_name} exited with code {result.returncode}")
    return result.returncode

def main():
    print("=" * 75)
    print(" PathConFuse: Master End-to-End Experimental Replication Pipeline")
    print("=" * 75)

    # 1. Zero-data quick smoke test
    run_script("demo_smoke_test.py")

    # 2. Gate 0 multi-center cohort & MNAR audit
    run_script("run_gate0_audit.py")

    # 3. Main benchmark evaluation & conformal calibration
    run_script("evaluate.py")

    # 4. Systematic ablation study analysis
    run_script("run_ablations.py", ["--from_results"])

    # 5. Modality conflict perturbation stress test
    run_script("run_stress_test.py")

    # 6. TCGA-LUAD cross-cancer generalization evaluation
    run_script("run_luad_eval.py")

    # 7. Camera-ready figures reproduction
    run_script("reproduce_figures.py")

    print("\n" + "=" * 75)
    print(" Complete experimental reproduction pipeline finished successfully!")
    print("=" * 75)

if __name__ == "__main__":
    main()
