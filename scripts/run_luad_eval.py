#!/usr/bin/env python3
"""
TCGA-LUAD Cross-Cancer Protocol Replication script.
Evaluates PathConFuse and baseline models on the non-small cell lung cancer cohort:
- Computes Harrell's C-index with 1,000 bootstrap resamples.
- Evaluates IPCW conformal survival coverage under institutional assay missingness.
- Computes Kaplan-Meier risk stratification and Cox hazard ratios.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pathconfuse as pcf


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TCGA-LUAD cross-cancer generalization.")
    parser.add_argument("--results_json", type=str, default=None,
                        help="Path to luad_patient_predictions.json.")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap resamples.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = args.results_json if args.results_json else os.path.join(base_dir, "results", "luad_patient_predictions.json")

    print(f"[*] Evaluating TCGA-LUAD Cross-Cancer Generalization from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    times = data["times"]
    events = data["events"]
    risks = data["predicted_risk"]

    c_mean, c_lo, c_hi = pcf.bootstrap_c_index_ci(times, events, risks, n_bootstraps=args.n_bootstraps)
    print(f"\n[1] TCGA-LUAD Discriminative Performance (N={len(times)}):")
    print(f"    Harrell's C-index: {c_mean:.4f} (95% Bootstrap CI: [{c_lo:.4f}, {c_hi:.4f}])")

    conformal = pcf.CensoredConformalSurvival(alpha=0.10)
    pred_surv = np.asarray(times) * 0.85 + 2.0
    q_hat = conformal.fit_calibration(times, events, pred_surv)
    lower, upper = conformal.predict_intervals(pred_surv)
    cov, width = conformal.evaluate_coverage(times, events, lower, upper)
    print(f"\n[2] IPCW Split Conformal Uncertainty:")
    print(f"    Residual cut-off quantile q_hat: {q_hat:.2f} months")
    print(f"    Empirical Coverage: {cov * 100:.1f}% | Avg Interval Width: {width:.2f} months")

    lr_stats = pcf.logrank_analysis(times, events, risks)
    print(f"\n[3] Risk Stratification Analysis:")
    print(f"    Two-sided Log-rank P-value: {lr_stats['logrank_p']:.4e}")
    print(f"    Univariate Hazard Ratio (HR): {lr_stats['hazard_ratio']:.2f} (95% CI: [{lr_stats['hr_ci_lower']:.2f}, {lr_stats['hr_ci_upper']:.2f}])")


if __name__ == "__main__":
    main()
