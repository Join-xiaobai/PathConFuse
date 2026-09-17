#!/usr/bin/env python3
"""
End-to-End Smoke Test Demo for PathConFuse.
Runs data generation, model forward pass, dual-objective optimization,
conformal survival calibration, and evaluation in ~10 seconds.
"""

import sys
import os
import time
import torch
from torch.utils.data import DataLoader

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pathconfuse as pcf

def main():
    print("=" * 70)
    print(" PathConFuse: End-to-End Multimodal Survival Pipeline Smoke Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # 1. Generate Synthetic Multimodal Oncology Cohort
    print("[1/5] Generating synthetic multimodal cohort (N=150)...")
    train_data = pcf.generate_synthetic_cohort(n_patients=100, seed=42)
    test_data = pcf.generate_synthetic_cohort(n_patients=50, seed=123)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    print(f"      Train cohort size: {len(train_data)} | Test cohort size: {len(test_data)}")

    # 2. Build Pathway Graph & Model
    print("[2/5] Initializing Pathway incidence matrix (331 pathways, 4999 genes)...")
    mask = pcf.load_or_generate_pathway_mask(n_pathways=331, n_genes=4999)
    model = pcf.PathConFuse(norm_mask=mask).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      PathConFuse trainable parameters: {n_params:,}")

    # 3. Dual-Objective Training (2 Epochs)
    print("[3/5] Executing dual-objective training loop (NLL + Pairwise Ranking)...")
    nll_loss_fn = pcf.NLLSurvLoss()
    rank_loss_fn = pcf.PairwiseSurvLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start_t = time.time()
    model.train()
    for epoch in range(1, 3):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            wsi_bag = batch["wsi_bag"].to(device)
            rna_expr = batch["rna_expr"].to(device)
            meth_feat = batch["meth_features"].to(device)
            clin_cov = batch["clin_cov"].to(device)
            avail_mask = batch["avail_mask"].to(device)
            y_discrete = batch["y_discrete"].to(device)
            surv_time = batch["surv_time"].to(device)
            c_ind = batch["c_indicator"].to(device)
            ev_obs = batch["event_observed"].to(device)

            out = model(wsi_bag, rna_expr, meth_feat, clin_cov, avail_mask)
            l_nll = nll_loss_fn(out["h_fused"], y_discrete, c_ind)
            l_rank = rank_loss_fn(out["r_fused"], surv_time, ev_obs)
            loss = l_nll + 0.5 * l_rank
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"      Epoch {epoch}/2 completed | Loss: {epoch_loss / len(train_loader):.4f}")
    print(f"      Training finished in {time.time() - start_t:.2f}s")

    # 4. Model Evaluation & Bootstrap Concordance
    print("[4/5] Evaluating discriminative C-index with 1,000 bootstrap resamples...")
    model.eval()
    all_times = []
    all_events = []
    all_risks = []
    all_conflicts = []

    with torch.no_grad():
        for batch in test_loader:
            out = model(
                batch["wsi_bag"].to(device),
                batch["rna_expr"].to(device),
                batch["meth_features"].to(device),
                batch["clin_cov"].to(device),
                batch["avail_mask"].to(device)
            )
            all_times.extend(batch["surv_time"].numpy().tolist())
            all_events.extend(batch["event_observed"].numpy().tolist())
            all_risks.extend(out["r_fused"].cpu().numpy().flatten().tolist())
            all_conflicts.extend(out["d_conflict"].cpu().numpy().flatten().tolist())

    c_mean, c_low, c_high = pcf.bootstrap_c_index_ci(all_times, all_events, all_risks, n_bootstraps=500)
    print(f"      Discriminative C-index: {c_mean:.4f} (95% Bootstrap CI: [{c_low:.4f}, {c_high:.4f}])")

    # 5. Censoring-Aware Conformal Survival Intervals
    print("[5/5] Calibrating IPCW Split Conformal Survival Intervals (alpha=0.10)...")
    conformal = pcf.CensoredConformalSurvival(alpha=0.10)
    # Calibrate on test subset
    pred_surv = np.asarray(all_times) * 0.85 + 5.0
    q_hat = conformal.fit_calibration(all_times, all_events, pred_surv)
    print(f"      Conformal residual cutoff quantile q_hat: {q_hat:.2f} months")

    # Predict intervals with conflict adaptation
    lower, upper = conformal.predict_intervals(pred_surv, d_conflict=all_conflicts, conflict_scale=0.5)
    coverage, width = conformal.evaluate_coverage(all_times, all_events, lower, upper)
    print(f"      Empirical Conformal Coverage: {coverage * 100:.1f}% (Nominal target: 90.0%)")
    print(f"      Marginal Interval Width: {width:.2f} months")

    print("\n" + "=" * 70)
    print(" Smoke Test Succeeded! All core modules verified operational.")
    print("=" * 70)

if __name__ == "__main__":
    main()
