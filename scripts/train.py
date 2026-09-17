#!/usr/bin/env python3
"""
Training script for PathConFuse and comparative baselines.
Supports synthetic cohorts for rapid verification and real TCGA cohorts.
"""

import os
import sys
import time
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pathconfuse as pcf


def parse_args():
    parser = argparse.ArgumentParser(description="Train PathConFuse or comparative baseline models.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--model", type=str, default="pathconfuse",
                        choices=["pathconfuse", "mcat", "flat", "histology", "genomic", "clinical"],
                        help="Model architecture to train.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic benchmark cohort if real data is not specified.")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[*] Training model '{args.model}' on {device}")

    # Load Data
    print("[*] Loading dataset...")
    if args.synthetic:
        train_data = pcf.generate_synthetic_cohort(n_patients=200, seed=args.seed)
        val_data = pcf.generate_synthetic_cohort(n_patients=60, seed=args.seed + 1)
    else:
        raise NotImplementedError("Specify real data paths in configs/default_brca.yaml")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Instantiate Model
    if args.model == "pathconfuse":
        mask = pcf.load_or_generate_pathway_mask(n_pathways=331, n_genes=4999, seed=args.seed)
        model = pcf.PathConFuse(norm_mask=mask).to(device)
    elif args.model == "mcat":
        model = pcf.MCATSurv().to(device)
    elif args.model == "flat":
        model = pcf.FlatConcatSurv().to(device)
    elif args.model == "histology":
        model = pcf.HistologyOnlySurv().to(device)
    elif args.model == "genomic":
        model = pcf.GenomicOnlySurv().to(device)
    elif args.model == "clinical":
        model = pcf.ClinicalOnlySurv().to(device)

    nll_loss_fn = pcf.NLLSurvLoss()
    rank_loss_fn = pcf.PairwiseSurvLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training Loop
    print(f"[*] Starting {args.epochs} training epochs...")
    best_c_index = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
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

            if args.model == "pathconfuse":
                out = model(wsi_bag, rna_expr, meth_feat, clin_cov, avail_mask)
            elif args.model == "mcat":
                out = model(wsi_bag, rna_expr, meth_feat, clin_cov, avail_mask)
            elif args.model == "flat":
                out = model(wsi_bag, rna_expr, meth_feat, clin_cov, avail_mask)
            elif args.model == "histology":
                out = model(wsi_bag)
            elif args.model == "genomic":
                out = model(rna_expr)
            elif args.model == "clinical":
                out = model(clin_cov)

            l_nll = nll_loss_fn(out["h_fused"], y_discrete, c_ind)
            l_rank = rank_loss_fn(out["r_fused"], surv_time, ev_obs)
            loss = l_nll + 0.5 * l_rank
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation C-Index
        model.eval()
        val_times, val_events, val_risks = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                wsi_bag = batch["wsi_bag"].to(device)
                rna_expr = batch["rna_expr"].to(device)
                meth_feat = batch["meth_features"].to(device)
                clin_cov = batch["clin_cov"].to(device)
                avail_mask = batch["avail_mask"].to(device)

                if args.model in ["pathconfuse", "mcat", "flat"]:
                    out = model(wsi_bag, rna_expr, meth_feat, clin_cov, avail_mask)
                elif args.model == "histology":
                    out = model(wsi_bag)
                elif args.model == "genomic":
                    out = model(rna_expr)
                elif args.model == "clinical":
                    out = model(clin_cov)

                val_times.extend(batch["surv_time"].numpy().tolist())
                val_events.extend(batch["event_observed"].numpy().tolist())
                val_risks.extend(out["r_fused"].cpu().numpy().flatten().tolist())

        c_val = pcf.compute_c_index(val_times, val_events, val_risks)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {total_loss / len(train_loader):.4f} | Val C-index: {c_val:.4f}")

        if c_val > best_c_index:
            best_c_index = c_val
            ckpt_path = os.path.join(args.output_dir, f"{args.model}_best.pt")
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "c_index": c_val}, ckpt_path)

    print(f"[*] Training complete. Best Val C-index: {best_c_index:.4f}")


if __name__ == "__main__":
    main()
