"""
Comparative Baseline Architectures for Multimodal Survival Prognosis:
- MCATSurv: Multimodal Co-Attention Transformer (Chen et al., ICCV 2021).
- FlatConcatSurv: Naive early concatenation MLP.
- HistologyOnlySurv: Unimodal WSI ABMIL model.
- GenomicOnlySurv: Unimodal high-dimensional RNA MLP.
- ClinicalOnlySurv: Shallow clinical staging baseline.
- LateFusionSurv: Post-hoc hazard logit averaging.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pathconfuse import PathwayTokenEncoder, WSIBagEncoder


class MCATSurv(nn.Module):
    """
    Multimodal Co-Attention Transformer for survival prediction (Chen et al., ICCV 2021).
    Omics tokens cross-attend to histology patch embeddings.
    """
    def __init__(self, d_wsi=768, n_genes=4999, d_model=128, n_classes=4):
        super().__init__()
        self.wsi_proj = nn.Linear(d_wsi, d_model)
        self.omics_proj = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.GELU(),
            nn.Linear(512, d_model * 4)
        )
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.clin_fc = nn.Linear(4, 64)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 5 + 64, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes)
        )

    def forward(self, wsi_bag, rna_expr, meth_features=None, clin_cov=None, avail_mask=None):
        B = rna_expr.shape[0]
        omics_tokens = self.omics_proj(rna_expr).view(B, 4, self.d_model)
        wsi_feats = self.wsi_proj(wsi_bag)

        Q = self.q_proj(omics_tokens)
        K = self.k_proj(wsi_feats)
        V = self.v_proj(wsi_feats)

        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.d_model)
        attn = F.softmax(scores, dim=-1)
        attended_wsi = torch.bmm(attn, V)

        omics_pool = omics_tokens.mean(dim=1)
        attended_pool = attended_wsi.view(B, -1)

        clin_feat = F.gelu(self.clin_fc(clin_cov))
        fused = torch.cat([omics_pool, attended_pool, clin_feat], dim=-1)
        h = self.classifier(fused)
        r = PathwayTokenEncoder.hazard_to_risk(h)
        return {"h_fused": h, "r_fused": r}


class FlatConcatSurv(nn.Module):
    """
    Early Concatenation MLP baseline:
    Concatenates pooled WSI embeddings, raw transcriptomics, methylation, and clinical features.
    """
    def __init__(self, d_wsi=768, n_genes=4999, d_clin=4, hidden_dim=256, n_classes=4):
        super().__init__()
        self.wsi_pool = nn.AdaptiveAvgPool1d(1)
        total_in = d_wsi + n_genes + 256 + d_clin
        self.mlp = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, wsi_bag, rna_expr, meth_features, clin_cov, avail_mask):
        wsi_feat = wsi_bag.mean(dim=1)  # [B, d_wsi]
        feat_all = torch.cat([wsi_feat, rna_expr, meth_features, clin_cov], dim=-1)
        h = self.mlp(feat_all)
        r = PathwayTokenEncoder.hazard_to_risk(h)
        return {"h_fused": h, "r_fused": r}


class HistologyOnlySurv(nn.Module):
    """Unimodal Histology ABMIL Survival Model."""
    def __init__(self, d_in=768, hidden_dim=256):
        super().__init__()
        self.enc = WSIBagEncoder(input_dim=d_in, hidden_dim=hidden_dim)

    def forward(self, wsi_bag):
        z_I, h_I, r_I, attn = self.enc(wsi_bag)
        return {"h_fused": h_I, "r_fused": r_I, "wsi_attn": attn}


class GenomicOnlySurv(nn.Module):
    """Unimodal Transcriptomics MLP Survival Model."""
    def __init__(self, n_genes=4999, hidden_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, rna_expr):
        h = self.fc(rna_expr)
        r = PathwayTokenEncoder.hazard_to_risk(h)
        return {"h_fused": h, "r_fused": r}


class ClinicalOnlySurv(nn.Module):
    """Unimodal Clinical Covariates Survival Model."""
    def __init__(self, d_clin=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_clin, 64),
            nn.GELU(),
            nn.Linear(64, 4)
        )

    def forward(self, clin_cov):
        h = self.fc(clin_cov)
        r = PathwayTokenEncoder.hazard_to_risk(h)
        return {"h_fused": h, "r_fused": r}


class LateFusionSurv(nn.Module):
    """Late Fusion combining unimodal hazard logits."""
    def __init__(self, d_wsi=768, n_genes=4999, hidden_dim=256):
        super().__init__()
        self.wsi = WSIBagEncoder(input_dim=d_wsi, hidden_dim=hidden_dim)
        self.rna = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )
        self.fusion = nn.Linear(8, 4)

    def forward(self, wsi_bag, rna_expr):
        _, h_w, _, _ = self.wsi(wsi_bag)
        h_r = self.rna(rna_expr)
        h = self.fusion(torch.cat([h_w, h_r], dim=-1))
        r = PathwayTokenEncoder.hazard_to_risk(h)
        return {"h_fused": h, "r_fused": r}
