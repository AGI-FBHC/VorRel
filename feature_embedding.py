import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbedding(nn.Module):
    """Side-injection feature embedding.

    Architecture:
      Main path:  ESM-2 (1280d) ──→ hidden_dim ──→ hidden_dim   (big capacity)
      Side path:  PSSM+DSSP+Atomic (40d) ──→ 64d               (small bypass)
      Output:     main + gate * side_proj(bypass)                (gated injection)

    This preserves ESM's full representational power while letting
    small evolutionary/structural/physicochemical features inject
    complementary signal through a controlled side channel.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.esm_encoder = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.side_encoder = nn.Sequential(
            nn.LayerNorm(40),
            nn.Linear(40, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )

        self.side_proj = nn.Linear(64, hidden_dim)

        self.side_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features):
        """features: (B, N, 1320)"""
        f_esm = features[..., :1280]
        f_side = torch.cat([
            features[..., 1280:1300],
            features[..., 1300:1313],
            features[..., 1313:1320],
        ], dim=-1)

        h_main = self.esm_encoder(f_esm)

        h_side = self.side_encoder(f_side)
        h_side = self.side_proj(h_side)

        gate = torch.sigmoid(self.side_gate)

        return h_main + gate * h_side
