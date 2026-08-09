import math

import torch
import torch.nn as nn

class GeosteeringCorrelationModel(nn.Module):
    """
    Two-branch registration network with an EXPLICIT matching layer.

    Horizontal branch: CNN + BiLSTM over the (GR, Z_rel, dZ) window —
      encodes WHAT the bit has been drilling through lately.
    Typewell branch: CNN over the (GR, position) excerpt — but in contrast
      to a plain fusion design, its feature map is NOT only globally pooled:
      the positions are kept alive for matching.
    Matching layer: the horizontal representation is projected to a query
      vector and compared (scaled dot product) against the typewell feature
      map at EVERY position, yielding a similarity profile along the
      excerpt. A softmax over the profile gives an attention distribution
      whose expected position is a differentiable soft-argmax — i.e., a
      direct cross-correlation estimate of the registration offset.
    Fusion head: combines global features of both branches, the raw
      similarity profile, and the soft-argmax offset into the final
      correction (normalized to [-1, 1] by jitter_ft).
    """

    def __init__(self, horiz_features=3, type_features=2, type_len=128):
        super(GeosteeringCorrelationModel, self).__init__()

        # --------------------------------------------
        # 1a. Horizontal branch CNN (Batch, 3, W)
        # --------------------------------------------
        self.horiz_cnn = nn.Sequential(
            nn.Conv1d(horiz_features, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # 1b. Horizontal branch BiLSTM (sequence context along MD)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        # mean+max pooling over time: 2*256 = 512 features

        # --------------------------------------------
        # 2. Typewell branch CNN (Batch, 2, type_len)
        # --------------------------------------------
        self.type_cnn = nn.Sequential(
            nn.Conv1d(type_features, 32, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # feature map length after two MaxPools:
        self.t_out_len = type_len // 4          # e.g. 128 -> 32
        # mean+max pooling over length: 2*64 = 128 global features

        # --------------------------------------------
        # 3. Matching layer (cross-correlation / attention)
        # --------------------------------------------
        self.match_dim = 64
        self.query_proj = nn.Linear(512, self.match_dim)
        # grid positions of the downsampled typewell features in [-1, 1]
        self.register_buffer(
            'match_pos', torch.linspace(-1.0, 1.0, self.t_out_len))

        # --------------------------------------------
        # 4. Fusion head:
        #    512 (horizontal) + 128 (typewell global)
        #    + t_out_len (similarity profile) + 1 (soft-argmax offset)
        # --------------------------------------------
        fusion_in = 512 + 128 + self.t_out_len + 1
        self.fc = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),  # normalized registration correction
        )

    def forward(self, x_h, x_t):
        """
        x_h: (Batch, 3, window_size)   — horizontal window
        x_t: (Batch, 2, type_len)      — typewell excerpt around the estimate
        returns: (Batch,) normalized correction in [-1, 1]-ish
        """
        # Horizontal branch
        h = self.horiz_cnn(x_h)          # (B, 64, W/4)
        h = h.permute(0, 2, 1)           # (B, W/4, 64)
        lstm_out, _ = self.lstm(h)       # (B, W/4, 256)
        h_avg = torch.mean(lstm_out, dim=1)
        h_max, _ = torch.max(lstm_out, dim=1)
        h_feat = torch.cat([h_avg, h_max], dim=1)     # (B, 512)

        # Typewell branch (positions kept alive)
        t_map = self.type_cnn(x_t)       # (B, 64, L')  with L' = type_len/4
        t_avg = torch.mean(t_map, dim=2)
        t_max, _ = torch.max(t_map, dim=2)
        t_feat = torch.cat([t_avg, t_max], dim=1)     # (B, 128)

        # Matching: similarity of the horizontal query against every
        # typewell position (scaled dot-product cross-correlation)
        q = self.query_proj(h_feat)                   # (B, 64)
        sim = torch.einsum('bc,bcl->bl', q, t_map)    # (B, L')
        sim = sim / math.sqrt(self.match_dim)

        # Soft-argmax: expected position of the best match in [-1, 1]
        attn = torch.softmax(sim, dim=1)              # (B, L')
        soft_offset = (attn * self.match_pos).sum(dim=1, keepdim=True)  # (B,1)

        # Fusion
        fused = torch.cat([h_feat, t_feat, sim, soft_offset], dim=1)
        correction = self.fc(fused)                   # (B, 1)
        return correction.squeeze(-1)


if __name__ == "__main__":
    model = GeosteeringCorrelationModel(type_len=128)
    x_h = torch.randn(32, 3, 100)
    x_t = torch.randn(32, 2, 128)
    out = model(x_h, x_t)
    print(f"x_h: {x_h.shape}, x_t: {x_t.shape} -> out: {out.shape}")
    print(f"First predictions: {out[:5]}")