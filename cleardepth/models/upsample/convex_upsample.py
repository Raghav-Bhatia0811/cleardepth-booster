"""
Convex Upsampling
=================
Learned 4× upsampling from GRU's 1/4-resolution disparity to full resolution.

At inference time the GRU outputs disparity at 1/4 of the input image. A naive
bilinear upsample blurs fine structural edges — exactly the signal ClearDepth
needs to be accurate on. Convex upsampling (from RAFT / RAFT-Stereo) avoids
this by predicting a per-pixel blend mask from the network's hidden state.

How it works
------------
For every output pixel (full resolution) we predict 9 weights — one for each
cell in the 3×3 neighbourhood of the corresponding coarse pixel. The output
pixel is a softmax-weighted sum of those 9 disparity values, scaled by the
upsampling factor so that the disparity is in full-resolution pixel units.

Prediction:
  mask = MaskNet(hidden)   # (B, 9 × scale², H/4, W/4)

Upsampling:
  unfold  coarse disp into 9-neighbour patches → (B, 9, H/4 × W/4)
  reshape mask →            (B, 9, scale², H/4, W/4)
  softmax over dim 1  (the 9-neighbour dim)
  weighted sum        → (B, 1, scale², H/4, W/4)
  pixel-shuffle       → (B, 1, H, W)

For scale=4: each coarse pixel produces a 4×4 block of 16 output pixels,
each blended from 9 coarse neighbours — 9 × 16 = 144 mask channels.

Paper connection: Section III-C / "coarse to fine gradual optimisation".
RAFT-Stereo (which ClearDepth extends) describes this module explicitly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvexUpsample(nn.Module):
    """
    Learned convex combination upsampling.

    Args:
        hidden_dim : Channel dimension of the input hidden state (= GRU hidden_dim).
        scale      : Integer upsampling factor. Default 4 (1/4 → full resolution).
    """

    def __init__(self, hidden_dim: int = 128, scale: int = 4):
        super().__init__()
        self.scale = scale

        # MaskNet: hidden state → per-pixel blend weights.
        # Output has 9 × scale² channels (9 neighbours × scale² sub-pixels).
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 9 * scale * scale, kernel_size=1, padding=0),
        )

    def forward(
        self,
        disp: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            disp   : Coarse disparity  (B, 1, H, W)  — at 1/4 resolution.
            hidden : GRU hidden state  (B, hidden_dim, H, W)  — same spatial size.
                     (The finest GRU hidden state, upsampled to match disp if needed.)

        Returns:
            Full-resolution disparity (B, 1, H*scale, W*scale).
        """
        scale = self.scale
        B, _, H, W = disp.shape

        # ── Predict blend mask ─────────────────────────────────────────────
        mask = self.mask_predictor(hidden)             # (B, 9*scale², H, W)
        mask = mask.view(B, 9, scale * scale, H, W)   # (B, 9, s², H, W)
        mask = F.softmax(mask, dim=1)                  # softmax over 9 neighbours

        # ── Gather 3×3 neighbourhood for every coarse pixel ───────────────
        # Scale disparity to full-res units: a disparity of d at 1/4 scale
        # corresponds to scale*d pixels at full resolution.
        disp_scaled = scale * disp                     # (B, 1, H, W)

        # unfold: extract 3×3 patches (flattened to 9) around every pixel
        # → (B, 9, H*W)
        neighbours = F.unfold(disp_scaled, kernel_size=3, padding=1)
        neighbours = neighbours.view(B, 1, 9, 1, H, W)  # (B, 1, 9, 1, H, W)

        # ── Weighted combination ───────────────────────────────────────────
        # mask:       (B, 9, s², H, W)
        # neighbours: (B, 1, 9, 1, H, W)
        mask = mask.unsqueeze(1)                       # (B, 1, 9, s², H, W)
        up_disp = (mask * neighbours).sum(dim=2)      # (B, 1, s², H, W)

        # ── Pixel-shuffle to full resolution ──────────────────────────────
        # Reshape s² → (scale, scale), then interleave with (H, W):
        #   (B, 1, s², H, W) → (B, 1, scale, scale, H, W)
        #                     → permute → (B, 1, H, scale, W, scale)
        #                     → reshape → (B, 1, H*scale, W*scale)
        up_disp = up_disp.view(B, 1, scale, scale, H, W)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)   # (B, 1, H, scale, W, scale)
        up_disp = up_disp.reshape(B, 1, H * scale, W * scale)

        return up_disp


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W, HIDDEN, SCALE = 1, 16, 32, 128, 4

    module = ConvexUpsample(hidden_dim=HIDDEN, scale=SCALE)
    disp   = torch.randn(B, 1, H, W)
    hidden = torch.randn(B, HIDDEN, H, W)

    up = module(disp, hidden)
    expected = (B, 1, H * SCALE, W * SCALE)
    assert up.shape == torch.Size(expected), \
        f"Expected {expected}, got {tuple(up.shape)}"

    params = sum(p.numel() for p in module.parameters())
    print(f"Input disp:  {list(disp.shape)}")
    print(f"Input hidden:{list(hidden.shape)}")
    print(f"Output:      {list(up.shape)}")
    print(f"Parameters:  {params:,}")
    print("ConvexUpsample smoke test passed.")
