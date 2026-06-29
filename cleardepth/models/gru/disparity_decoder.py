"""
Disparity Decoder
=================
Converts a GRU hidden state into a disparity update Δd.

Architecture: two convolutional layers with ReLU in between.
The output is a signed delta — positive or negative correction
to the current disparity estimate.

Paper reference: Section III-C
  "Decoder consist of two convolutional layers"
  "dk+1 = dk + Δdk"  (Equation 15)
"""

import torch
import torch.nn as nn


class DisparityDecoder(nn.Module):
    """
    Two-layer conv decoder: hidden state → disparity update.

    Args:
        hidden_dim   : GRU hidden state channel dimension.
        mid_channels : Intermediate channel count (default 256).
    """

    def __init__(self, hidden_dim: int = 128, mid_channels: int = 256):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_dim, mid_channels,
                               kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, 1,
                               kernel_size=3, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : GRU hidden state (B, hidden_dim, H, W).

        Returns:
            delta_d : Disparity update (B, 1, H, W).
        """
        x = self.relu(self.conv1(h))
        delta_d = self.conv2(x)
        return delta_d


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W = 1, 16, 32
    decoder = DisparityDecoder(hidden_dim=128)
    h = torch.randn(B, 128, H, W)
    delta = decoder(h)
    assert delta.shape == (B, 1, H, W)
    print(f"Hidden state: {list(h.shape)} → Δd: {list(delta.shape)}")
    print("✅ DisparityDecoder smoke test passed.")