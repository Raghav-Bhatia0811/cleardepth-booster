"""
Sequence Loss
=============
Supervises all GRU iterations with exponentially increasing weights.

L = Σ_{i=1}^{N}  γ^(N-i) · mean(|d_i - d_gt|[valid])

Paper reference: Section III-D, Equation (16)
  γ = 0.9 (reported in training details)
"""

import torch
import torch.nn as nn
from typing import List


class SequenceLoss(nn.Module):
    """
    Weighted L1 loss across all GRU iteration predictions.

    Args:
        gamma    : Exponential decay factor. Final pred weight = 1.0,
                   earlier preds weighted by γ^(N-i). Default 0.9.
        max_disp : Maximum valid disparity value. Pixels with gt > max_disp
                   are excluded from loss computation. Default 192.
    """

    def __init__(self, gamma: float = 0.9, max_disp: float = 192.0):
        super().__init__()
        self.gamma    = gamma
        self.max_disp = max_disp

    def forward(
        self,
        disp_preds: List[torch.Tensor],
        disp_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the sequence loss.

        Args:
            disp_preds : List of N disparity predictions, each (B, 1, H, W).
                         disp_preds[0] = earliest (roughest) iteration.
                         disp_preds[-1] = final (most refined) iteration.
            disp_gt    : Ground truth disparity (B, 1, H, W).
                         Invalid pixels have value <= 0 or > max_disp.

        Returns:
            loss : Scalar tensor. Weighted mean L1 error across all iterations
                   and all valid pixels.
        """
        N = len(disp_preds)

        # Build validity mask: True where ground truth is meaningful
        # disp_gt > 0      : removes invalid/occluded pixels (marked as 0)
        # disp_gt < max_disp: removes extremely far points (near-zero disparity
        #                     leads to numerical issues in depth conversion)
        valid = (disp_gt > 0) & (disp_gt < self.max_disp)   # (B, 1, H, W)

        # If no valid pixels (e.g. during debugging with synthetic gt),
        # return zero loss to avoid NaN
        if valid.sum() == 0:
            return torch.tensor(0.0, device=disp_gt.device,
                                requires_grad=True)

        total_loss = torch.tensor(0.0, device=disp_gt.device)

        for i, pred in enumerate(disp_preds):
            # Weight: γ^(N-1-i)
            # i=0 (first/roughest): weight = γ^(N-1)  ← smallest
            # i=N-1 (final):        weight = γ^0 = 1.0 ← largest
            weight = self.gamma ** (N - 1 - i)

            # L1 error per pixel, masked to valid regions
            l1_error = (pred - disp_gt).abs()          # (B, 1, H, W)
            masked_error = l1_error[valid]              # (num_valid,)

            total_loss = total_loss + weight * masked_error.mean()

        return total_loss


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W = 1, 16, 32
    N = 4

    loss_fn = SequenceLoss(gamma=0.9, max_disp=192.0)

    # Simulate N predictions improving over iterations
    gt = torch.rand(B, 1, H, W) * 100   # Random gt in [0, 100]
    preds = [gt + torch.randn_like(gt) * (N - i) for i in range(N)]

    loss = loss_fn(preds, gt)
    assert loss.shape == torch.Size([])   # scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0

    print(f"Sequence loss (N={N}, γ=0.9): {loss.item():.4f}")

    # Verify weights decrease for earlier iterations
    single_losses = []
    for i, pred in enumerate(preds):
        w = 0.9 ** (N - 1 - i)
        err = (pred - gt).abs().mean().item()
        single_losses.append(w * err)
        print(f"  Iter {i+1}: weight={w:.4f}, error={err:.4f}, "
              f"contribution={w*err:.4f}")

    print("✅ SequenceLoss smoke test passed.")