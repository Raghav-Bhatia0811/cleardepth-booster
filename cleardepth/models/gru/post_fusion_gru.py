"""
Post-Fusion GRU
===============
Modified multi-scale GRU that injects structural context features as
additive biases into gate pre-activations.

Core equations (Section III-C):
    x_k  = [C_k, d_k, c_k, c_r, c_h]           (Eq. 7)
    z_k  = σ(Conv([h_{k-1}, x_k], W_z) + c_k)   (Eq. 8)
    r_k  = σ(Conv([h_{k-1}, x_k], W_r) + c_r)   (Eq. 9)
    h̃_k  = tanh(Conv([r_k⊙h_{k-1}, x_k], W_h) + c_h)  (Eq. 10)
    h_k  = (1 - z_k) ⊙ h_{k-1} + z_k ⊙ h̃_k    (Eq. 11)

Multi-scale decoding (Eqs. 12-14):
    Δd_{1/32} = Decoder(h_{1/32})
    Δd_{1/16} = Decoder(h_{1/16} + Interp(Δd_{1/32}))
    Δd_{1/8}  = Decoder(h_{1/8}  + Interp(Δd_{1/16}))

The structural biases c_k, c_r, c_h are computed ONCE by the context
encoder and passed into every GRU iteration. They are downsampled to
match each GRU scale's spatial resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .disparity_decoder import DisparityDecoder


class PostFusionGRUCell(nn.Module):
    """
    Single-scale Post-Fusion GRU cell.

    This is one GRU operating at one spatial resolution.
    The full multi-scale GRU stacks three of these.

    Args:
        input_dim  : Channels in the concatenated input x_k.
                     = corr_channels + 1 (disp) + 3 * hidden_dim (c_k,c_r,c_h)
        hidden_dim : Hidden state channel dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Each gate has its own conv: operates on [h_prev, x_k] concatenated
        # Input to each conv = hidden_dim + input_dim channels
        combined_dim = hidden_dim + input_dim

        # Update gate W_z
        self.conv_z = nn.Conv2d(combined_dim, hidden_dim,
                                kernel_size=3, padding=1)
        # Reset gate W_r
        self.conv_r = nn.Conv2d(combined_dim, hidden_dim,
                                kernel_size=3, padding=1)
        # Candidate gate W_h
        # Input: [r⊙h_prev, x_k] — same combined_dim
        self.conv_h = nn.Conv2d(combined_dim, hidden_dim,
                                kernel_size=3, padding=1)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        c_k: torch.Tensor,
        c_r: torch.Tensor,
        c_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        One GRU step with Post-Fusion structural biases.

        Args:
            h   : Previous hidden state  (B, hidden_dim, H, W).
            x   : Input features         (B, input_dim,  H, W).
                  = [corr, disparity, c_k, c_r, c_h] concatenated.
            c_k : Structural bias for update gate  (B, hidden_dim, H, W).
            c_r : Structural bias for reset gate   (B, hidden_dim, H, W).
            c_h : Structural bias for candidate    (B, hidden_dim, H, W).

        Returns:
            h_new : Updated hidden state (B, hidden_dim, H, W).
        """
        # Concatenate previous hidden state with input
        hx = torch.cat([h, x], dim=1)   # (B, hidden_dim + input_dim, H, W)

        # ── Update gate (Eq. 8) ────────────────────────────────────────────
        # c_k shifts the pre-activation — structural features bias how much
        # the hidden state should be updated at each position
        z = torch.sigmoid(self.conv_z(hx) + c_k)

        # ── Reset gate (Eq. 9) ─────────────────────────────────────────────
        # c_r biases how much of the previous hidden state to forget
        r = torch.sigmoid(self.conv_r(hx) + c_r)

        # ── Candidate hidden state (Eq. 10) ────────────────────────────────
        # r gates how much of h_prev survives into the candidate computation
        rh_x = torch.cat([r * h, x], dim=1)   # (B, combined_dim, H, W)
        h_tilde = torch.tanh(self.conv_h(rh_x) + c_h)

        # ── New hidden state (Eq. 11) ──────────────────────────────────────
        # Interpolate between old memory and new candidate
        h_new = (1.0 - z) * h + z * h_tilde

        return h_new


class PostFusionGRU(nn.Module):
    """
    Multi-scale Post-Fusion GRU with coarse-to-fine disparity decoding.

    Runs three GRU cells at three spatial scales (1/8, 1/16, 1/32 of
    original image resolution, which correspond to 1/2, 1/4, 1/8 of
    the 1/4-scale feature maps from the encoders).

    At each iteration:
      1. Downsample correlation and context features to each GRU scale
      2. Run each GRU cell to update its hidden state
      3. Decode hidden states coarse-to-fine to get Δd
      4. Accumulate: d_{k+1} = d_k + Δd

    Args:
        corr_channels : Channels from correlation pyramid (default 36).
        hidden_dim    : GRU hidden state dimension (default 128).
        n_gru_layers  : Number of GRU scales (default 3).
    """

    def __init__(
        self,
        corr_channels: int = 36,
        hidden_dim: int = 128,
        n_gru_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.n_gru_layers = n_gru_layers

        # Input to GRU cell x_k = [corr, disp, c_k, c_r, c_h]
        # corr_channels + 1 (disparity) + 3 * hidden_dim (structural biases)
        input_dim = corr_channels + 1 + 3 * hidden_dim

        # One GRU cell per scale
        # All scales share the same input_dim and hidden_dim
        self.gru_cells = nn.ModuleList([
            PostFusionGRUCell(input_dim=input_dim, hidden_dim=hidden_dim)
            for _ in range(n_gru_layers)
        ])

        # One decoder per scale
        self.decoders = nn.ModuleList([
            DisparityDecoder(hidden_dim=hidden_dim)
            for _ in range(n_gru_layers)
        ])

        # Learned 1→hidden_dim projection for coarse-to-fine delta injection.
        # Paper Eqs 13-14: Decoder(h_{finer} + Interp(Δd_{coarser}))
        # The coarse Δd has 1 channel; h has hidden_dim channels. A 1×1 conv
        # projects Δd into the hidden state space before addition.
        # One projection per non-coarsest scale (indices 0 .. n_gru_layers-2).
        self.delta_proj = nn.ModuleList([
            nn.Conv2d(1, hidden_dim, kernel_size=1, bias=True)
            for _ in range(n_gru_layers - 1)
        ])

    def _downsample_to_scale(
        self,
        feat: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Bilinear downsample feat to (target_h, target_w)."""
        if feat.shape[-2] == target_h and feat.shape[-1] == target_w:
            return feat
        return F.interpolate(feat, size=(target_h, target_w),
                             mode='bilinear', align_corners=False)

    def forward(
        self,
        feat_left: torch.Tensor,
        feat_right: torch.Tensor,
        c_k: torch.Tensor,
        c_r: torch.Tensor,
        c_h: torch.Tensor,
        corr_fn,
        n_iters: int = 12,
    ):
        """
        Run n_iters of multi-scale GRU disparity refinement.

        Args:
            feat_left  : Left  feature map (B, C, H, W) at 1/4 scale.
            feat_right : Right feature map (B, C, H, W) at 1/4 scale.
            c_k        : Structural bias for update gate (B, hidden_dim, H, W).
            c_r        : Structural bias for reset gate  (B, hidden_dim, H, W).
            c_h        : Structural bias for candidate   (B, hidden_dim, H, W).
            corr_fn    : Callable(feat_left, feat_right, disparity) → corr.
                         This is the CorrelationPyramid.forward method.
            n_iters    : Number of refinement iterations.

        Returns:
            disp_preds       : List of disparity predictions, one per iteration.
                               Each is (B, 1, H, W) at the 1/4-scale feature resolution.
                               Used by the sequence loss (all iterations matter).
        """
        B, C, H, W = feat_left.shape

        # Spatial resolutions for each GRU scale.
        # Scale 0 (finest, 1/2 of feat): H/2 × W/2
        # Scale 1 (middle,  1/4 of feat): H/4 × W/4
        # Scale 2 (coarsest, 1/8 of feat): H/8 × W/8
        # Note: we index from finest (0) to coarsest (n-1)
        scale_shapes = [
            (max(H // (2 ** i), 1), max(W // (2 ** i), 1))
            for i in range(self.n_gru_layers)
        ]

        # Initialise hidden states to zeros at each scale
        hidden_states = [
            torch.zeros(B, self.hidden_dim, sh, sw,
                        device=feat_left.device, dtype=feat_left.dtype)
            for sh, sw in scale_shapes
        ]

        # Initial disparity estimate: all zeros at finest GRU scale
        disparity = torch.zeros(B, 1, scale_shapes[0][0], scale_shapes[0][1],
                                device=feat_left.device, dtype=feat_left.dtype)

        disp_preds = []

        for _ in range(n_iters):
            disparity = disparity.detach()   # stop gradient through iterations

            # ── Coarse-to-fine GRU update ──────────────────────────────────
            delta_d_prev = None   # carries the coarse delta upward

            # Iterate from coarsest to finest (reverse order)
            for scale_i in range(self.n_gru_layers - 1, -1, -1):
                sh, sw = scale_shapes[scale_i]

                # Downsample disparity, features, context to this scale
                disp_s = self._downsample_to_scale(disparity, sh, sw)
                fl_s   = self._downsample_to_scale(feat_left,  sh, sw)
                fr_s   = self._downsample_to_scale(feat_right, sh, sw)
                ck_s   = self._downsample_to_scale(c_k, sh, sw)
                cr_s   = self._downsample_to_scale(c_r, sh, sw)
                ch_s   = self._downsample_to_scale(c_h, sh, sw)

                # Compute correlation at this scale
                corr_s = corr_fn(fl_s, fr_s, disp_s)   # (B, corr_ch, sh, sw)

                # Build input x_k = [corr, disp, c_k, c_r, c_h]
                x_k = torch.cat([corr_s, disp_s, ck_s, cr_s, ch_s], dim=1)

                # Add upsampled coarse delta (Eqs. 13, 14)
                if delta_d_prev is not None:
                    delta_d_upsampled = F.interpolate(
                        delta_d_prev, size=(sh, sw),
                        mode='bilinear', align_corners=False
                    )
                    # Add to hidden state as in the paper equations
                    h_input = hidden_states[scale_i]
                    # Pad delta to hidden_dim channels by repeating or projecting
                    # Paper eq: h_{1/16} + Interp(Δd_{1/32})
                    # We interpret this as adding delta into the decoder input
                    # rather than directly to hidden state (avoids dimension mismatch)
                    # Store for use in decoder step below
                else:
                    delta_d_upsampled = None

                # Update hidden state
                hidden_states[scale_i] = self.gru_cells[scale_i](
                    hidden_states[scale_i], x_k, ck_s, cr_s, ch_s
                )

                # Decode: hidden state → disparity delta (paper Eqs 12-14).
                # For non-coarsest scales, inject the upsampled coarse delta
                # into the hidden state BEFORE decoding, as the paper specifies:
                #   Decoder(h_{finer} + Proj(Interp(Δd_{coarser})))
                h_for_decode = hidden_states[scale_i]
                if delta_d_upsampled is not None:
                    # scale_i < n_gru_layers-1 here, so delta_proj[scale_i] exists
                    projected = self.delta_proj[scale_i](delta_d_upsampled)
                    h_for_decode = h_for_decode + projected

                delta_d = self.decoders[scale_i](h_for_decode)
                delta_d_prev = delta_d

            # ── Update disparity ───────────────────────────────────────────
            # delta_d_prev now holds the finest-scale delta (from scale_i=0)
            # Upsample to full feature resolution if needed
            delta_d_full = F.interpolate(
                delta_d_prev, size=(H, W),
                mode='bilinear', align_corners=False
            )
            disparity = disparity + delta_d_full
            disp_preds.append(disparity)

        return disp_preds


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from cleardepth.models.correlation.correlation_pyramid import CorrelationPyramid

    B, C, H, W = 1, 64, 16, 32
    HIDDEN = 128
    N_ITERS = 4

    corr_fn = CorrelationPyramid(num_levels=4, radius=4)
    gru = PostFusionGRU(corr_channels=corr_fn.out_channels,
                        hidden_dim=HIDDEN, n_gru_layers=3)

    feat_l = torch.randn(B, C, H, W)
    feat_r = torch.randn(B, C, H, W)
    c_k = torch.randn(B, HIDDEN, H, W)
    c_r = torch.randn(B, HIDDEN, H, W)
    c_h = torch.randn(B, HIDDEN, H, W)

    preds = gru(feat_l, feat_r, c_k, c_r, c_h, corr_fn, n_iters=N_ITERS)

    assert len(preds) == N_ITERS
    for i, d in enumerate(preds):
        assert d.shape == (B, 1, H, W), f"Iter {i}: {tuple(d.shape)}"

    print(f"GRU output: {N_ITERS} predictions, each {list(preds[0].shape)}")
    print("PostFusionGRU smoke test passed.")