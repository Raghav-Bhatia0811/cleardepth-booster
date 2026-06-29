"""
Stereo Data Augmentations
=========================
All transforms operate on stereo pairs (left, right, disparity) jointly
to preserve geometric consistency.

Augmentation pipeline used during training:
  1. Random crop to training resolution
  2. Random horizontal flip (swaps left/right, negates disparity)
  3. Asymmetric chromatic jitter (different params for left vs right)
  4. Normalise to [-1, 1]

Paper reference: Section IV-A (Training Details)
"""

import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
from typing import Tuple


class RandomCrop:
    """
    Randomly crop left image, right image, and disparity to target size.
    The same crop coordinates are applied to all three to preserve alignment.

    Args:
        height : Target crop height.
        width  : Target crop width.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width  = width

    def __call__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            left, right : (3, H, W) float tensors
            disp        : (1, H, W) float tensor

        Returns:
            Cropped (left, right, disp), each at target resolution.
        """
        _, H, W = left.shape

        # If image is smaller than crop size, skip cropping
        if H <= self.height or W <= self.width:
            return left, right, disp

        # Random top-left corner
        top  = random.randint(0, H - self.height)
        left_coord = random.randint(0, W - self.width)

        left  = left[:,  top:top+self.height, left_coord:left_coord+self.width]
        right = right[:, top:top+self.height, left_coord:left_coord+self.width]
        disp  = disp[:,  top:top+self.height, left_coord:left_coord+self.width]

        return left, right, disp


class RandomHorizontalFlip:
    """
    Randomly flip the stereo pair horizontally with probability p.

    When flipping:
    - Left and right images are swapped (left becomes right, right becomes left)
    - Disparity values are negated (left-to-right shift becomes right-to-left)
    - Both images are also mirrored along the width axis

    Args:
        p : Flip probability (default 0.5).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            # Flip each image along width dimension
            left_flipped  = torch.flip(left,  dims=[2])
            right_flipped = torch.flip(right, dims=[2])
            disp_flipped  = torch.flip(disp,  dims=[2])

            # Swap left/right and negate disparity
            left  = right_flipped
            right = left_flipped
            disp  = -disp_flipped

        return left, right, disp


class AsymmetricColorJitter:
    """
    Apply chromatic jitter with DIFFERENT random parameters to left and right.

    This simulates real stereo cameras where the two sensors have slightly
    different colour responses and exposure settings.

    With probability p_asymmetric, left and right get different jitter params.
    With probability (1 - p_asymmetric), both get the same params (symmetric).

    Args:
        brightness, contrast, saturation, hue : Jitter ranges (same as
            torchvision.transforms.ColorJitter).
        p_asymmetric : Probability of applying asymmetric jitter.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        p_asymmetric: float = 0.5,
    ):
        self.brightness    = brightness
        self.contrast      = contrast
        self.saturation    = saturation
        self.hue           = hue
        self.p_asymmetric  = p_asymmetric

    def _sample_params(self):
        """Sample random jitter parameters."""
        b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0, 1 - self.contrast),   1 + self.contrast)
        s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)
        return b, c, s, h

    def _apply(self, img: torch.Tensor, params) -> torch.Tensor:
        """Apply jitter with given parameters to a (3, H, W) tensor."""
        b, c, s, h = params
        # torchvision functional expects values in [0,1]
        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        img = TF.adjust_saturation(img, s)
        img = TF.adjust_hue(img, h)
        return img.clamp(0.0, 1.0)

    def __call__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params_left = self._sample_params()
        if random.random() < self.p_asymmetric:
            params_right = self._sample_params()   # different params
        else:
            params_right = params_left              # same params

        left  = self._apply(left,  params_left)
        right = self._apply(right, params_right)
        # Disparity is geometric — not affected by colour jitter
        return left, right, disp


class Normalise:
    """
    Convert images from [0, 1] float to [-1, 1] float.
    Disparity is left unchanged (already in pixel units).
    """

    def __call__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left  = left  * 2.0 - 1.0
        right = right * 2.0 - 1.0
        return left, right, disp


class StereoTransform:
    """
    Composes the full augmentation pipeline for training.

    Usage:
        transform = StereoTransform(height=360, width=720, augment=True)
        left, right, disp = transform(left, right, disp)

    Args:
        height, width : Target crop resolution.
        augment       : If True, apply random augmentations (training).
                        If False, only normalise (validation/test).
    """

    def __init__(self, height: int, width: int, augment: bool = True):
        self.augment = augment
        self.crop    = RandomCrop(height, width)
        self.flip    = RandomHorizontalFlip(p=0.5)
        self.jitter  = AsymmetricColorJitter()
        self.norm    = Normalise()

    def __call__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.augment:
            left, right, disp = self.crop(left, right, disp)
            left, right, disp = self.flip(left, right, disp)
            left, right, disp = self.jitter(left, right, disp)
        left, right, disp = self.norm(left, right, disp)
        return left, right, disp