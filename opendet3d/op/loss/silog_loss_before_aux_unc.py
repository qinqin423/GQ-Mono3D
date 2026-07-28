"""SILog loss for depth estimation."""

from __future__ import annotations

import torch
from torch import Tensor
from vis4d.common.typing import ArgsType
from vis4d.op.loss.base import Loss

from .util import masked_mean_var


class SILogLoss(Loss):
    """SILogLoss with optional probabilistic depth uncertainty."""

    def __init__(
        self,
        *args: ArgsType,
        scale_pred_weight: float = 0.15,
        eps: float = 1e-5,
        min_depth: float = 0.0,
        prob_lambda: float = 0.01,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.scale_pred_weight = scale_pred_weight
        self.eps = eps
        self.min_depth = min_depth
        self.prob_lambda = prob_lambda

    def forward(
        self,
        depths: Tensor | tuple[Tensor, Tensor],
        target_depths: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            depths: Predicted depth or (depth_mu, depth_logvar).
            target_depths: Target depth. Shape: (B, H, W).
            mask: Valid depth mask. Shape: (B, H, W).
        """
        if mask is None:
            mask = target_depths > self.min_depth
        else:
            mask = mask.to(torch.bool)
            mask = torch.logical_and(mask, target_depths > self.min_depth)

        if isinstance(depths, Tensor):
            depth_mu = depths
            depth_logvar = None
        else:
            depth_mu, depth_logvar = depths
            depth_logvar = depth_logvar.clamp(min=-3.0, max=2.0)

        log_depths = torch.log(depth_mu.clamp(min=self.eps))
        log_target_depths = torch.log(target_depths.clamp(min=self.eps))
        log_error = log_depths - log_target_depths

        if depth_logvar is None:
            mean_error, var_error = masked_mean_var(log_error, mask=mask)
            scale_error = mean_error**2
            loss = var_error + self.scale_pred_weight * scale_error
            out_loss = torch.sqrt(loss.clamp(min=self.eps))
            return out_loss.mean()

        # ===== Probabilistic depth NLL / learned attenuation =====
        nll = torch.exp(-depth_logvar) * log_error.abs() + self.prob_lambda * depth_logvar
        nll = nll[mask]

        return nll.mean()