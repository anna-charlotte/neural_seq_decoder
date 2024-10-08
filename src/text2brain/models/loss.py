from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# LossResults = namedtuple("LossResults", ["mse", "kld", "loss"])

@dataclass
class LossResults:
    mse: torch.Tensor
    kld: torch.Tensor
    loss: torch.Tensor


def compute_kl_divergence(logvar, mu, reduction: str):
    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "sum":
        return torch.sum(kl_div)
    elif reduction == "mean":
        return torch.mean(kl_div)
    elif reduction == "none":
        return kl_div
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class ELBOLoss(nn.Module):
    def __init__(self, reduction: str, beta: float = 0.1):
        super(ELBOLoss, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, reconstructed_x, x, mu, logvar) -> LossResults:
        mse = F.mse_loss(reconstructed_x, x, reduction=self.reduction)
        kld = compute_kl_divergence(logvar=logvar, mu=mu, reduction=self.reduction)
        loss = mse + self.beta * kld
        return LossResults(mse=mse, kld=kld, loss=loss)


# implementation from: https://github.com/applied-ai-lab/genesis/blob/master/utils/geco.py
class GECOLoss(nn.Module):
    def __init__(
        self,
        goal: float,
        step_size: float,
        reduction: str,
        device: str,
        alpha: float = 0.99,
        beta_init: float = 1.0,
        beta_min: float = 1e-10,
        beta_max: float = 1e10,
        speedup=None,
    ):
        super(GECOLoss, self).__init__()
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.reduction = reduction
        self.alpha = alpha
        self.beta = torch.tensor(beta_init)
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(beta_max)
        self.speedup = speedup

        self.to(device)

    def to(self, device: str):
        self.beta = self.beta.to(device)
        self.beta_min = self.beta_min.to(device)
        self.beta_max = self.beta_max.to(device)
        if self.err_ema is not None:
            self.err_ema = self.err_ema.to(device)

    def compute_contrained_loss(self, err, kld):
        # Compute loss with current beta
        loss = err + self.beta * kld

        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0 - self.alpha) * err + self.alpha * self.err_ema
            constraint = self.goal - self.err_ema

            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)

            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)

        return loss

    def forward(self, reconstructed_x, x, mu, logvar) -> LossResults:
        mse = F.mse_loss(reconstructed_x, x, reduction=self.reduction)
        kld = compute_kl_divergence(logvar=logvar, mu=mu, reduction=self.reduction)
        loss = self.compute_contrained_loss(err=mse, kld=kld)

        return LossResults(mse=mse, kld=kld, loss=loss)
