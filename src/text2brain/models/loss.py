import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kl_divergence(logvar, mu, reduction: str):
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif reduction == "mean":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif reduction == "none":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class ELBOLoss(nn.Module):
    def __init__(self, reduction: str):
        super(ELBOLoss, self).__init__()
        self.reduction = reduction

    def forward(self, reconstructed_x, x, mu, logvar):
        mse = F.mse_loss(reconstructed_x, x, reduction=self.reduction)
        kld = compute_kl_divergence(logvar=logvar, mu=mu, reduction=self.reduction)
        return mse, kld


# implementation from: https://github.com/applied-ai-lab/genesis/blob/master/utils/geco.py
class GECOLoss(nn.Module):
    def __init__(
        self,
        goal,
        step_size,
        reduction: str,
        device: str,
        alpha=0.99,
        beta_init=1.0,
        beta_min=1e-10,
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
        self.beta_max = torch.tensor(1e10)
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

    def forward(self, reconstructed_x, x, mu, logvar):
        mse = F.mse_loss(reconstructed_x, x, reduction=self.reduction)
        kld = compute_kl_divergence(logvar=logvar, mu=mu, reduction=self.reduction)
        loss = self.compute_contrained_loss(err=mse, kld=kld)

        return loss
