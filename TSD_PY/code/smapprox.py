import torch
import torch.nn as nn
import math
import sys

def softmax_taylor(x: torch.Tensor, dim: int = -1, n_terms: int = 3) -> torch.Tensor:
    # Shift for numerical stability
    x.sub_(x.max(dim=dim, keepdim=True).values)

    taylor = torch.ones_like(x)

    for k in range(n_terms-1, 0, -1):
        # Using horner's rule for taylor expansion
        taylor.mul_(x).div_(k).add_(1)
    
    # Clamp to avoid out of bounds, plus some internal margins for imprecisions
    return taylor.div(taylor.sum(dim=dim, keepdim=True)).clamp_(0.0+1e-6,1.0-1e-6)

class SoftmaxTaylor(nn.Module):
  def __init__(self, dim: int = -1, n_terms: int = 3):
      super().__init__()
      self.dim     = dim
      self.n_terms = n_terms

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return softmax_taylor(x, dim=self.dim, n_terms=self.n_terms)


