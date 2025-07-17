import torch
import torch.nn as nn
import math

P, A = -0.5, -0.00390625
M1, M2, B =  0.5, -0.125, -0.3125


def gelupw(x: torch.Tensor) -> torch.Tensor:
    res = torch.empty_like(x)

    # case 1: x >= 0
    mask = x >= 0
    res[mask] = x[mask]

    # case 2: P <= x < 0
    mask = (x < 0) & (x >= P)
    res[mask] = M1 * x[mask]

    # case 3: A <= x < P
    mask = (x < P) & (x >= A)
    res[mask] = M2 * x[mask] + B

    # case 4: x < A
    mask = x < A
    res[mask] = 0

    return res


class GeluPW(nn.Module):
  def __init__(self):
      super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return gelupw(x)


