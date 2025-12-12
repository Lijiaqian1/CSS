import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)

    def forward(self, h):
        dtype = self.W.weight.dtype
        h = h.to(dtype)
        z = self.W(h)
        z = F.normalize(z, dim=-1)
        return z
