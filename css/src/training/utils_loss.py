import torch
import torch.nn.functional as F


def supcon_loss(z, labels, temperature):
    N = z.size(0)
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.t()) / temperature
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    logits = sim - 1e4 * torch.eye(N, device=z.device)
    exp_logits = torch.exp(logits) * (~torch.eye(N, device=z.device).bool())
    pos_logits = exp_logits * mask
    pos_sum = pos_logits.sum(1)
    all_sum = exp_logits.sum(1)
    loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
    return loss.mean()


def ortho_reg(W):
    WT_W = torch.matmul(W, W.t())
    I = torch.eye(W.size(0), device=W.device)
    return ((WT_W - I) ** 2).sum()


def l_rank(z, pair_ids, safer, proto_safe, margin):
    z = F.normalize(z, dim=-1)
    proto = proto_safe
    dist = 1 - torch.matmul(z, proto)
    safer = safer.float()
    d_safer = dist * safer
    d_less = dist * (1 - safer)
    pos = d_safer.mean()
    neg = d_less.mean()
    return torch.relu(margin + pos - neg)
