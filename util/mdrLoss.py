import torch
import torch.nn as nn
import torch.nn.functional as F

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class MDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels = nn.Parameter(torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]))
        self.momentum = 0.9

        momented_mean = torch.zeros(1)
        momented_std = torch.zeros(1)
        self.register_buffer("momented_mean", momented_mean)
        self.register_buffer("momented_std", momented_std)

        # The variable is used to check whether momented_mean and momented_std are initialized
        self.init = False

    def initialize_statistics(self, mean, std):
        self.momented_mean = mean
        self.momented_std = std
        self.init = True

    def forward(self, embeddings):
        dist_mat = pdist(embeddings)
        pdist_mat = dist_mat[
            ~torch.eye(dist_mat.shape[0], dtype=torch.bool, device=dist_mat.device,)
        ]
        dist_mat = dist_mat.view(-1)

        mean = dist_mat.mean().detach()
        std = dist_mat.std().detach()

        if not self.init:
            self.initialize_statistics(mean, std)
        else:
            self.momented_mean = (
                1 - self.momentum
            ) * mean + self.momentum * self.momented_mean
            self.momented_std = (
                1 - self.momentum
            ) * std + self.momentum * self.momented_std

        normalized_dist = (pdist_mat - self.momented_mean) / self.momented_std
        difference = (normalized_dist[None] - self.levels[:, None]).abs().min(dim=0)[0]
        loss = difference.mean()

        return loss


if __name__=='__main__':
    criterion = MDRLoss()
    embeddings = torch.randn((16, 256))
    loss = criterion(embeddings)