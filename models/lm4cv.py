import torch
from torch import nn


def _mean_squared_mahalanobis(vecs: torch.Tensor, distribution: torch.Tensor):
    '''Computes the mean of squared mahalanobis distances from a vector or a set of vectors to the distribution.
    Implementation from https://github.com/wangyu-ustc/LM4CV/blob/main/utils/train_utils.py#L263
    
    Args:
        vec (Tensor[M, D]): a set of vector of length D.
        distribution (Tensor[N, D]) a matrix of N vectors of length D
    
    Returns:
        Tensor[]: a scaler tensor, which is the mahalanobis distance from vec to the distribution.
    '''

    mu = torch.mean(distribution, dim=0)
    sigma_inv = torch.linalg.inv(torch.cov(distribution.T))
    vecs = vecs - mu.unsqueeze(0)
    return torch.diag(vecs @ sigma_inv @ vecs.T).mean()


class Stage1Criterion(nn.Module):
    def __init__(self, regularization=True, division_power=3) -> None:
        super().__init__()
        self.xe = nn.CrossEntropyLoss()
        self.regularization = regularization
        self.division_power = division_power
    
    def forward(self, outputs: torch.Tensor, targets, weights, full_concept_emb):
        xe_loss = self.xe(outputs, targets)
        if not self.regularization:
            return xe_loss

        # Original implementation from https://github.com/wangyu-ustc/LM4CV/blob/main/utils/train_utils.py#L208
        # which is different to the one described in the paper.
        weights_norm = torch.linalg.norm(weights, dim=-1, keepdim=True)

        mu = torch.mean(full_concept_emb, dim=0)
        sigma_inv = torch.inverse(torch.cov(full_concept_emb.T))
        mean_distance = torch.stack([_mean_squared_mahalanobis(embed, mu, sigma_inv)
                                     for embed
                                     in full_concept_emb]).mean().to(outputs.device)

        mahalanobis_loss = _mean_squared_mahalanobis(weights / weights_norm, mu, sigma_inv) 

        return xe_loss + (mahalanobis_loss - mean_distance) / (mean_distance ** self.division_power)


class ImageEncoder(nn.Module):
    @torch.no_grad()
    def __init__(self, model) -> None:
        self.model = model
    
    def forward(self, images):
        encoded = self.model.encode_image(images)