import torch
from torch import nn


def mean_mahalanobis_distance(vecs: torch.Tensor, distribution: torch.Tensor):
    '''Computes the mean of mahalanobis distance from a set of vectors to the distribution.
    
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
    def __init__(self, regularization=True) -> None:
        super().__init__()
        self.xe = nn.CrossEntropyLoss()
        self.regularization = regularization
    
    def forward(self, outputs: torch.Tensor, targets, weights, full_concept_emb):
        xe_loss = self.xe(outputs, targets)
        if not self.regularization:
            return xe_loss

        weights_norm = torch.linalg.norm(weights, dim=-1, keepdim=True)

        device = outputs.device
        mu = torch.mean(full_concept_emb, dim=0).to(device)
        sigma_inv = torch.linalg.inv(torch.cov(full_concept_emb.T)).to(device)
        mean_distance = torch.mean([mean_mahalanobis_distance(embed, mu, sigma_inv) for embed in full_concept_emb]).to(device)

        mahalanobis_loss = (mean_mahalanobis_distance(weights / weights_norm, mu, sigma_inv) - mean_distance) / (mean_distance ** 3)

        return xe_loss + mahalanobis_loss


class ImageEncoder(nn.Module):
    @torch.no_grad()
    def __init__(self, model) -> None:
        self.model = model
    
    def forward(self, images):
        encoded = self.model.encode_image(images)