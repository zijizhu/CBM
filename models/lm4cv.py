import torch
from torch import nn


def mahalanobis_distance(x, mu, sigma_inv):
    x = x - mu.unsqueeze(0)
    return torch.diag(x @ sigma_inv @ x.T).mean()


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
        mean_distance = torch.mean([mahalanobis_distance(embed, mu, sigma_inv) for embed in full_concept_emb]).to(device)

        mahalanobis_loss = (mahalanobis_distance(weights / weights_norm, mu, sigma_inv) - mean_distance) / (mean_distance ** 3)

        return xe_loss + mahalanobis_loss


class ImageEncoder(nn.Module):
    @torch.no_grad()
    def __init__(self, model) -> None:
        self.model = model
    
    def forward(self, images):
        encoded = self.model.encode_image(images)