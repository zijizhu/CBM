import torch
import clip
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, model) -> None:
        self.model = model
    
    def forward(self, images):
        encoded = self.model.encode_image(images)