import os
import clip
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn


def mahalanobis_distance(x, mu, sigma_inv):
    x = x - mu.unsqueeze(0)
    return torch.diag(x @ sigma_inv @ x.T).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch-size', default=4096, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--model-type', default='clip', choices=['clip', 'open-clip'], type=str)
    parser.add_argument('--model-variant', default='ViT-B/32', type=str)

    parser.add_argument('--num-concepts', default=200, type=int, choices=[32, 200, 400])

    args = parser.parse_args()
    print(args)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Get clip model
    if args.model_type == 'clip':
        model, preprocess = clip.load(args.model_variant)
    elif args.model_type == 'open_clip':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # 
    attributes = open("data/LM4CV/CUB_200_2011/cub_attributes.txt", 'r').read().strip().split("\n")
    # print(len(attributes))

    attr_emb = []
    batch_size = 32

    prompt_prefix = 'The bird has '
    for i in range((len(attributes) // batch_size) + 1):
        batch_attr = attributes[i * batch_size: (i + 1) * batch_size]
        batch_attr_emb = clip.tokenize([prompt_prefix + attr for attr in batch_attr])
        print(batch_attr_emb.size())
        print(model.encode_text(batch_attr_emb).size(), model.encode_text(batch_attr_emb).dtype)
        if args.device == 'cuda':
            batch_attr_emb.cuda()
        attr_emb += [emb.detach().cpu() for emb in model.encode_text(batch_attr_emb)]
    
    attr_emb = torch.stack(attr_emb).float()
    attr_emb = attr_emb / attr_emb.norm(dim=-1, keepdim=True)   # Matrix T

    print ("Number of concepts: ", args.num_concepts)

    # If not using the full matrix T
    if args.num_concepts < 1000:
        # Use linear method to cluster features

        mu = torch.mean(attr_emb, dim=0)
        sigma_inv = torch.tensor(np.linalg.inv(torch.cov(attr_emb.T)))
        configs = {
            'mu': mu,
            'sigma_inv': sigma_inv,
            'mean_distance': np.mean([mahalanobis_distance(embed, mu, sigma_inv) for embed in attr_emb])
        }

        output_dim = len(np.unique(get_labels('cub')[0]))

        # output_dim = 200
        model = nn.Sequential(nn.Linear(attr_emb.shape[-1], args.num_concepts, bias=False),
                              nn.Linear(args.num_concepts, output_dim))

    
    