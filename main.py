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

    parser.add_argument('--num-concepts', default=None, type=int)

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
    raw_concepts = open("data/LM4CV/CUB_200_2011/cub_attributes.txt", 'r').read().strip().split("\n")
    # print(len(attributes))

    full_concept_emb = []   # Matrix T
    batch_size = 32

    prompt_prefix = 'The bird has '
    num_batches = len(raw_concepts) // batch_size + 1
    for i in range(num_batches):
        batch_concepts = raw_concepts[i * batch_size: (i + 1) * batch_size]
        batch_concept_emb = clip.tokenize([prompt_prefix + attr for attr in batch_concepts])
        if args.device == 'cuda':
            batch_concept_emb.cuda()
        full_concept_emb += [emb.detach().cpu() for emb in model.encode_text(batch_concept_emb)]
    
    full_concept_emb = torch.stack(full_concept_emb).float()
    full_concept_emb = full_concept_emb / full_concept_emb.norm(dim=-1, keepdim=True)   # Matrix T

    print ("Number of concepts: ", args.num_concepts)

    # If not using the full matrix T
    if args.num_concepts < 1000:
        # Use linear method to cluster features

        mu = torch.mean(full_concept_emb, dim=0)
        sigma_inv = torch.tensor(np.linalg.inv(torch.cov(full_concept_emb.T)))
        mean_distance = np.mean([mahalanobis_distance(embed, mu, sigma_inv) for embed in full_concept_emb])

        output_dim = 200

        # output_dim = 200
        model = nn.Sequential(nn.Linear(full_concept_emb.shape[-1], args.num_concepts, bias=False),
                              nn.Linear(args.num_concepts, output_dim))

    
    