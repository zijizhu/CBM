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

        def clean_label(true_labels):
            true_labels = np.array(true_labels)
            if np.min(true_labels) > 0:
                true_labels -= np.min(true_labels)
            return true_labels

        def get_labels(dataset):
            if dataset == 'cub':
                with open("./data/CUB_200_2011/image_class_labels.txt", 'r') as file:
                    true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
                true_labels = clean_label(true_labels)
                train_test_split = pd.read_csv(os.path.join('./data/', 'CUB_200_2011', 'train_test_split.txt'),
                                            sep=' ', names=['img_id', 'is_training_img'])
                train_test_split = train_test_split['is_training_img'].values
                train_indices = np.where(train_test_split == 1)
                test_indices = np.where(train_test_split == 0)
                train_labels, test_labels = true_labels[train_indices], true_labels[test_indices]

                return train_labels, test_labels

        output_dim = len(np.unique(get_labels('cub')[0]))

        # output_dim = 200
        model = nn.Sequential(nn.Linear(attr_emb.shape[-1], args.num_concepts, bias=False),
                              nn.Linear(args.num_concepts, output_dim))

    
    