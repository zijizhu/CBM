import os
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from datasets.cub_dataset import CUBDataset
from engine import train_one_epoch, evaluate
from models.lm4cv import Stage1Criterion, TopConceptSearcher


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch-size', default=4096, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--stage-one-epochs', default=5000, type=int)
    parser.add_argument('--stage-two-epochs', default=5000, type=int)
    parser.add_argument('--dataset-dir', type=str)

    parser.add_argument('--num-concepts', default=None, type=int)

    args = parser.parse_args()
    print(args)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Load dataset
    train_img_dataset = CUBDataset(args.dataset_dir, 'train', encoded=True)
    train_img_dataloader = DataLoader(train_img_dataset, 256)
    test_img_dataset = CUBDataset(args.dataset_dir, 'test', encoded=True)
    test_img_dataloader = DataLoader(test_img_dataset, 256)

    concepts_encoded = torch.load(os.path.join(args.dataset_dir, 'concepts_encoded.pt'))

    # Number of concepts and classes
    if not args.num_concepts:
        args.num_concepts = concepts_encoded.size(0)
    print ("Number of concepts to search for: ", args.num_concepts)
    num_classes = 200

    # Model, loss and optimizer
    model = nn.Sequential(nn.Linear(concepts_encoded.size(-1), args.num_concepts, bias=False),
                          nn.Linear(args.num_concepts, num_classes))
    
    criterion = Stage1Criterion()
    searcher = TopConceptSearcher(args.num_concepts)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Stage 1 training
    print('Stage 1 training:')
    for epoch in range(args.stage_one_epochs):
        train_stats = train_one_epoch(model, criterion, concepts_encoded,
                                      train_img_dataloader, optimizer, args.device, epoch)
        
        test_stats = evaluate(model, criterion, concepts_encoded, test_img_dataset, args.device)

    # TODO: Select the best model
    
    # Select top concepts
    top_concepts_encoded, selected_idxs = searcher(model[0].weight, concepts_encoded)


    # Stage 2 training
    print('Stage 2 training:')
    model[0].weight.data = top_concepts_encoded * torch.linalg.vector_norm(model[0].weight.data, dim=-1)
    for param in model[0].parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.stage_one_epochs):
        train_stats = train_one_epoch(model, criterion, concepts_encoded,
                                      train_img_dataloader, optimizer, args.device, epoch)
        
        test_stats = evaluate(model, criterion, concepts_encoded, test_img_dataset, args.device)
