import os
import clip
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.lm4cv import Stage1Criterion
from datasets.cub_dataset import CUBDataset
from engine import train_one_epoch, evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch-size', default=4096, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--stage-one-epochs', default=5000, type=int)
    parser.add_argument('--embedder-type', default='clip', choices=['clip', 'open-clip'], type=str)
    parser.add_argument('--embedder-variant', default='ViT-B/32', type=str)
    parser.add_argument('--img-emb-dir', default=None, type=str)
    parser.add_argument('--load-encoded', default=None, type=str,
                        help='Load encoded image features instead of raw images.')



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
        embedder, preprocess = clip.load(args.model_variant)
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
        print(batch_concept_emb.size())
        full_concept_emb.append(embedder.encode_text(batch_concept_emb).detach().cpu())

    full_concept_emb = torch.concat(full_concept_emb).float()
    full_concept_emb = full_concept_emb / full_concept_emb.norm(dim=-1, keepdim=True)   # Matrix T, Tensor[N, D]


    # If not using the full matrix T
    if not args.num_concepts:
        args.num_concepts = full_concept_emb.size(0)
    print ("Number of concepts to search for: ", args.num_concepts)
    output_dim = 200

    train_img_dataset = CUBDataset('data', 'train')
    train_img_dataloader = DataLoader(train_img_dataset, 256)
    test_img_dataset = CUBDataset('data', 'test')
    test_img_dataloader = DataLoader(test_img_dataset, 256)

    print('Encode images...')
    all_imgs_encoded = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_img_dataloader), total=len(train_img_dataloader)):
            imgs, targets = batch
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)
            imgs_encoded = embedder.encode_image(imgs)
            imgs_encoded /= torch.linalg.norm(imgs_encoded, dim=-1, keepdim=True)
            all_imgs_encoded.append(imgs_encoded.to('cpu'))
    all_imgs_encoded = torch.cat(all_imgs_encoded).numpy()
    np.save('data/CUB_200_2011/images_encoded.npy', all_imgs_encoded)

    # output_dim = 200
    model = nn.Sequential(nn.Linear(full_concept_emb.shape[-1], args.num_concepts, bias=False),
                            nn.Linear(args.num_concepts, output_dim))
    
    criterion = Stage1Criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.stage_one_epochs):
        train_stats = train_one_epoch(model, criterion, full_concept_emb,
                                        train_img_dataloader, optimizer, args.device, epoch)
        
        test_stats = evaluate(model, criterion, full_concept_emb, test_img_dataset, args.device)

    # TODO: Select the best model
    
    