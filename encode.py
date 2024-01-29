import os
import clip
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.cub_dataset import CUBDataset


def encode_concepts(model, concepts_dir, output_dir):
    ...


def encode_images(model, preprocessor, data_loader, split, output_dir):
    all_encoded = []
    all_filenames = []
    for filenames, imgs, _ in tqdm(data_loader):
        imgs_preprocessed = preprocessor(imgs)
        encoded = model.encode_image(imgs_preprocessed)
        all_encoded.append(encoded)
        all_filenames += filenames
    all_encoded = torch.cat(all_encoded).cpu()
    torch.save(all_encoded, os.path.join(output_dir, f'{split}_images_encoded.pt'))
    torch.save(all_filenames, os.path.join(output_dir, f'{split}_filename2idx.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder-type', default='clip', choices=['clip', 'open-clip'], type=str)
    parser.add_argument('--encoder-backbone', default='ViT-B/32', type=str)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], type=str)

    args = parser.parse_args()
    print(args)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Load encoder model
    if args.encoder_type == 'clip':
        encoder, preprocessor = clip.load(args.encoder_backbone)
    elif args.encoder_type == 'open_clip':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # Load data
    train_image_dataset = CUBDataset('data/CUB_200_2011', encoded=False, split='train')
    test_image_dataset = CUBDataset('data/CUB_200_2011', encoded=False, split='test')
    train_data_loader = DataLoader(train_image_dataset, shuffle=False, num_workers=8)
    test_data_loader = DataLoader(test_image_dataset, shuffle=False, num_workers=8)

    # Encode Concepts
    print('Encoding concepts...')


    # Encode Images
    output_dir = os.path.join(args.data_dir)
    print('Encoding training set images...')
    encode_images(encoder, preprocessor, train_data_loader, 'train', args.data_dir)

    print('Encoding test set images...')
    encode_images(encoder, preprocessor, test_data_loader, 'test', args.data_dir)