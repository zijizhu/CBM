import os
import clip
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.cub_dataset import CUBDataset


dataset2clip_prompt_prefix = {'CUB_200_2011': 'The bird has '}


@torch.inference_mode()
def encode_concepts(model, raw_concepts, prompt_prefix, output_dir, batch_size, rescale=True, device='cpu'):
    all_encoded = []   # Matrix T

    prompt_prefix = 'The bird has '
    num_batches = len(raw_concepts) // batch_size + 1
    for i in range(num_batches):
        batch_concepts = raw_concepts[i * batch_size: (i + 1) * batch_size]
        batch_concepts_token = clip.tokenize([prompt_prefix + attr for attr in batch_concepts]).to(device)
        all_encoded.append(model.encode_text(batch_concepts_token))

    all_encoded = torch.cat(all_encoded).cpu()
    # Rescale each row to a unit vector
    if rescale:
        all_encoded /= torch.linalg.vector_norm(all_encoded, dim=-1, keepdim=True)   # Matrix T, Tensor[N, D]

    torch.save(all_encoded, os.path.join(output_dir, 'concepts_encoded.pt'))
    

@torch.inference_mode()
def encode_images(model, data_loader, split, output_dir, rescale=True, device='cpu'):
    all_encoded = []
    all_filenames = []
    for filenames, imgs, _ in tqdm(data_loader):
        imgs = imgs.to(device)
        encoded = model.encode_image(imgs)
        all_encoded.append(encoded)
        all_filenames += filenames

    all_encoded = torch.cat(all_encoded).cpu()
    if rescale:
        all_encoded /= torch.linalg.vector_norm(all_encoded, dim=-1, keepdim=True)

    torch.save(all_encoded, os.path.join(output_dir, f'{split}_images_encoded.pt'))
    torch.save(all_filenames, os.path.join(output_dir, f'{split}_filename2idx.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder-type', default='clip', choices=['clip', 'open-clip'], type=str)
    parser.add_argument('--encoder-backbone', default='ViT-B/32', type=str)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], type=str)
    parser.add_argument('--image-batch-size', default=128, type=int)
    parser.add_argument('--concept-batch-size', default=128, type=int)
    parser.add_argument('--dataset-dir', type=str)
    parser.add_argument('--concept-dir', type=str)
    parser.add_argument('--seed', default=42, type=int)

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
    raw_concepts = open(args.concept_dir, 'r').read().strip().split("\n")
    train_image_dataset = CUBDataset(args.dataset_dir, encoded=False, split='train', transforms=preprocessor)
    test_image_dataset = CUBDataset(args.dataset_dir, encoded=False, split='test', transforms=preprocessor)
    train_data_loader = DataLoader(train_image_dataset, batch_size=args.image_batch_size,
                                   shuffle=False, num_workers=8)
    test_data_loader = DataLoader(test_image_dataset, batch_size=args.image_batch_size,
                                  shuffle=False, num_workers=8)

    # Encode Concepts
    print('Encoding concepts...')
    encode_concepts(encoder, raw_concepts, dataset2clip_prompt_prefix[os.path.basename(os.path.normpath(args.dataset_dir))],
                    args.dataset_dir, args.concept_batch_size, device=args.device)

    # Encode Images
    print('Encoding training set images...')
    encode_images(encoder, train_data_loader, 'train', args.dataset_dir, device=args.device)

    print('Encoding test set images...')
    encode_images(encoder, test_data_loader, 'test', args.dataset_dir, device=args.device)
