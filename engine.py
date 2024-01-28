import os
import sys
import math
import torch
import pickle as pkl
from tqdm.auto import tqdm
from typing import Iterable

import utils


def encode_concetps():
    ...


def encode_images(data_loader: Iterable, encoder, output_dir: str, device: torch.device):
    all_imgs_encoded = []
    for imgs, _ in tqdm(data_loader):
        imgs.to(device)
        imgs_encoded = encoder.encode_image(imgs)
        imgs_encoded /= torch.linalg.norm(imgs_encoded, dim=-1, keepdim=True)
        all_imgs_encoded.append(imgs_encoded.cpu())
    all_imgs_encoded = torch.cat(all_imgs_encoded)
    torch.save(all_imgs_encoded, output_dir)


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        concepts: torch.Tensor,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="\t")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train_acc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in tqdm(metric_logger.log_every(data_loader, print_freq, header),
                                 total=len(data_loader)):
        samples = samples.to(device)

        outputs = model(samples)
        
        loss = criterion(outputs=outputs,
                         targets=targets,
                         weights=model[0].weight,
                         full_concept_emb=concepts)

        acc = torch.sum(outputs.argmax(-1) == targets) / targets.size(0)

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss)
        metric_logger.update(train_acc=acc)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        concepts: torch.Tensor,
        data_loader: Iterable,
        device: torch.device):
    model.eval()
    criterion.eval()
    all_preds, all_targets = [], []

    header = 'Test: '
    metric_logger = utils.MetricLogger(delimiter="\t")
    for samples, targets in tqdm(metric_logger.log_every(data_loader, 100, header),
                                 total=len(data_loader)):
        samples, targets = samples.to(device), targets.to(device)
        outputs = model(samples).to('cpu')

        loss = criterion(outputs=outputs,
                         targets=targets,
                         weights=model[0].weight,
                         full_concept_emb=concepts)

        preds = torch.argmax(outputs, dim=-1)
        acc = (torch.sum(preds == targets) / all_targets.size(0) * 100)

        metric_logger.update(loss=loss)
        metric_logger.update(test_acc=acc)
    
        all_preds.append(preds)
        all_targets.append(targets)

    # all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
    # acc = (torch.sum(all_preds == all_targets) / all_targets.size(0) * 100)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
