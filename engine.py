import os
import sys
import math
import torch
import pickle as pkl
from tqdm.auto import tqdm
from typing import Iterable

import utils


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        concepts: torch.Tensor,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int):
    model.to(device).train()
    criterion.to(device).train()

    metric_logger = utils.MetricLogger(delimiter="\t")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train_acc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000

    for _, samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        outputs = model(samples)
        
        if concepts is not None:
            loss = criterion(outputs=outputs,
                            targets=targets,
                            weights=model[0].weight,
                            concepts_encoded=concepts)
        else:
            loss = criterion(outputs=outputs, targets=targets)
                            
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
        data_loader: Iterable,
        device: torch.device):
    model.eval()
    criterion.eval()
    all_preds, all_targets = [], []

    header = 'Test: '
    metric_logger = utils.MetricLogger(delimiter="\t")
    for _, samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples, targets = samples.to(device), targets.to(device)
        outputs = model(samples)

        preds = torch.argmax(outputs, dim=-1)
        acc = (torch.sum(preds == targets) / all_targets.size(0) * 100)

        metric_logger.update(test_acc=acc)
    
        all_preds.append(preds)
        all_targets.append(targets)

    # all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
    # acc = (torch.sum(all_preds == all_targets) / all_targets.size(0) * 100)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
