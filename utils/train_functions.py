import torch
from utils.pytorch_tools import AverageMeter
from tqdm import tqdm
import numpy as np


def train_one_epoch(model, loader, optimizer, criterion, epoch, device):

    model.train()

    loss_meter = AverageMeter(metric_name='Loss')
    correct_meter = AverageMeter(metric_name='Correct')

    pbar = tqdm(enumerate(filter(None, loader)), total=len(loader))

    for i, (inputs, labels) in pbar:
        # zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, dim=1) # get classes
        loss = criterion(outputs, labels) # loss

        loss_meter.update(loss.item(), len(labels))
        correct_meter.update((predicted == labels).sum().item(), len(labels))

        loss.backward()
        optimizer.step()

        pbar.desc = f'[{epoch:0>3}] Train -- loss: {loss_meter.get_total():0.2f} -- acc: {(100 * correct_meter.get_avg()):0.2f}%'
    
    return {'train_loss': loss_meter.get_total(), 'train_acc': correct_meter.get_avg()}


def eval_one_epoch(model, loader, criterion, epoch, device):

    loss_meter = AverageMeter(metric_name='Loss')
    correct_meter = AverageMeter(metric_name='Correct')

    with torch.no_grad():
        model.eval()

        pbar = tqdm(enumerate(filter(None, loader)), total=len(loader))
        for i, (inputs, labels) in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, dim=1) # get classes
            loss = criterion(outputs, labels) # loss

            loss_meter.update(loss.item(), len(labels))
            correct_meter.update((predicted == labels).sum().item(), len(labels))

            pbar.desc = f'[{epoch:0>3}] Eval  -- loss: {loss_meter.get_total():0.2f} -- acc: {(100 * correct_meter.get_avg()):0.2f}%'

    return {'val_loss': loss_meter.get_total(), 'val_acc': correct_meter.get_avg()}