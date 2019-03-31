import numpy as np
import os
import re
import csv
import pickle
import logging
import time

import torch
from torchvision import transforms, datasets
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import HiDDen

def get_data_loaders(hidden_config: HiDDenConfiguration,
                     train_options: TrainingOptions):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.img_height, hidden_config.img_width),
                                  pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.img_height, hidden_config.img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_options.train_folder,
                                        data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images,
                                               batch_size=train_options.batch_size,
                                               shuffle=True, num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder,
                                             data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images,
                                                    batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)

    return train_loader, validation_loader

def print_progress(losses_accu):
    log_print_helper(losses_accu, print)

def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)

def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4)
                          + f'{loss_value.avg:.4f}')

def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu] \
                          + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()]\
                       + ['{:.0f}'.format(duration)]
        writer.writerow(row_to_write)

def save_checkpoint(model: HiDDen, experiment_name: str, epoch: int,
                     checkpoint_folder: str):
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info(f'Saving checkpoint to {checkpoint_filename}')
    checkpoint = {
        'enc-dec-model': model.enc_dec.state_dict(),
        'enc-dec-optim': model.opt_enc_dec.state_dict(),
        'discrim-model': model.discr.state_dict(),
        'discrim-optim': model.opt_discr.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done')

def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.join(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name}{time.strftime("%Y.%m.%d")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))

    return this_run_folder

def load_options(option_folder):
    with open(os.path.join(option_folder, 'options.pickle'), 'rb') as f:
        train_options = pickle.load(f)
        hidden_config = pickle.load(f)

    return train_options, hidden_config

def model_from_checkpoint(model: HiDDen, checkpoint):
    model.enc_dec.load_state_dict(checkpoint['enc-dec-model'])
    model.opt_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    model.discr.load_state_dict(checkpoint['discrim-model'])
    model.opt_discr.load_state_dict(checkpoint['discrim-optim'])
