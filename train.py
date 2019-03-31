import os
import time
import numpy as np
import logging

import torch

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import HiDDen
from average_meter import AverageMeter
import hidden_utils


def train(model: HiDDen, device: torch.device, this_run_folder: str,
          hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    train_data, val_data = hidden_utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    steps_in_epoch = file_count // train_options.batch_size \
                    + int(file_count % train_options.batch_size != 0)

    print_each = 10
    saved_images_size = (512, 512)

    for epoch in range(train_options.number_of_epochs):
        logging.info(f'\nStarting epoch {epoch + 1} / {train_options.number_of_epochs}')
        logging.info(f'Batch size = {train_options.batch_size}')
        logging.info(f'Steps in epoch {steps_in_epoch}')
        losses_accu = {} 
        epoch_start = time.time()
        step = 1

        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(
                np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))
            ).to(device)
            losses, _ = model.train_on_batch([image, message])

            if not losses_accu:
                for name in losses:
                    losses_accu[name] = AverageMeter()

            for name, loss in losses.items():
                losses_accu[name].update(loss)

            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(f'Epoch: {epoch + 1}/{train_options.number_of_epochs} Step: {step}/{steps_in_epoch}')
                hidden_utils.log_progress(losses_accu)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info(f'Epoch {epoch + 1} training duration {train_duration:.2f}')
        logging.info('-' * 40)
        hidden_utils.write_losses(os.path.join(this_run_folder, 'train.csv'),
                                  losses_accu, epoch, train_duration)

        logging.info(f'Running validation for epoch {epoch + 1} / {train_options.number_of_epochs}')
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(
                np.random.choice([0, 1],
                                 (image.shape[0], hidden_config.message_length))
            ).to(device)
        losses, (encoded_images, decoded_messages) = model.validate_on_batch([image, message])
        hidden_utils.log_progress(losses_accu)
        logging.info('-' * 40)
        hidden_utils.save_checkpoint(model, train_options.experiment_name,
                                     epoch, os.path.join(this_run_folder, 'checkpoints'))
        hidden_utils.write_losses(os.path.join(this_run_folder, 'validation.csv'),
                                  losses_accu, epoch, time.time() - epoch_start)
