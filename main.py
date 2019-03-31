import os
import pprint
import argparse
import pickle
import logging
import os
import sys

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import HiDDen
from train import train
import hidden_utils

import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parent_parser = argparse.ArgumentParser(description='Training the HiDDeN nets')
    parent_parser.add_argument('--data-dir', '-d', required=True, type=str,
                               help='The directory where images stored')
    parent_parser.add_argument('--batch-size', '-b', required=True, type=int,
                               help='The batch size')
    parent_parser.add_argument('--epochs', '-e', default=300, type=int,
                               help='Number of epochs to run simulation')
    parent_parser.add_argument('--name', required=True, type=str,
                               help='The name of experiments')
    parent_parser.add_argument('--size', '-s', default=128, type=int,
                               help='The size of the image')
    parent_parser.add_argument('--message', '-m', default=30, type=int,
                               help='The length of the message')

    args = parent_parser.parse_args()
    train_options = TrainingOptions(
        batch_size = args.batch_size,
        number_of_epochs = args.epochs,
        train_folder = os.path.join(args.data_dir, 'train'),
        validation_folder = os.path.join(args.data_dir, 'val'),
        runs_folder = os.path.join('.', 'runs'),
        experiment_name = args.name
    )

    hidden_config = HiDDenConfiguration(img_width=args.size, img_height=args.size,
                                        message_length=args.message,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        decoder_loss=1, encoder_loss=0.7,
                                        adversarial_loss=1e-3)

    this_run_folder = hidden_utils.create_folder_for_run(train_options.runs_folder,
                                                         args.name)
    with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
        pickle.dump(train_options, f)
        pickle.dump(hidden_config, f)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(
                                os.path.join(this_run_folder,
                                            f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])

    model = HiDDen(hidden_config, device)
    logging.info(f'HiDDeN Model {model.to_stirng}')
    logging.info('\nModel Configuration:\n')
    logging.info(pprint.pformat(vars(hidden_config)))
    logging.info('\nTraining Options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    train(model, device, this_run_folder, hidden_config, train_options)


if __name__ == '__main__':
    main()
