'''
This module contains options for training and hidden networks
'''

class TrainingOptions(object):
    '''
    Configurations for training
    '''
    def __init__(self, batch_size: int, number_of_epochs: int,
                 train_folder: str, validation_folder: str,
                 runs_folder: str, experiment_name: str):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.experiment_name = experiment_name


class HiDDenConfiguration(object):
    '''The hidden network configuration'''
    def __init__(self, img_width: int, img_height: int,
                 message_length: int,
                 encoder_blocks: int, encoder_channels: int,
                 decoder_blocks: int, decoder_channels: int,
                 discriminator_blocks: int, discriminator_channels: int,
                 decoder_loss: float, encoder_loss: float,
                 adversarial_loss: float):
        self.img_width, self.img_height = img_width, img_height
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.encoder_loss = encoder_loss
        self.decoder_loss = decoder_loss
        self.adversarial_loss = adversarial_loss
        self.message_length = message_length
