import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration


class EncoderDecoder(nn.Module):
    ''' Combine encoder and decoder '''
    def __init__(self, config: HiDDenConfiguration):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, image, message):
        '''Run image and message through generator'''
        encoded_image = self.encoder(image, message)
        decoded_message = self.decoder(encoded_image)
        return encoded_image, decoded_message
