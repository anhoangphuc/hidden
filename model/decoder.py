import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Decoder(nn.Module):
    '''
    Decoder module
    '''
    def __init__(self, config: HiDDenConfiguration):
        super().__init__()
        self.channels = config.decoder_channels
        self.num_blocks = config.decoder_blocks

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(self.num_blocks - 1):
            layer = ConvBNRelu(self.channels, self.channels)
            layers.append(layer)
        layers.append(ConvBNRelu(self.channels, config.message_length))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, encoded_image):
        x = self.layers(encoded_image)
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
