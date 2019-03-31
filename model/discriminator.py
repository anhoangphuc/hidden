import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Discriminator(nn.Module):
    '''
    Discriminator network. Predict whether the image is real or fake
    '''
    def __init__(self, config: HiDDenConfiguration):
        super().__init__()
        layers = [ConvBNRelu(3, config.discriminator_channels)]
        for _ in range(config.discriminator_blocks - 1):
            layer = ConvBNRelu(config.discriminator_channels,
                               config.discriminator_channels)
            layers.append(layer)

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(config.discriminator_channels, 1)

    def  forward(self, image):
        x = self.before_linear(image)
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
