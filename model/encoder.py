import torch
import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu
from options import HiDDenConfiguration


class Encoder(nn.Module):
    '''
    Encoder block
    '''
    def __init__(self, config: HiDDenConfiguration):
        super().__init__()
        self.img_width = config.img_width
        self.img_height = config.img_height

        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]
        for _ in range(self.num_blocks - 1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)
        self.conv_layers = nn.Sequential(*layers)

        self.after_concat_layer = ConvBNRelu(
            self.conv_channels + 3 + config.message_length,
            self.conv_channels
        )

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):
        '''
        Insert message into image
        '''
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1, -1,
                                                   self.img_height,
                                                   self.img_width)
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        temp_img = self.after_concat_layer(concat)
        temp_img = self.final_layer(temp_img)

        return temp_img
