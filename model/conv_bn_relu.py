import torch.nn as nn

class ConvBNRelu(nn.Module):
    '''
    Building block of a hidden networks. It is a sequence of Convolution,
    Batch normalization and Relu Activation
    '''
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
