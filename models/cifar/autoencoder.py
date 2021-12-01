import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [
    'daf_small_bottleneck_cnn_autoenconder'
]


def make_cnn_block(inplanes, outplanes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.BatchNorm2d(inplanes),
        nn.ReLU(inplace=True),
        nn.Conv2d(inplanes, outplanes, kernel_size = kernel_size, stride = stride, padding = padding)
    )


def make_tcnn_block(inplanes, outplanes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.BatchNorm2d(inplanes),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(inplanes, outplanes, kernel_size = kernel_size, stride = stride, padding = padding),
        nn.BatchNorm2d(outplanes),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(outplanes, outplanes, kernel_size=2)
    )


class BottleNeckCNNAutoEnconder(nn.Module):
    def __init__(self, planes, num_layers=None, **kwargs):
        super(BottleNeckCNNAutoEnconder, self).__init__()

        # Loss function
        self.criterion = nn.MSELoss(reduction='mean')
        self.kwargs = kwargs

        # Plane mappings
        self.planes = planes

        # Number of intermediate blocks
        if num_layers is None: num_layers = [0, 0, 0, 0, 0, 0]
        self.num_layers = num_layers

        # Create encoder-decoder structure
        self.enc = self._make_encoder()
        self.dec = self._make_decoder()

        # Output function
        self.foutput = nn.Tanh()

    def reinitialise(self):
        return BottleNeckCNNAutoEnconder(
            self.planes, self.num_layers, **self.kwargs
        )

    def _make_encoder(self):
        layers = []
        layers.append(make_cnn_block(self.planes[0], self.planes[1], kernel_size=3, stride=2, padding=1))
        layers.extend([make_cnn_block(self.planes[1], self.planes[1], kernel_size=3, stride=1, padding=1)
                       for _ in range(self.num_layers[0])])

        layers.append(make_cnn_block(self.planes[1], self.planes[2], kernel_size=3, stride=2, padding=1))
        layers.extend([make_cnn_block(self.planes[2], self.planes[2], kernel_size=3, stride=1, padding=1)
                       for _ in range(self.num_layers[1])])

        layers.append(make_cnn_block(self.planes[2], self.planes[3], kernel_size=3, stride=2, padding=1))
        layers.extend([make_cnn_block(self.planes[3], self.planes[3], kernel_size=3, stride=1, padding=1)
                       for _ in range(self.num_layers[2])])

        return nn.Sequential(*layers)

    def _make_decoder(self):
        layers = []
        layers.append(make_tcnn_block(self.planes[3], self.planes[4], kernel_size=3, stride=2, padding=0))
        layers.extend([make_cnn_block(self.planes[4], self.planes[4], kernel_size=3, stride=1, padding=1)
                       for _ in range(self.num_layers[3])])

        layers.append(make_tcnn_block(self.planes[4], self.planes[5], kernel_size=3, stride=2, padding=0))
        layers.extend([make_cnn_block(self.planes[5], self.planes[5], kernel_size=3, stride=1, padding=1)
                       for _ in range(self.num_layers[4])])

        layers.append(make_tcnn_block(self.planes[5], self.planes[6], kernel_size=3, stride=2, padding=0))
        layers.extend([make_cnn_block(self.planes[6], self.planes[6], kernel_size=3, stride=1, padding=1)
                       for _ in range(self.num_layers[5])])

        return nn.Sequential(*layers)

    def forward_features(self, x):
        return self.enc(x)

    def forward_output(self, x):
        return self.foutput(self.dec(x))

    def forward(self, x):
        v = self.forward_features(x)
        r = self.forward_output(v)

        extra = {'latent': v, 'input': x}
        return r, extra

    def get_loss(self, args, outputs, extra, targets):
        return self.criterion(outputs, extra['input']), {}


def daf_small_bottleneck_cnn_autoenconder(**kwargs):

    # Transition of planes between blocks
    planes = [3, 6, 12, 32, 12, 6, 3]

    return BottleNeckCNNAutoEnconder(
        planes = planes,
        **kwargs
    )


def daf_xl_small_bottleneck_cnn_autoenconder(**kwargs):
    # Transition of planes between blocks
    planes = [3, 6, 12, 32, 12, 6, 3]

    # Additional intermediate layers
    num_layers = [1, 1, 1, 1, 1, 1]

    return BottleNeckCNNAutoEnconder(
        planes=planes,
        num_layers=num_layers,
        **kwargs
    )


def daf_medium_bottleneck_cnn_autoenconder(**kwargs):

    # Transition of planes between blocks
    planes = [3, 12, 24, 64, 24, 12, 3]

    return BottleNeckCNNAutoEnconder(
        planes=planes,
        **kwargs
    )


def daf_xl_medium_bottleneck_cnn_autoenconder(**kwargs):
    # Transition of planes between blocks
    planes = [3, 12, 24, 64, 24, 12, 3]

    # Additional intermediate layers
    num_layers = [1, 1, 1, 1, 1, 1]

    return BottleNeckCNNAutoEnconder(
        planes=planes,
        num_layers=num_layers,
        **kwargs
    )


def daf_large_bottleneck_cnn_autoenconder(**kwargs):

    # Transition of planes between blocks
    planes = [3, 12, 32, 128, 32, 12, 3]

    return BottleNeckCNNAutoEnconder(
        planes=planes,
        **kwargs
    )


def daf_xl_large_bottleneck_cnn_autoenconder(**kwargs):
    # Transition of planes between blocks
    planes = [3, 12, 32, 128, 32, 12, 3]

    # Additional intermediate layers
    num_layers = [1, 1, 1, 1, 1, 1]

    return BottleNeckCNNAutoEnconder(
        planes=planes,
        num_layers=num_layers,
        **kwargs
    )

