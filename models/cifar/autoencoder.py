import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [

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
    def __init__(self, **kwargs):
        super(BottleNeckCNNAutoEnconder, self).__init__()

        # Loss function
        self.criterion = nn.MSELoss(reduction='mean')

        # Create encoder-decoder structure
        self.enc = self._make_encoder()
        self.dec = self._make_decoder()

        # Output function
        self.foutput = nn.Tanh()

    def _make_encoder(self):
        enc_block1 = make_cnn_block(3, 12, kernel_size=3, stride=2, padding=1)
        enc_block2 = make_cnn_block(12, 24, kernel_size=3, stride=2, padding=1)
        enc_block3 = make_cnn_block(24, 128, kernel_size=3, stride=2, padding=1)
        return nn.Sequential(enc_block1, enc_block2, enc_block3)

    def _make_decoder(self):
        dec_block1 = make_tcnn_block(128, 24, kernel_size=3, stride=2, padding=0)
        dec_block2 = make_tcnn_block(24, 12, kernel_size=3, stride=2, padding=0)
        dec_block3 = make_tcnn_block(12, 3, kernel_size=3, stride=2, padding=0)
        return nn.Sequential(dec_block1, dec_block2, dec_block3)

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

