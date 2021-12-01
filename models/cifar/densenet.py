import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [
    'densenet',
    'baseline10densenet',
    'baseline100densenet',
    'baselineda10densenet',
    'baselineda100densenet',
]


class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()

        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropRate)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)

        out = torch.cat((x, out), 1)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()

        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropRate)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)

        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(
            self,
            depth = 22,
            block = Bottleneck,
            drop_rate = 0.0,
            num_classes = 10,
            growth_rate = 12,
            compression_rate = 2,
            **kwargs
    ):
        super(DenseNet, self).__init__()

        # Check variable
        assert (depth - 4) % 3 == 0, 'Depth should be of form 3n + 4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction = 'mean')

        self.depth = depth
        self.block = block,
        self.growthRate = growth_rate
        self.dropRate = drop_rate
        self.compression_rate = compression_rate
        self.kwargs = kwargs

        # self.inplanes is a global variable used across multiple helper functions
        self.inplanes = growth_rate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compression_rate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compression_rate)
        self.dense3 = self._make_denseblock(block, n)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        self.num_classes = num_classes
        self.fc = nn.Linear(self.inplanes, num_classes)

        self._initialise()

    def reinitialise(self):
        return DenseNet(
            depth = self.depth,
            block = self.block,
            dropRate = self.dropRate,
            num_classes = self.num_classes,
            growth_rate = self.growth_rate,
            compression_rate = self.compression_rate,
            **self.kwargs
        )

    def _initialise(self):
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward_features(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_output(self, x):
        return self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_output(x)

        extra = {}
        return x, extra

    def get_loss(self, args, outputs, extra, targets):
        extra = {}
        return self.criterion(outputs, targets), extra


class SecondHeadDenseNet(DenseNet):
    def __init__(
            self,
            depth=22,
            block=Bottleneck,
            drop_rate=0,
            num_classes=10,
            growth_rate=12,
            compression_rate=2,
            **kwargs
    ):
        super(SecondHeadDenseNet, self).__init__(
            depth=depth,
            block=block,
            drop_rate=drop_rate,
            num_classes=num_classes,
            growth_rate=growth_rate,
            compression_rate=compression_rate,
            **kwargs
        )

        # Define additional layer for predicting diagonal scale
        self.fcg = nn.Linear(self.inplanes, num_classes)

        # Reinitialise layers
        self._initialise()

    def forward(self, x):
        v = self.forward_features(x)

        # Get head outputs
        x = (self.fc(v), self.fcg(v))

        extra = {'hidden_layer_feature': v}
        return x, extra

    def get_loss(self, args, outputs, extra, targets):
        raise NotImplementedError


def densenet(**kwargs):
    """
    Constructs a DenseNet model.
    """
    return DenseNet(**kwargs)


def baseline10densenet(**kwargs):
    return DenseNet(
        depth = 100,
        block = Bottleneck,
        drop_rate = 0.20,
        num_classes = 10,
        growth_rate = 12,
        compression_rate = 2,
    )


def baseline100densenet(**kwargs):
    return DenseNet(
        depth = 100,
        block = Bottleneck,
        drop_rate = 0.20,
        num_classes = 100,
        growth_rate = 12,
        compression_rate = 2,
    )


def baselineda10densenet(**kwargs):
    return DenseNet(
        depth = 100,
        block = Bottleneck,
        drop_rate = 0.00,
        num_classes = 10,
        growth_rate = 12,
        compression_rate = 2,
    )


def baselineda100densenet(**kwargs):
    return DenseNet(
        depth = 100,
        block = Bottleneck,
        drop_rate = 0.00,
        num_classes = 100,
        growth_rate = 12,
        compression_rate = 2,
    )