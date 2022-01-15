import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from src.utils.modelutils import spectral_norm_fc, spectral_norm_conv


class WideBasic(nn.Module):
    def __init__(
        self,
        wrapped_conv,
        input_size,
        in_c,
        out_c,
        stride,
        dropout_rate,
        mod=True,
        batchnorm_momentum=0.01,
    ):
        super().__init__()

        self.mod = mod
        self.bn1 = nn.BatchNorm2d(in_c, momentum=batchnorm_momentum)
        self.conv1 = wrapped_conv(input_size, in_c, out_c, 3, stride)

        self.bn2 = nn.BatchNorm2d(out_c, momentum=batchnorm_momentum)
        self.conv2 = wrapped_conv(math.ceil(input_size / stride), out_c, out_c, 3, 1)
        self.activation = F.leaky_relu if self.mod else F.relu

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        if stride != 1 or in_c != out_c:
            if mod:

                def shortcut(x):
                    x = F.avg_pool2d(x, stride, stride)
                    pad = torch.zeros(
                        x.shape[0],
                        out_c - in_c,
                        x.shape[2],
                        x.shape[3],
                        device=x.device,
                    )
                    x = torch.cat((x, pad), dim=1)
                    return x

                self.shortcut = shortcut
            else:
                # Just use a strided conv
                self.shortcut = wrapped_conv(input_size, in_c, out_c, 1, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.activation(self.bn1(x))
        out = self.conv1(out)
        out = self.activation(self.bn2(out))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        spectral_normalization=True,
        mod=True,
        depth=28,
        widen_factor=10,
        num_classes=None,
        dropout_rate=0.3,
        coeff=3,
        n_power_iterations=1,
        batchnorm_momentum=0.01,
        temp=1.0,
        **kwargs
    ):
        """
        If the "mod" parameter is set to True, the architecture uses 2 modifications:
        1. LeakyReLU instead of normal ReLU
        2. Average Pooling on the residual connections.
        """
        super().__init__()

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"

        self.dropout_rate = dropout_rate
        self.mod = mod

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, shapes, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        strides = [1, 1, 2, 2]
        input_sizes = 32 // np.cumprod(strides)

        self.conv1 = wrapped_conv(input_sizes[0], 3, nStages[0], 3, strides[0])
        self.layer1 = self._wide_layer(nStages[0:2], n, strides[1], input_sizes[0])
        self.layer2 = self._wide_layer(nStages[1:3], n, strides[2], input_sizes[1])
        self.layer3 = self._wide_layer(nStages[2:4], n, strides[3], input_sizes[2])

        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=batchnorm_momentum)
        self.activation = F.leaky_relu if self.mod else F.relu

        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(nStages[3], num_classes)

        nonlinearity = "leaky_relu" if self.mod else "relu"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L17
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=nonlinearity
                )
            elif isinstance(m, nn.Linear):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L21
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=nonlinearity
                )
                nn.init.constant_(m.bias, 0)
        self.feature = None
        self.temp = temp

    def _wide_layer(self, channels, num_blocks, stride, input_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        in_c, out_c = channels

        for stride in strides:
            layers.append(
                WideBasic(
                    self.wrapped_conv,
                    input_size,
                    in_c,
                    out_c,
                    stride,
                    self.dropout_rate,
                    self.mod,
                )
            )
            in_c = out_c
            input_size = math.ceil(input_size / stride)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.flatten(1)
        self.feature = out.clone().detach()

        if self.num_classes is not None:
            out = self.linear(out) / self.temp
        return out


## VGG
"""
Pytorch implementation of VGG models.
Reference:
[1] . Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.
"""


cfg_cifar = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

inp_size_cifar = {
    "VGG11": [32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2, 1],
    "VGG13": [32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2, 1],
    "VGG16": [32, 32, 16, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 1],
    "VGG19": [32, 32, 16, 16, 16, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1],
}

cfg_mnist = {
    "VGG11": [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

inp_size_mnist = {
    "VGG11": [28, 28, 14, 14, 14, 7, 7, 7, 3, 3, 3, 1],
    "VGG13": [28, 28, 28, 28, 14, 14, 14, 7, 7, 7, 3, 3, 3, 1],
    "VGG16": [28, 28, 28, 28, 14, 14, 14, 14, 7, 7, 7, 7, 3, 3, 3, 3, 1],
    "VGG19": [28, 28, 28, 28, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 1],
}


class VGG(nn.Module):
    def __init__(
        self,
        vgg_name,
        num_classes=10,
        temp=1.0,
        spectral_normalization=True,
        mod=True,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
    ):
        """
        If the "mod" parameter is set to True, the architecture uses 2 modifications:
        1. LeakyReLU instead of normal ReLU
        2. Average Pooling on the residual connections.
        """
        super(VGG, self).__init__()
        self.temp = temp
        self.mod = mod

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, shapes, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        self.mnist = mnist
        if mnist:
            self.inp_sizes = inp_size_mnist[vgg_name]
            self.features = self._make_layers(cfg_mnist[vgg_name])
        else:
            self.inp_sizes = inp_size_cifar[vgg_name]
            self.features = self._make_layers(cfg_cifar[vgg_name])

        self.classifier = nn.Linear(512, num_classes)
        self.feature = None

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        self.feature = out.clone().detach()
        out = self.classifier(out) / self.temp
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1 if self.mnist else 3
        for i, x in enumerate(cfg):
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    self.wrapped_conv(
                        self.inp_sizes[i], in_channels, x, kernel_size=3, stride=1
                    ),
                    nn.BatchNorm2d(x),
                    nn.LeakyReLU(inplace=True) if self.mod else nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


## Resnet

"""
Pytorch implementation of ResNet models.
Reference:
[1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR, 2016.
"""


class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(
            x.shape[0],
            self.out_c - self.in_c,
            x.shape[2],
            x.shape[3],
            device=x.device,
        )
        x = torch.cat((x, pad), dim=1)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True):
        super(BasicBlock, self).__init__()
        self.conv1 = wrapped_conv(
            input_size, in_planes, planes, kernel_size=3, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(
            math.ceil(input_size / stride), planes, planes, kernel_size=3, stride=1
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if mod:
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(stride, self.expansion * planes, in_planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    wrapped_conv(
                        input_size,
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True):
        super(Bottleneck, self).__init__()
        self.conv1 = wrapped_conv(
            input_size, in_planes, planes, kernel_size=1, stride=1
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(
            input_size, planes, planes, kernel_size=3, stride=stride
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = wrapped_conv(
            math.ceil(input_size / stride),
            planes,
            self.expansion * planes,
            kernel_size=1,
            stride=1,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if mod:
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(stride, self.expansion * planes, in_planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    wrapped_conv(
                        input_size,
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


## RESNET


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        temp=1.0,
        spectral_normalization=True,
        mod=True,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
    ):
        """
        If the "mod" parameter is set to True, the architecture uses 2 modifications:
        1. LeakyReLU instead of normal ReLU
        2. Average Pooling on the residual connections.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.mod = mod

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, shapes, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        self.bn1 = nn.BatchNorm2d(64)

        if mnist:
            self.conv1 = wrapped_conv(28, 1, 64, kernel_size=3, stride=1)
            self.layer1 = self._make_layer(block, 28, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 28, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 14, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 7, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = wrapped_conv(32, 3, 64, kernel_size=3, stride=1)
            self.layer1 = self._make_layer(block, 32, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 16, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 8, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.activation = F.leaky_relu if self.mod else F.relu
        self.feature = None
        self.temp = temp

    def _make_layer(self, block, input_size, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    input_size,
                    self.wrapped_conv,
                    self.in_planes,
                    planes,
                    stride,
                    self.mod,
                )
            )
            self.in_planes = planes * block.expansion
            input_size = math.ceil(input_size / stride)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.feature = out.clone().detach()
        out = self.fc(out) / self.temp
        return out


## LENET

"""Implementation of Lenet in pytorch.
Refernece:
[1] LeCun,  Y.,  Bottou,  L.,  Bengio,  Y.,  & Haffner,  P. (1998).
    Gradient-based  learning  applied  to  document  recognition.
    Proceedings of the IEEE, 86, 2278-2324.
"""


class LeNet(nn.Module):
    def __init__(self, num_classes, temp=1.0, mnist=True, **kwargs):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1 if mnist else 3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.temp = temp
        self.feature = None

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        self.feature = out
        out = self.fc3(out) / self.temp
        return out


def lenet(num_classes=10, temp=1.0, mnist=True, **kwargs):
    return LeNet(num_classes=num_classes, temp=temp, mnist=True, **kwargs)


def resnet18(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def resnet50(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def resnet101(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def resnet110(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 26, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def resnet152(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def vgg11(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = VGG(
        "VGG11",
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def vgg13(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = VGG(
        "VGG13",
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def vgg16(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = VGG(
        "VGG16",
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def vgg19(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = VGG(
        "VGG19",
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model


def wrn(temp=1.0, spectral_normalization=True, mod=True, **kwargs):
    model = WideResNet(
        spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs
    )
    return model


# These should suffice
model_mapper = {"wrn": wrn, "vgg16": vgg16, "resnet18": resnet18}
