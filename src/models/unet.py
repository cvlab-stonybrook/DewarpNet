# cropped skip connection unet

import torch.nn as nn

from src.models.utils import *

class unet(nn.Module):

    def __init__(self, feature_scale=1, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # print('c1:{}'.format(maxpool1.shape))

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print('c2:{}'.format(maxpool2.shape))

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print('c3:{}'.format(maxpool3.shape))

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # print('c4:{}'.format(maxpool4.shape))

        center = self.center(maxpool4)
        # print('center:{}'.format(center.shape))
        up4 = self.up_concat4(conv4, center)
        # print('up4:{}'.format(up4.shape))
        up3 = self.up_concat3(conv3, up4)
        # print('up3:{}'.format(up3.shape))
        up2 = self.up_concat2(conv2, up3)
        # print('up2:{}'.format(up2.shape))
        up1 = self.up_concat1(conv1, up2)
        # print('up1:{}'.format(up1.shape))

        final = self.final(up1)
        # print('fin:{}'.format(final.shape))


        return final
