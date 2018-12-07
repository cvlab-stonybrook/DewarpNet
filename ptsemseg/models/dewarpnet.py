import torch.nn as nn

from ptsemseg.models.utils import *
from collections import OrderedDict

class wcunet(nn.Module):
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


class invcoordnet(nn.Module):
    def __init__(self, layers=5, in_classes=1, out_classes=1, in_channels=3, is_batchnorm=True, conv_layer1_filter_size=64,dense_layer_units=100,dense_layers=3, dense_scale=2):
        super(invcoordnet, self).__init__()
        self.BN=is_batchnorm
        self.in_channels=in_classes
        self.out_channels=out_classes
        self.init_conv_filtersize=conv_layer1_filter_size
        self.init_fc_units= dense_layer_units
        self.fc_layers=dense_layers
        self.conv_layers=layers
        self.fc_scale=dense_scale

        filters=[self.init_conv_filtersize*x for x in range(1, layers+1)]

        conv_encoder_layers=OrderedDict()

        for i in range(self.conv_layers):
            if i==0 and BN:
                conv_encoder_layers['conv_'+str(i+1)]=nn.Sequential(nn.Conv2d(self.in_channels, filters[i],kernel_size=5,stride=1,padding=2),
                                                                    nn.Batchnorm2d(filters[i]),
                                                                    nn.ReLU(),
                                                                    nn.MaxPool2d(kernel_size=2))
            elif i!=0 and BN:
                conv_encoder_layers['conv_'+str(i+1)]=nn.Sequential(nn.Conv2d(filters[i-1], filters[i],kernel_size=3,stride=1,padding=1),
                                                                    nn.Batchnorm2d(filters[i]),
                                                                    nn.ReLU(),
                                                                    nn.MaxPool2d(kernel_size=2))
            elif i==0 and not BN:
                conv_encoder_layers['conv_'+str(i+1)]=nn.Sequential(nn.Conv2d(self.in_channels, filters[i],kernel_size=5,stride=1,padding=2),
                                                                    nn.Batchnorm2d(filters[i]),
                                                                    nn.ReLU(),
                                                                    nn.MaxPool2d(kernel_size=2))
            elif i!=0 and not BN:
                conv_encoder_layers['conv_'+str(i+1)]=nn.Sequential(nn.Conv2d(filters[i-1], filters[i],kernel_size=3,stride=1,padding=1),
                                                                    nn.ReLU(),
                                                                    nn.MaxPool2d(kernel_size=2))


        conv_encoder=nn.Sequential(conv_encoder_layers)

        # fc_encoder_layers=OrderedDict()
        # fc_units=[self.init_fc_units/1 for x in range(1, int(math.ceil(float(fc_layers)/2))+1)]

        # for i in range(self.fc_layers):

        conv_decoder_layers=OrderedDict()

        for i in range(self.conv_layers):
            if n-i-1==0:
                conv_decoder_layers['deconv_'+str(i+1)]=nn.Sequential(nn.Conv2d(filters[n-i-1], self.out_channels,kernel_size=2,stride=1,padding=2),
                                                                    nn.ReLU(),
                                                                    nn.ConvTransposed2d(self.out_channels,self.out_channels,kernel_size=2,stride=2))
            else:
                conv_decoder_layers['deconv_'+str(i+1)]=nn.Sequential(nn.Conv2d(filters[n-i-1], filters[n-i-2],kernel_size=3,stride=1,padding=1),
                                                                    nn.ReLU(),
                                                                    nn.ConvTransposed2d(filters[n-i-2],filters[n-i-2],kernel_size=2,stride=2))

        conv_decoder=nn.Sequential(conv_decoder_layers)

    def forward(self,inputs):
        endoded=self.encoder(inputs)
        decoded=self.decoder(encoded)

        return decoded


