import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.icnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.frrn import *
from ptsemseg.models.densenet import *
from ptsemseg.models.densenetns import *
from ptsemseg.models.densenet_ import *
from ptsemseg.models.unetnc import *



def get_model(name, n_classes, version=None,in_channels=3,is_batchnorm=True):
    model = _get_model_instance(name)

    if name in ['frrnA', 'frrnB']:
        model = model(n_classes, model_type=name[-1])

    elif name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'unet':
        model = model(n_classes=n_classes,
                      is_batchnorm=is_batchnorm,
                      in_channels=in_channels,
                      is_deconv=True)

    elif name == 'pspnet':
        model = model(n_classes=n_classes, version=version)

    elif name == 'icnet':
        model = model(n_classes=n_classes, with_bn=False, version=version)
    elif name == 'icnetBN':
        model = model(n_classes=n_classes, with_bn=True, version=version)
    elif name == 'dnet':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'dneto':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'dnetns':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'unetnc':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)

    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'fcn32s': fcn32s,
            'fcn8s': fcn8s,
            'fcn16s': fcn16s,
            'unet': unet,
            'segnet': segnet,
            'pspnet': pspnet,
			'icnet': icnet,
			'icnetBN': icnet,
            'linknet': linknet,
            'frrnA': frrn,
            'frrnB': frrn,
            'dnet': dnet,
            'dnetns': dnetns,
            'dneto': dneto,
            'unetnc': UnetGenerator,
        }[name]
    except:
        print('Model {} not available'.format(name))
