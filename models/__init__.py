import torchvision.models as models
from models.densenetccnl import *
from models.unetnc import *


def get_model(name, n_classes=1, filters=64,version=None,in_channels=3, is_batchnorm=True, norm='batch', model_path=None, use_sigmoid=True, layers=3):
    model = _get_model_instance(name)

    if name == 'dnetccnl':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'unetnc':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    else:
        model = model(n_classes=n_classes)
    return model

def _get_model_instance(name):
    try:
        return {
            'dnetccnl': dnetccnl,
            'unetnc': UnetGenerator,
        }[name]
    except:
        print('Model {} not available'.format(name))
