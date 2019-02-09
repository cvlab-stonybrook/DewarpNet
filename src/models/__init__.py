import torchvision.models as models

from src.models.densenet import *
from src.models.densenetns import *
from src.models.densenet_ import *
from src.models.densenetcc import *
from src.models.densenetccnl import *
from src.models.unetnc import *
from src.models.hourglass import *
from src.models.hourglass_cat import *
from src.models.lapsrn import *



def get_model(name, n_classes=1, version=None,in_channels=3,is_batchnorm=True, model_path=None):
    model = _get_model_instance(name)

    if name == 'dnet':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'dneto':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'dnetcc':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'dnetccnl':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'dnetnbn':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'unetnc':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'hourglass':
        model = model(input_nc=in_channels, output_nc=n_classes, module1_model_path=model_path)
    elif name == 'hourglass_cat':
        model = model(input_nc=in_channels, output_nc=n_classes, module1_model_path=model_path)
    elif name == 'lapsrn':
        model = model()

    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'dnet': dnet,
            'dnetns': dnetns,
            'dneto': dneto,
            'dnetcc': dnetcc,
            'dnetccnl': dnetccnl,
            'dnetnbn': dnetnBN,
            'unetnc': UnetGenerator,
            'hourglass': Hourglass,
            'hourglass_cat': HourglassC,
            'lapsrn': LapSRN,
        }[name]
    except:
        print('Model {} not available'.format(name))
