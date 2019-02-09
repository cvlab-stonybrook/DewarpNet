import json

from src.loader.swat3dwcg_loader import swat3dwcgLoader
from src.loader.swat3dwc_loader import swat3dwcLoader
from src.loader.swat3dbmnoimg_loader import swat3dbmnoimgLoader
from src.loader.swat3dlapsrn_loader import swat3dlapsrnLoader
from src.loader.swat3de2e_loader import swat3de2eLoader
from src.loader.swat3dbmnoimgd_loader import swat3dbmnoimgdLoader



def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'swat3dwcg':swat3dwcgLoader,
        'swat3dwc':swat3dwcLoader,
        'swat3dbmni':swat3dbmnoimgLoader,
        'swat3dbmnid':swat3dbmnoimgdLoader,
        'swat3dlapsrn':swat3dlapsrnLoader,
        'swat3de2e':swat3de2eLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
