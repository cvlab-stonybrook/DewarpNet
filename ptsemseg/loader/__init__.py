import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.foldeddoc_loader import foldeddocLoader
from ptsemseg.loader.foldeddocsph_loader import foldeddocsphLoader
from ptsemseg.loader.foldeddocwc_loader import foldeddocwcLoader
from ptsemseg.loader.dewarpnetwc_loader import dewarpnetwcLoader
from ptsemseg.loader.dewarpnetwcg_loader import dewarpnetwcgLoader
from ptsemseg.loader.dewarpnetms_loader import dewarpnetmsLoader
from ptsemseg.loader.dewarpnetdm_loader import dewarpnetdmLoader
from ptsemseg.loader.dewarpnetuv_loader import dewarpnetuvLoader
from ptsemseg.loader.dewarpnetbm_loader import dewarpnetbmLoader
from ptsemseg.loader.dewarpnetbmdm_loader import dewarpnetbmdmLoader
from ptsemseg.loader.dewarpnetbmdmcc_loader import dewarpnetbmdmccLoader
from ptsemseg.loader.dewarpnetbmns_loader import dewarpnetbmnsLoader
from ptsemseg.loader.dewarpnetbmuv_loader import dewarpnetbmuvLoader
from ptsemseg.loader.dewarpnetbmnoimg_loader import dewarpnetbmnoimgLoader
from ptsemseg.loader.dewarpnetbmnoimgcc_loader import dewarpnetbmnoimgccLoader
from ptsemseg.loader.dewarpnete2e_loader import dewarpnete2eLoader



def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'pascal': pascalVOCLoader,
        'camvid': camvidLoader,
        'ade20k': ADE20KLoader,
        'mit_sceneparsing_benchmark': MITSceneParsingBenchmarkLoader,
        'cityscapes': cityscapesLoader,
        'nyuv2': NYUv2Loader,
        'sunrgbd': SUNRGBDLoader,
        'foldeddoc':foldeddocLoader,
        'foldeddocsph':foldeddocsphLoader,
        'foldeddocwc':foldeddocwcLoader,
        'dewarpnetwc':dewarpnetwcLoader,
        'dewarpnetwcg':dewarpnetwcgLoader,
        'dewarpnetms':dewarpnetmsLoader,
        'dewarpnetdm':dewarpnetdmLoader,
        'dewarpnetuv':dewarpnetuvLoader,
        'dewarpnetbm':dewarpnetbmLoader,
        'dewarpnetbmdm':dewarpnetbmdmLoader,
        'dewarpnetbmdmcc':dewarpnetbmdmccLoader,
        'dewarpnetbmuv':dewarpnetbmuvLoader,
        'dewarpnetbmns':dewarpnetbmnsLoader,
        'dewarpnetbmni':dewarpnetbmnoimgLoader,
        'dewarpnetbmnicc':dewarpnetbmnoimgccLoader,
        'dewarpnete2e':dewarpnete2eLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
