from .TRMfusion import MultiModalFusionModel
from .CAMERAfusion import CAMERAfusion
from .CAMERAMOE import CAMERAMOE

MODELS = {
    'MultiModalFusionModel':MultiModalFusionModel,
    'CAMERAfusion':CAMERAfusion,
    'CAMERAMOE':CAMERAMOE,
    }

def build_model(cfg):
    model = MODELS[cfg.model['name']]
    model_kwargs = cfg.model['kwargs'] if cfg.model['kwargs'] else {}
    return model(cfg=cfg,**model_kwargs)