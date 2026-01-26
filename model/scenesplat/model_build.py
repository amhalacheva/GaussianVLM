from .pointcept.models import build_model as build_model_pointcept

import os
from .pointcept.utils.config import Config, DictAction
from .pointcept.engines.defaults import create_ddp_model
import torch

def default_config_parser(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))
    
    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg

def create_pointcept(cfg):
    config = default_config_parser(cfg.config_path,{"grid_size":0.01})
    model = build_model_pointcept(config.model)
    if config.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"Model: \n{self.model}")
    # print torch cuda version
    print(f"torch.cuda version: {torch.version.cuda}")
    # check cuda if is available
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available!")
    else:
        print("CUDA is available!")

    # check if multi-gpu is available
    # model = create_ddp_model(
    #     model.cuda(),
    #     broadcast_buffers=False,
    #     find_unused_parameters=config.find_unused_parameters,
    # )
    return model