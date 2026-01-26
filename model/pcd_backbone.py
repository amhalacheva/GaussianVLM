import os

import torch
from accelerate.logging import get_logger
from einops import rearrange
from hydra.utils import instantiate
from torch import nn

# from model.pointnetpp.pointnetpp import PointNetPP
# from model.pointnext.pointnext import PointNext
# from model.pointbert.pointbert import PointBERT
from model.utils import disabled_train

# from model.scenesplat.pointcept.models import build_model as build_model_pointcept

# import os
# from model.scenesplat.pointcept.utils.config import Config, DictAction
# from model.scenesplat.pointcept.engines.defaults import create_ddp_model
from model.scenesplat.model_build import create_pointcept

logger = get_logger(__name__)


class PointcloudBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.net._target_ == "SceneSplat":
            self.pcd_net = create_pointcept(cfg)
            checkpoint = torch.load(cfg.path)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            is_ddp_model = type(self.pcd_net).__name__.startswith('DistributedDataParallel')
            state_dict_has_module = any(k.startswith('module.') for k in state_dict.keys())
            
            if is_ddp_model and not state_dict_has_module:
                # Add 'module.' prefix
                new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                self.pcd_net.load_state_dict(new_state_dict)
            elif not is_ddp_model and state_dict_has_module:
                # Remove 'module.' prefix
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.pcd_net.load_state_dict(new_state_dict)
            else:
                # No adjustment needed
                self.pcd_net.load_state_dict(state_dict)

            self.backbone_name = cfg.net._target_.split('.')[-1]
            self.out_dim = 768
            logger.info(f"Build PointcloudBackbone: {self.backbone_name}")
        else:
            self.pcd_net = instantiate(cfg.net)
            self.backbone_name = cfg.net._target_.split('.')[-1]
            self.out_dim = self.pcd_net.out_dim
            logger.info(f"Build PointcloudBackbone: {self.backbone_name}")

            path = cfg.path
            if path is not None and os.path.exists(path):
                self.pcd_net.load_state_dict(torch.load(path), strict=False)
                logger.info(f"Load {self.backbone_name} weights from {path}")

        self.freeze = cfg.freeze
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
            self.train = disabled_train
            logger.info(f"Freeze {self.backbone_name}")

    def forward_normal(self, obj_pcds):
        # obj_pcds: (batch_size, num_objs, num_points, 6)
        batch_size = obj_pcds.shape[0]
        obj_embeds = self.pcd_net(
            rearrange(obj_pcds, 'b o p d -> (b o) p d')
        )
        obj_embeds = rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        return obj_embeds

    def forward_scenesplat(self, obj_pcds, scene_id):
        # obj_pcds: (batch_size, num_objs, num_points, 6)
        obj_embeds = self.pcd_net(obj_pcds)
        return obj_embeds

    @torch.no_grad()
    def forward_frozen(self, obj_pcds, scene_id=None):
        if isinstance(obj_pcds, dict):
            return self.forward_scenesplat(obj_pcds, scene_id)
        else:
            return self.forward_normal(obj_pcds)

    def forward(self, obj_pcds, scene_id=None):
        if self.freeze:
            return self.forward_frozen(obj_pcds, scene_id)
        else:
            return self.forward_normal(obj_pcds, scene_id)

    # ########################################################
    # ########################################################
    # #### SceneSplat config and model builders
    # ########################################################
    
    # def scenesplat_config_parser(file_path, options):
    #     # config name protocol: dataset_name/model_name-exp_name
    #     if os.path.isfile(file_path):
    #         cfg = Config.fromfile(file_path)
    #     else:
    #         sep = file_path.find("-")
    #         cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))
        
    #     if options is not None:
    #         cfg.merge_from_dict(options)
    
    #     if cfg.seed is None:
    #         cfg.seed = get_random_seed()
    
    #     cfg.data.train.loop = cfg.epoch // cfg.eval_epoch
    
    #     os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    #     if not cfg.resume:
    #         cfg.dump(os.path.join(cfg.save_path, "config.py"))
    #     return cfg

