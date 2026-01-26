import torch

from pointcept.models import build_model as build_model_pointcept

import os
from pointcept.utils.config import Config, DictAction
from pointcept.engines.defaults import create_ddp_model
from knn_sparsify import Group


########################################################
########################################################
#### Config and model builders
########################################################

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


def build_model(cfg):
        model = build_model_pointcept(cfg.model)
        if cfg.sync_bn:
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
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=cfg.find_unused_parameters,
        )
        return model

# model_cfg = dict(
#     type="LangPretrainer",
#     backbone=dict(
#         type="PT-v3m1",
#         in_channels=6,  # gaussian: color 3, quaternion 4, scale 3, opacity 1, normal 3
#         order=("z", "z-trans", "hilbert", "hilbert-trans"),
#         stride=(2, 2, 2),
#         enc_depths=(2, 2, 2, 6),
#         enc_channels=(32, 64, 128, 256), # -> this direction
#         enc_num_head=(2, 4, 8, 16),
#         enc_patch_size=(1024, 1024, 1024, 1024),
#         dec_depths=(2, 2, 2),
#         dec_channels=(768, 512, 256), # <- this direction
#         dec_num_head=(16, 16, 16),
#         dec_patch_size=(1024, 1024, 1024),
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.3,
#         shuffle_orders=True,
#         pre_norm=True,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=False,
#         upcast_softmax=False,
#         cls_mode=False,
#         pdnorm_bn=False,
#         pdnorm_ln=False,
#         pdnorm_decouple=True,
#         pdnorm_adaptive=False,
#         pdnorm_affine=True,
#         pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
#     ),
#     criteria=[
#         dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
#         dict(type="L2Loss", reduction="mean", loss_weight=1.0),
#         dict(type="AggregatedContrastiveLoss", temperature=0.2, reduction="mean", loss_weight=0.01, schedule='last_75'),
#     ],
# )



########################################################
########################################################
#### Build model and load weights from checkpoint
########################################################
model=build_model(default_config_parser("lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive/config_inference.py",{"grid_size":0.01}))


checkpoint = torch.load("lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive/model_best_lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive.pth")
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

is_ddp_model = type(model).__name__.startswith('DistributedDataParallel')
state_dict_has_module = any(k.startswith('module.') for k in state_dict.keys())

if is_ddp_model and not state_dict_has_module:
    # Add 'module.' prefix
    new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
elif not is_ddp_model and state_dict_has_module:
    # Remove 'module.' prefix
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
else:
    # No adjustment needed
    model.load_state_dict(state_dict)


################ Outcome: model is "LangPretrainer" with "PointTransformerV3" backbone [model.backbone]



########################################################
########################################################
#### Load data and do a forward pass
########################################################
from torch.utils.data import Dataset
import numpy as np

class ScanNetDataset(Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "quat",
        "scale",
        "opacity",
    ]

    def __init__(
        self,
        **kwargs,
    ):
        self.data_list = self.get_data_list()
        super().__init__(**kwargs)

    def get_data_list(self):
        data_list = [
                os.path.join("../../scan_data/3rscan_3dgs_mcmc_depth_true/grid1mm_chunk6x6_stride3x3", name) for name in os.listdir("../../scan_data/3rscan_3dgs_mcmc_depth_true/grid1mm_chunk6x6_stride3x3") if ('.' not in name and 'chunk' not in name and 'grid' not in name)
            ]
        #data_list[0] = os.path.join("../../scan_data/train/", os.listdir("../../scan_data/train/")[0]) 
        return data_list

    def get_data_name(self, idx):
        return self.data_list[idx % len(self.data_list)].split("/")[-1]

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        
        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["grid_size"]=torch.tensor([0.01 for i in range(len(data_dict["coord"]))])
        data_dict["coord"] = torch.tensor(data_dict["coord"].astype(np.float32))
        data_dict["color"] = torch.tensor(data_dict["color"].astype(np.float32))
        data_dict["normal"] = torch.tensor(data_dict["normal"].astype(np.float32))
        data_dict["quat"] = torch.tensor(data_dict["quat"].astype(np.float32))
        data_dict["scale"] = torch.tensor(data_dict["scale"].astype(np.float32))
        data_dict["opacity"] = torch.tensor(data_dict["opacity"].astype(np.float32))

        ##### 'color', 'opacity', 'quat', 'scale', 'normal' according to "feat_keys" in config in lang-pretrain-ppv2..
        data_dict["opacity"] = data_dict["opacity"].unsqueeze(1) if data_dict["opacity"].dim() == 1 else data_dict["opacity"]

        merged = torch.cat([
            data_dict["color"],
            data_dict["opacity"],
            data_dict["quat"],
            data_dict["scale"],
            data_dict["normal"],
        ], dim=-1)  # Final shape: [N, 14]
        # Optional: assign to a key
        data_dict["feat"] = merged.to("cuda")
        return {"feat": merged.to("cuda"), "grid_size":data_dict["grid_size"], "coord":data_dict["coord"]}

ds = ScanNetDataset()
d = ds.get_data(1)
# train_loader = torch.utils.data.DataLoader(
#             ds,
#             batch_size=1,
#             shuffle=True,
#             num_workers=4,
#             drop_last=True,
#             persistent_workers=True,
#         )
# data_iterator = enumerate(train_loader)
# for d in data_iterator:
#     print(d)
#     out_dict = model(d, chunk_size=600000)
out_dict = model(d, chunk_size=600000)
print(out_dict["point_feat"].keys())
print(out_dict["point_feat"]["coord"].shape)
print(len(out_dict["point_feat"]["hidden_states"]))
for h in out_dict["point_feat"]["hidden_states"]:
    print("Hidden dimensions")
    print(h.shape)


########################################################
########################################################
#### Sparsify features, Layers 1-2: Attention Pooling
########################################################
import torch
import torch.nn as nn

class LearnedAttentionPooling(nn.Module):
    def __init__(self, num_queries, feature_dim):
        super(LearnedAttentionPooling, self).__init__()
        # Learnable query tokens - shape [num_queries, feature_dim]
        self.query_tokens = nn.Parameter(torch.randn(num_queries, feature_dim, device="cuda"))
        
    def forward(self, dense_features):
        # Dense features (K, V): [B, num_features, feature_dim]
        # Learnable queries (Q): [num_queries, feature_dim]
        
        B, num_features, _ = dense_features.shape
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # Shape: [B, num_queries, feature_dim]
        
        # Q * K^T, softmax
        attention_scores = torch.bmm(queries, dense_features.transpose(1, 2))  # Shape: [B, num_queries, num_features]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Shape: [B, num_queries, num_features]
        
        # Mult with V
        output = torch.bmm(attention_weights, dense_features)  # Shape: [B, num_queries, feature_dim]
        
        return output

l1_shape = out_dict["point_feat"]["hidden_states"][0].shape
att_pool_1 = LearnedAttentionPooling(128, l1_shape[-1])
print(att_pool_1(out_dict["point_feat"]["hidden_states"][0].unsqueeze(0).to("cuda")).shape)



########################################################
########################################################
#### Sparsify features, Layer 3: kNN
########################################################
pointcloud_tokenizer = Group(num_group=128, group_size=3900)

print(out_dict["point_feat"]["coord"].unsqueeze(0).shape)
print(out_dict["point_feat"]["feat"].unsqueeze(0).shape)
scene_pointclouds = torch.cat([out_dict["point_feat"]["coord"].unsqueeze(0).to("cuda"), out_dict["point_feat"]["feat"].unsqueeze(0).to("cuda")], dim=-1).contiguous()

# Add xyz only for grouping
print(scene_pointclouds.shape)
scene_neighborhood, scene_center = pointcloud_tokenizer(scene_pointclouds, normalize=True)
## Drop xyz
neighborhood_xyz = scene_neighborhood[... , :3]
scene_fts = scene_neighborhood[... , 3:].mean(-2)
print(scene_fts.shape)