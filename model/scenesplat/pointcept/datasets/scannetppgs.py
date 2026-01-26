"""
ScanNet++ dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ScanNetPPGSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "segment",
        "segment200", # temp
        "instance",
        "quat",
        "scale",
        "opacity",
        "lang_feat",
        "valid_feat_mask",
        "normal",
    ]

    # denoting tailed classes, below 0.01% of total vertices
    TAIL_CLASSES = np.array(
        [82, 74, 84, 76, 86, 80, 85, 92, 88, 83, 93, 87, 94, 89, 55, 90, 96, 97, 91, 95, 98]
    )

    def __init__(
        self,
        multilabel=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)
            # print("color", data_dict["color"].shape)

        if "opacity" in data_dict.keys():
            data_dict["opacity"] = data_dict["opacity"].astype(np.float32)
            data_dict["opacity"] = data_dict["opacity"].reshape(-1, 1)

        if "quat" in data_dict.keys():
            data_dict["quat"] = data_dict["quat"].astype(np.float32)

        if "sh" in data_dict.keys():
            data_dict["sh"] = data_dict["sh"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "scale" in data_dict.keys():
            data_dict["scale"] = data_dict["scale"].astype(np.float32).clip(0, 1.5) # clip scale max to 1.5
        
        if "lang_feat" in data_dict.keys():
            data_dict["lang_feat"] = data_dict["lang_feat"].astype(np.float16)
        
        if "valid_feat_mask" in data_dict.keys():
            data_dict["valid_feat_mask"] = data_dict["valid_feat_mask"].astype(bool)

        if not self.multilabel:
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][:, 0].astype(np.int32)
            elif "segment200" in data_dict.keys(): # temp update
                data_dict["segment"] = data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            else:
                data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )

            if "instance" in data_dict.keys():
                try:
                    data_dict["instance"] = data_dict["instance"][:, 0].astype(np.int32)
                except:
                    data_dict["instance"] = data_dict.pop("instance").reshape([-1]).astype(np.int32)
            else:
                data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )
        else:
            raise NotImplementedError
        
        if self.sample_tail:
            # use data_dict["sampled_index"] to denote tail classes are sampled
            tail_mask = np.isin(data_dict["segment"], self.TAIL_CLASSES)
            data_dict["sampled_index"] = np.where(tail_mask)[0]
        return data_dict
