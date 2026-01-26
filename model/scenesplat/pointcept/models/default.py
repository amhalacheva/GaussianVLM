import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        print("backbone", backbone)
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(seg_logits=seg_logits, loss=loss)
        # test
        else:
            #return dict(seg_logits=seg_logits)
            return {"point_feat":{"hidden_states":point["hidden_states"]}}
        
@MODELS.register_module()
class LangPretrainer(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.training=False # TODO: changed

    def forward(self, input_dict, scene_id, chunk_size=None):
        
        if chunk_size is not None and chunk_size > 0 and input_dict["coord"].shape[0] > chunk_size:
            return self._chunked_forward(input_dict, chunk_size)
        point = Point(input_dict)
        point_feat = self.backbone(point, scene_id)
            
        # normalize the feature
        try:
            point_feat['feat'] = nn.functional.normalize(point_feat['feat'], p=2, dim=1)
        except:
            return point_feat
    
        # train
        if self.training:
            segment = input_dict["segment"] if "segment" in input_dict.keys() else None
            loss = self.criteria(point_feat['feat'], input_dict["lang_feat"], 
                                 valid_feat_mask=input_dict["valid_feat_mask"],
                                 segment=segment, epoch_progress=input_dict["epoch_progress"])
            return dict(loss=loss)
        # test
        else:
            return dict(point_feat=point_feat)

    def _chunked_forward(self, input_dict, chunk_size):
        """
        Break the large point set into smaller chunks, pass each chunk through backbone,
        and concat the output features.
        NOTE: This only works if your model's global context isn't critical across chunks.
        """

        # We'll assume "coord" (Nx3 or NxD) is the main key to figure out total #points N.
        # Modify if your data structure is different.
        coords = input_dict["coord"]
        N = coords.shape[0]

        # Prepare a list to store chunk outputs
        chunk_outputs = []

        # We'll do the same logic as normal forward, but inside a loop
        # that processes chunk by chunk.
        is_training = self.training  # track if we are in training or eval

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            # split input_dict into chunks
            chunk_input_dict = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == N:
                    chunk_input_dict[k] = v[start_idx:end_idx]
            # need to address the 'offset' key separately, which is the same as N
            chunk_input_dict["offset"] = torch.tensor([end_idx - start_idx], device=coords.device)
            chunk_point = Point(chunk_input_dict)

            chunk_point_feat = self.backbone(chunk_point)
            chunk_point_feat["feat"] = nn.functional.normalize(
                chunk_point_feat["feat"], p=2, dim=1
            )

            if is_training:
                segment = chunk_input_dict.get("segment", None)
                loss = self.criteria(
                    chunk_point_feat["feat"],
                    chunk_input_dict["lang_feat"],
                    valid_feat_mask=chunk_input_dict["valid_feat_mask"],
                    segment=segment,
                    epoch_progress=chunk_input_dict.get("epoch_progress", None)
                )
                chunk_outputs.append(loss)
            else:
                # If eval, store chunk feats to concat
                chunk_outputs.append(chunk_point_feat["feat"])

        if is_training:
            # sum or average the chunk losses
            # e.g., total_loss = sum(chunk_outputs) / len(chunk_outputs)
            total_loss = torch.stack(chunk_outputs).mean()
            return dict(loss=total_loss)
        else:
            full_feat = torch.cat(chunk_outputs, dim=0)  # shape [N, C]
            return dict(point_feat={"feat": full_feat})


@MODELS.register_module()
class DefaultSegmentorSkip(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        # (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)



@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)

@MODELS.register_module()
class DefaultPretrainer(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        # self.seg_head = (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        # seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(feat, input_dict["clip_feat"])
            return dict(loss=loss)
        # eval
        elif "clip_feat" in input_dict.keys():
            loss = self.criteria(feat, input_dict["clip_feat"])
            return dict(loss=loss, seg_logits=feat)
        # test
        else:
            return dict(seg_logits=feat)



@MODELS.register_module()
class DefaultContrastiver(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        # self.seg_head = (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        # split the global and local augmentation feature here
        point = Point(input_dict[''])
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        global_feat = feat.mean(dim=1) # BxNxC -> BxC


        if self.training:
            loss = self.criteria(feat, input_dict["clip_feat"])
            return dict(loss=loss)
        # eval
        elif "clip_feat" in input_dict.keys():
            loss = self.criteria(feat, input_dict["clip_feat"])
            return dict(loss=loss, seg_logits=feat)
        # test
        else:
            return dict(seg_logits=feat)
