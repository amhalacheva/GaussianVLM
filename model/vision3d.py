import numpy as np
import torch
import torch.nn as nn
from accelerate.logging import get_logger

from model.build import MODULE_REGISTRY
from model.pcd_backbone import PointcloudBackbone
from model.transformers import TransformerEncoderLayer, TransformerSpatialEncoderLayer
from model.utils import calc_pairwise_locs, layer_repeat, _init_weights_bert
from model.knn_sparsify import Group
from torch.nn.utils.rnn import pad_sequence

logger = get_logger(__name__)


def generate_fourier_features(pos, num_bands=10, max_freq=15, concat_pos=True, sine_only=False):
    # Input: B, N, C
    # Output: B, N, C'
    batch_size = pos.shape[0]
    device = pos.device

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.linspace(start=min_freq, end=max_freq, steps=num_bands, device=device)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos.unsqueeze(-1).repeat(1, 1, 1, num_bands) * freq_bands
    per_pos_features = torch.reshape(
        per_pos_features, [batch_size, -1, np.prod(per_pos_features.shape[2:])])
    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat(
            [pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features


@MODULE_REGISTRY.register()
class OSE3D(nn.Module):
    # Open-vocabulary, Spatial-attention, Embodied-token, 3D-agent
    def __init__(self, cfg):
        super().__init__()
        self.use_spatial_attn = cfg.use_spatial_attn   # spatial attention
        self.use_embodied_token = cfg.use_embodied_token   # embodied token
        hidden_dim = cfg.hidden_dim

        # pcd backbone
        self.obj_encoder = PointcloudBackbone(cfg.backbone)
        self.obj_proj = nn.Linear(self.obj_encoder.out_dim, hidden_dim)

        # embodied token
        if self.use_embodied_token:
            self.anchor_feat = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.anchor_size = nn.Parameter(torch.ones(1, 1, 3))
        self.orient_encoder = nn.Linear(cfg.fourier_size, hidden_dim)
        self.obj_type_embed = nn.Embedding(2, hidden_dim)

        # spatial encoder
        if self.use_spatial_attn:
            spatial_encoder_layer = TransformerSpatialEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=cfg.spatial_encoder.dim_feedforward,
                dropout=cfg.spatial_encoder.dropout,
                activation=cfg.spatial_encoder.activation,
                spatial_dim=cfg.spatial_encoder.spatial_dim,
                spatial_multihead=cfg.spatial_encoder.spatial_multihead,
                spatial_attn_fusion=cfg.spatial_encoder.spatial_attn_fusion,
            )
        else:
            spatial_encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=cfg.spatial_encoder.dim_feedforward,
                dropout=cfg.spatial_encoder.dropout,
                activation=cfg.spatial_encoder.activation,
            )

        self.spatial_encoder = layer_repeat(
            spatial_encoder_layer,
            cfg.spatial_encoder.num_layers,
        )
        self.pairwise_rel_type = cfg.spatial_encoder.pairwise_rel_type
        self.spatial_dist_norm = cfg.spatial_encoder.spatial_dist_norm
        self.spatial_dim = cfg.spatial_encoder.spatial_dim
        self.obj_loc_encoding = cfg.spatial_encoder.obj_loc_encoding

        # location encoding
        if self.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.obj_loc_encoding == 'diff_all':
            num_loc_layers = cfg.spatial_encoder.num_layers

        loc_layer = nn.Sequential(
            nn.Linear(cfg.spatial_encoder.dim_loc, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.loc_layers = layer_repeat(loc_layer, num_loc_layers)

        logger.info("Build 3D module: OSE3D")

        # only initialize spatial encoder and loc layers
        self.spatial_encoder.apply(_init_weights_bert)
        self.loc_layers.apply(_init_weights_bert)

        if self.use_embodied_token:
            nn.init.normal_(self.anchor_feat, std=0.02)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, data_dict):
        """
        data_dict requires keys:
            obj_fts: (B, N, P, 6), xyz + rgb
            obj_masks: (B, N), 1 valid and 0 masked
            obj_locs: (B, N, 6), xyz + whd
            anchor_locs: (B, 3)
            anchor_orientation: (B, C)
        """

        obj_feats = self.obj_encoder(data_dict['obj_fts'])
        obj_feats = self.obj_proj(obj_feats)
        obj_masks = ~data_dict['obj_masks']   # flipped due to different convention of TransformerEncoder

        B, N = obj_feats.shape[:2]
        device = obj_feats.device

        obj_type_ids = torch.zeros((B, N), dtype=torch.long, device=device)
        obj_type_embeds = self.obj_type_embed(obj_type_ids)

        if self.use_embodied_token:
            # anchor feature
            anchor_orient = data_dict['anchor_orientation'].unsqueeze(1)
            anchor_orient_feat = self.orient_encoder(generate_fourier_features(anchor_orient))
            anchor_feat = self.anchor_feat + anchor_orient_feat
            anchor_mask = torch.zeros((B, 1), dtype=bool, device=device)

            # anchor loc (3) + size (3)
            anchor_loc = torch.cat(
                [data_dict['anchor_locs'].unsqueeze(1), self.anchor_size.expand(B, -1, -1).to(device)], dim=-1
            )

            # anchor type
            anchor_type_id = torch.ones((B, 1), dtype=torch.long, device=device)
            anchor_type_embed = self.obj_type_embed(anchor_type_id)

            # fuse anchor and objs
            all_obj_feats = torch.cat([anchor_feat, obj_feats], dim=1)
            all_obj_masks = torch.cat((anchor_mask, obj_masks), dim=1)

            all_obj_locs = torch.cat([anchor_loc, data_dict['obj_locs']], dim=1)
            all_obj_type_embeds = torch.cat((anchor_type_embed, obj_type_embeds), dim=1)

        else:
            all_obj_feats = obj_feats
            all_obj_masks = obj_masks

            all_obj_locs = data_dict['obj_locs']
            all_obj_type_embeds = obj_type_embeds

        all_obj_feats = all_obj_feats + all_obj_type_embeds

        # call spatial encoder
        if self.use_spatial_attn:
            pairwise_locs = calc_pairwise_locs(
                all_obj_locs[:, :, :3],
                all_obj_locs[:, :, 3:],
                pairwise_rel_type=self.pairwise_rel_type,
                spatial_dist_norm=self.spatial_dist_norm,
                spatial_dim=self.spatial_dim,
            )

        for i, pc_layer in enumerate(self.spatial_encoder):
            if self.obj_loc_encoding == 'diff_all':
                query_pos = self.loc_layers[i](all_obj_locs)
            else:
                query_pos = self.loc_layers[0](all_obj_locs)
            if not (self.obj_loc_encoding == 'same_0' and i > 0):
                all_obj_feats = all_obj_feats + query_pos

            if self.use_spatial_attn:
                all_obj_feats, _ = pc_layer(
                    all_obj_feats, pairwise_locs,
                    tgt_key_padding_mask=all_obj_masks
                )
            else:
                all_obj_feats, _ = pc_layer(
                    all_obj_feats,
                    tgt_key_padding_mask=all_obj_masks
                )

        data_dict['obj_tokens'] = all_obj_feats
        data_dict['obj_masks'] = ~all_obj_masks

        return data_dict


class LearnedAttentionPooling(nn.Module):
    def __init__(self, num_queries=4, input_dim=64, feature_dim=768):
        super(LearnedAttentionPooling, self).__init__()
        # Learnable query tokens - shape [num_queries, feature_dim]

        self.query_tokens = nn.Parameter(torch.randn(num_queries, feature_dim))
        self.proj = nn.Linear(input_dim, feature_dim)      
           
    def forward_old(self, dense_features):
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


    def forward(self, dense_features, attention_mask=None):
        B, N, _ = dense_features.shape
        # Project input features to 768 dim
        features = self.proj(dense_features)  # [B, N, 768]

        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 768]

        attention_scores = torch.bmm(queries, features.transpose(1, 2))  # [B, 4, N]

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # [B, 1, N]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)  # [B, 4, N]
        output = torch.bmm(attention_weights, features)  # [B, 4, 768]

        return output


class ROIAttentionPool(nn.Module):
    def __init__(self, queries_count=4, lang_feat_dim=768, radius=0.15):
        super().__init__()
        self.radius = torch.tensor(radius).to("cuda")
        self.in_dim = lang_feat_dim
        self.pool = LearnedAttentionPooling(queries_count, feature_dim=lang_feat_dim, input_dim=64)
        self.norm = nn.LayerNorm(lang_feat_dim)

    def compute_roi_feats(self, xyz, feat, poi, radius):
        '''
        xyz: (B, N, 3)
        feat: (B, N, C)
        poi: (B, 3)
        '''
        B, N, _ = xyz.shape

        dists = torch.norm(xyz - poi.unsqueeze(1), dim=-1)  # (B, N)
        mask = dists < radius  # (B, N)
    
        roi_feats_list = []
        for b in range(B):
            roi_feats_list.append(feat[b][mask[b]])  # (n_valid, C)
        return roi_feats_list  # list of length B

    def forward(self, xyz, feat, poi, ds):
        '''
        xyz: (B, N, 3)
        feat: (B, N, C)
        poi: (B, 3)
        '''

        roi_feats_list = self.compute_roi_feats(xyz, feat, poi, self.radius)
        pooled_feats = []

        for b, roi_feats in enumerate(roi_feats_list):
            repeat = 2
            while roi_feats.shape[0] == 0 and repeat < 10:
                roi_feats = self.compute_roi_feats(xyz[b:b+1], feat[b:b+1], poi[b:b+1], self.radius * repeat)[0]
                repeat += 1
            if roi_feats.shape[0] == 0:
                # if ds[b]!= "SQA3d":
                #     print(f'POI: {poi[b].to("cpu")}, Dataset {ds[b]}')
                pooled_feats.append(torch.zeros(4, self.in_dim, device=xyz.device))
            else:
                # Apply attention pooling per batch individually
                pooled = self.pool(roi_feats.unsqueeze(0))[0]  # (C,)
                pooled_feats.append(pooled)
    
        pooled_feats = torch.stack(pooled_feats, dim=0) 
        pooled_feats = self.norm(pooled_feats)
        return  pooled_feats # (B, C)


class LearnableFourier(nn.Module):
    def __init__(self, d_in=3, d_pos=768, gauss_scale=1.0, normalize=False):
        super().__init__()
        assert d_pos % 2 == 0
        self.normalize = normalize
        self.gauss_scale = gauss_scale
        B = torch.empty((d_in, d_pos // 2)).normal_() * gauss_scale
        self.gauss_B = nn.Parameter(B) 

    def forward(self, xyz, num_channels=768, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2)
        return final_embeds

class UniformDownsample(nn.Module):
    def __init__(self, num_samples, in_dim=None, out_dim=None):
        """
        Args:
            num_samples: number of points to sample
            in_dim: input feature dimension (required if projection is used)
            out_dim: output feature dimension (set to enable projection)
        """
        super(UniformDownsample, self).__init__()
        self.num_samples = num_samples
        self.cut_xyz=False
        # If out_dim is given, we build a projection layer
        if out_dim is not None:
            assert in_dim is not None, "in_dim must be specified when using projection"
            if in_dim==67:
                self.proj = nn.Linear(64, out_dim)
                self.cut_xyz=True
            else:
                self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = None

    def forward(self, features, attention_mask=None):
        """
        features: [B, N, C]
        attention_mask: [B, N] with 1s for valid and 0s for padding
        """
        B, N, C = features.shape
        device = features.device

        if attention_mask is not None:
            # Mask invalid points with -inf so they don't get sampled
            rand_vals = torch.rand(B, N, device=device)
            rand_vals = rand_vals.masked_fill(attention_mask == 0, -1.0)  # invalid points get -1
        else:
            rand_vals = torch.rand(B, N, device=device)

        # Get top-k random points uniformly (masked invalids get ignored)
        _, sampled_idx = torch.topk(rand_vals, k=self.num_samples, dim=1)

        # Gather points batch-wise
        sampled_idx_expanded = sampled_idx.unsqueeze(-1).expand(-1, -1, C)  # [B, num_samples, C]
        sampled_features = torch.gather(features, dim=1, index=sampled_idx_expanded)  # [B, num_samples, C]

        # Apply projection if defined
        if self.proj is not None:
            if self.cut_xyz:
                sampled_features_xyz =sampled_features[..., :3]
                sampled_features = self.proj(sampled_features[..., 3:])
                sampled_features = torch.cat([sampled_features_xyz, sampled_features], dim=-1)
            else:
                sampled_features = self.proj(sampled_features)  # [B, num_samples, out_dim]

        return sampled_features

@MODULE_REGISTRY.register()
class GS_SceneSplat_Wrapper(nn.Module):
    # Open-vocabulary, Embodied-token, 3D-agent
    def __init__(self, cfg):
        super().__init__()
        self.use_embodied_token = cfg.use_embodied_token   # embodied token
        hidden_dim = cfg.hidden_dim

        # pcd backbone
        self.scene_transformer = PointcloudBackbone(cfg.backbone)
        self.scene_sparsify_1 = UniformDownsample(512, in_dim=256, out_dim=256)
        self.scene_sparsify_2 = UniformDownsample(512, in_dim=128, out_dim=128)
        self.scene_sparsify_3 = UniformDownsample(512, in_dim=67, out_dim=768)
        self.roi_pool = ROIAttentionPool()
        self.pos_emb = LearnableFourier()


        logger.info("Build 3D module: GS_SceneSplat_Wrapper")

        #ROI
        self.radius = 0.3

    @property
    def device(self):
        return list(self.parameters())[0].device

    

    def unpad_and_batch_from_all_one_padding(self, data_dict):
        coords = data_dict["coord"]  # (B, N, 3)
        feats = data_dict["feat"]    # (B, N, C)
        
        B, N, _ = coords.shape
        
        # Create a mask for all valid points across the batch
        mask = ~(coords == 100.0).all(dim=-1)  # (B, N)
        
        # Apply the mask
        coords_cat = coords[mask]  # (num_valid_points, 3)
        feats_cat = feats[mask]    # (num_valid_points, C)
        
        # Generate batch indices
        batch_range = torch.arange(B, device=coords.device).unsqueeze(1)  # (B, 1)
        batch_ids = batch_range.expand(-1, N)[mask]  # (num_valid_points,)
    
        return {
            "coord": coords_cat.to("cuda"),
            "feat": feats_cat.to("cuda"),
            "batch": batch_ids.to("cuda"),
            "grid_size": 0.1
        }

    

    def pad_features_by_batch(self, out_batched_unpadded_info):
        blocks_outputs = []
        attention_masks = []
    
        for i, (out_batched_unpadded, batch_ids) in enumerate(out_batched_unpadded_info):
            B = batch_ids.max().item() + 1  # batch size
            N = out_batched_unpadded.size(0)  # total points
            feat_dim = out_batched_unpadded.size(-1)
    
            # Sort batch_ids for efficient grouping
            sorted_ids, sorted_idx = batch_ids.sort()
            sorted_feats = out_batched_unpadded[sorted_idx]
    
            # Get counts per batch
            counts = torch.bincount(sorted_ids, minlength=B)
    
            # Get max length across batch
            max_len = 40000
    
            # Init padded tensors
            padded_feats = torch.zeros(B, max_len, feat_dim, device=out_batched_unpadded.device)
            attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=out_batched_unpadded.device)
    
            # Generate batch_idx and point_idx tensors for scatter
            batch_idx = torch.cat([torch.full((counts[b],), b, device=counts.device) for b in range(B)])
            point_idx = torch.cat([torch.arange(counts[b], device=counts.device) for b in range(B)])
    
            # Scatter features and mask
            padded_feats[batch_idx, point_idx] = sorted_feats
            attention_mask[batch_idx, point_idx] = 1
    
            blocks_outputs.append(padded_feats)
            attention_masks.append(attention_mask)
    
        return blocks_outputs, attention_masks



    def forward(self, data_dict):
        
        vis_tokens_for_crossattent = [[],[],[]] # For storage of chunks on 1st layer of decoder, 2nd layer of decoder, 3rd layer
        dict_batched_unpadded = self.unpad_and_batch_from_all_one_padding(data_dict)
        out_dict = self.scene_transformer(dict_batched_unpadded, data_dict["scene_id"])
        # out_dict["point_feat"]["hidden_states"] has format [[hidden_states, batch_index] for each block]
        blocks_outputs, attention_masks = self.pad_features_by_batch(out_dict["point_feat"]["hidden_states"])

        
        ####################################################################
        ########### Sparsification and storing for cross-attention #########
        ####################################################################
        vis_tokens_for_crossattent[0].append(self.scene_sparsify_1(blocks_outputs[0], attention_masks[0]))
        vis_tokens_for_crossattent[1].append(self.scene_sparsify_2(blocks_outputs[1], attention_masks[1]))
        
        
        # 3rd layer sparsification
        # add xyz only for spatial grouping
        locs = torch.as_tensor(data_dict["obj_locs"], device="cuda")  # Move to tensor directly
        
        
        
        coord_data = data_dict["coord"].to("cuda")
        blocks_outputs_2_tensor = blocks_outputs[2].to("cuda")  # Now it's a tensor
        
        ds = np.array(data_dict["ds"])
        ds_locs = np.isin(ds, ["Scan2Cap", "ScanObj"])
        if np.any(ds_locs):
            ds_locs = np.where(ds_locs)[0]
            f = self.roi_pool(coord_data[ds_locs], blocks_outputs_2_tensor[ds_locs], locs[ds_locs],[data_dict["ds"][i] for i in ds_locs])
            if f.ndim == 1:
                f = f.unsqueeze(0).expand(4, -1)
            data_dict["roi_feats"] = f

        '''
       
        f = self.roi_pool(coord_data, blocks_outputs_2_tensor, locs,np.array(data_dict["ds"]))
        if f.ndim == 1:
            f = f.unsqueeze(0).expand(4, -1)
        data_dict["roi_feats"] = f
        '''    
        # if valid_locs_mask.any():
        #     f = self.roi_pool(coord_data[valid_locs_mask], blocks_outputs_2_tensor[valid_locs_mask], locs[valid_locs_mask],[data_dict["ds"][i] for i in valid_indices.tolist()])
        #     if f.ndim == 1:
        #         f = f.unsqueeze(0).expand(4, -1)
        #     data_dict["roi_feats"] = f

        scene_pointclouds = torch.cat([coord_data, blocks_outputs_2_tensor], dim=-1)


    
        # neighborhood_xyz= []
        # scene_fts = []
        # for b, mask in zip(scene_pointclouds,attention_masks[1]):
        #     xyz, fts = self.scene_sparsify_3(b[mask.to("cuda")].unsqueeze(0), normalize=False)
        #     neighborhood_xyz.append(xyz)
        #     scene_fts.append(fts)
        # neighborhood_xyz = torch.cat(neighborhood_xyz_list, dim=0)  # [B, G, 3]


        # scene_fts=torch.cat(scene_fts, dim=0)
        # print(neighborhood_xyz.shape)
        # assert not torch.isnan(neighborhood_xyz).any(), "NaN after kNN"
    
        # pos_encoding = self.pos_emb(neighborhood_xyz)
        # assert not torch.isnan(pos_encoding).any(), "NaN after pos"
        # feat_and_pos = scene_fts + pos_encoding
    
        # vis_tokens_for_crossattent[2].append(feat_and_pos)

        # data_dict["scene_tokens"] = vis_tokens_for_crossattent

        B = scene_pointclouds.shape[0]

        neighborhood_xyz_list = []
        scene_fts_list = []
        
        for b_idx in range(B):
            mask = attention_masks[2][b_idx].to(scene_pointclouds.device)
            
            selected_points = scene_pointclouds[b_idx][mask]  # [num_valid, C]
            selected_points = selected_points.unsqueeze(0)  # [1, num_valid, C]
        
            sampled_points = self.scene_sparsify_3(selected_points)  # [1, G, C]
        
            xyz = sampled_points[..., :3]  # First 3 dims are xyz
            fts = sampled_points[..., 3:]  # Rest are features
        
            neighborhood_xyz_list.append(xyz)  # [1, G, 3]
            scene_fts_list.append(fts) 
        
        # Stack efficiently
        neighborhood_xyz = torch.cat(neighborhood_xyz_list, dim=0)  # [B, G, 3]
        scene_fts = torch.cat(scene_fts_list, dim=0)  # [B, G, F]
        
        # NaN check
        assert not torch.isnan(neighborhood_xyz).any(), "NaN after kNN"
        assert not torch.isnan(scene_fts).any(), "NaN after kNN, scene_fts"
        
        # Positional encoding
        pos_encoding = self.pos_emb(neighborhood_xyz)
        #assert not torch.isnan(pos_encoding).any(), "NaN after pos"
        
        # Fuse
        feat_and_pos = scene_fts + pos_encoding  # [B, G, F]
        vis_tokens_for_crossattent[2].append(feat_and_pos)
        
        data_dict["scene_tokens"] = vis_tokens_for_crossattent

        return data_dict

