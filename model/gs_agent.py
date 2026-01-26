import math

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast, GemmaTokenizer

from model.build import build_module
from model.utils import disabled_train, maybe_autocast
import numpy as np
import gc


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
        xyz=xyz.unsqueeze(1)
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

class CrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        B, Nq, _ = query.shape
        _, Nk, _ = key.shape

       
        # Linear projections
        Q = self.q_proj(query)   # (B, Nq, E)
        K = self.k_proj(key)     # (B, Nk, E)
        V = self.v_proj(value)   # (B, Nk, E)
        Q = F.normalize(Q, p=2, dim=-1)  # L2-normalize each query vector
        K = F.normalize(K, p=2, dim=-1)  # L2-normalize each key vector
        assert not torch.isnan(Q).any(), "NaN in Q after projection"
        assert not torch.isnan(K).any(), "NaN in K after projection"
        assert not torch.isnan(V).any(), "NaN in V after projection"

        # Reshape for multi-head: (B, num_heads, seq_len, head_dim)
        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, D)
        K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, D)
        V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, D)
        assert not torch.isnan(Q).any(), "NaN in Q after reshaping"
        assert not torch.isnan(K).any(), "NaN in K after reshaping"
        assert not torch.isnan(V).any(), "NaN in V after reshaping"

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)

        # if attn_mask is not None:
        #     attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        assert not torch.isinf(attn_scores).any(), "Inf in attn_scores before softmax"

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, Nq, Nk)
        assert not torch.isnan(attn_weights).any(), "NaN in CA after softmax"
        attn_weights = self.dropout(attn_weights)
        assert not torch.isnan(attn_weights).any(), "NaN in CA after dropout"

        attn_output = torch.matmul(attn_weights, V)  # (B, H, Nq, D)
        assert not torch.isnan(attn_output).any(), "NaN in CA after V matmul"

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, Nq, H, D)
        attn_output = attn_output.view(B, Nq, -1)  # (B, Nq, embed_dim)

        # Final projection + residual + norm
        out = self.out_proj(attn_output)
        assert not torch.isnan(attn_output).any(), "NaN in CA after out_proj"
        out = self.dropout(out)
        res = self.q_proj(query)
        assert not torch.isnan(attn_output).any(), "NaN in CA after q_proj"
        out = self.norm(out + res)
        return out

class GSDecoder(nn.Module):
    def __init__(self, query_dim, kv_dims, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(query_dim, kv_dim, embed_dim, num_heads, dropout)
            for kv_dim in kv_dims
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, text_embeddings, three_d_features):
        # text_embeddings: (batch, text_seq_len, embed_dim)
        # three_d_features: (batch, 3d_seq_len, embed_dim)
        assert not torch.isnan(text_embeddings).any(), "NaN before cross-attent, text"
        
        if isinstance(text_embeddings, list): 
            vis_text_embeddings = torch.from_numpy(np.array(text_embeddings)).to("cuda")
        else:
            vis_text_embeddings = text_embeddings.to("cuda")

        # for i, tensor in enumerate(three_d_features):
        #     tensor = torch.cat(tensor, dim=0)
        #     memory_bytes = tensor.element_size() * tensor.nelement()
        #     memory_megabytes = memory_bytes / (1024 ** 2)
            
        #     print(f"Tensor {i} uses {memory_megabytes:.2f} MB")
        #     print(f"Tensor {i} has shape {tensor.shape}")
        i=0
        for vis_feat, cross_attn in zip(three_d_features, self.cross_attn_layers):
            vis_tensor = torch.cat(vis_feat, dim=0)
            if torch.isnan(vis_tensor).any():
                print(f"!!!!! {vis_tensor.shape}")
            assert not torch.isnan(vis_tensor).any(), f"NaN before cross-attent, vis_tensor, iter {i}"
            vis_text_embeddings = cross_attn(vis_text_embeddings, vis_tensor, vis_tensor)  # 3D queries text
            del vis_tensor
            i+=1
            
        return self.norm(vis_text_embeddings)  # Final text-aware 3D features




class LearnedAttentionPooling(nn.Module):
    def __init__(self, num_queries, feature_dim):
        super(LearnedAttentionPooling, self).__init__()
        # Learnable query tokens - shape [num_queries, feature_dim]
        self.query_tokens = nn.Parameter(torch.randn(num_queries, feature_dim, device="cuda"))
        
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
        """
        dense_features: [B, num_features, feature_dim]
        attention_mask: [B, num_features] with 1s for valid tokens and 0s for padding
        """
        B, num_features, _ = dense_features.shape
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, feature_dim]

        # QK^T
        attention_scores = torch.bmm(queries, dense_features.transpose(1, 2))  # [B, num_queries, num_features]

        if attention_mask is not None:
            # Mask out padded positions by setting them to a large negative value before softmax
            attention_mask = attention_mask.unsqueeze(1)  # [B, 1, num_features]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)  # [B, num_queries, num_features]
        output = torch.bmm(attention_weights, dense_features)  # [B, num_queries, feature_dim]

        return output

class MLP768(nn.Module):
    def __init__(self, use_activation=True):
        super().__init__()
        self.linear = nn.Linear(768, 768)
        self.activation = nn.ReLU() if use_activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.linear(x))




class GSAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # LLM
        special_tokens = ['<scene>', '</scene>', '<loc>', '</loc>', '<obj>', '</obj>', "<placehold>"]
        if 'vicuna' in cfg.llm.name.lower():
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(
                cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side
            )
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm_model = LlamaForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        else:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]','additional_special_tokens':special_tokens})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            
            

        logger.info(f"Build {cfg.llm.name} from {cfg.llm.cfg_path}")

        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.llm_model.eval()
        self.llm_model.train = disabled_train
        logger.info("Freeze LLM")

        # 2D vision
        self.img_encoder = build_module(cfg.vision2d)
        self.img_proj = nn.Linear(
            self.img_encoder.out_channels, self.llm_model.config.hidden_size
        )

        # 3D vision 
        self.pcd_encoder = build_module(cfg.vision3d)
        self.text_attention_pool = LearnedAttentionPooling(128, 768)
        self.text_3dvis_crossattent = GSDecoder(query_dim=768, kv_dims=[256,128,768], embed_dim=768, num_heads=4, dropout=0.1)
        # self.text_3dvis_crossattent = GSDecoder(query_dim=768, kv_dims=[768,768,768], embed_dim=768, num_heads=4, dropout=0.1)
        self.pos_embed = nn.Embedding(128, 768)
        self.cache = {}
        self.siglip_txt_model = AutoModel.from_pretrained("siglip2-base-patch16-512/").text_model.to('cuda')
        for param in self.siglip_txt_model.parameters():
            param.requires_grad = False
        self.siglip_txt_model.eval()
        self.siglip_txt_model.train = disabled_train
        logger.info("Freeze SigLIP")
        self.siglip_tokenizer = GemmaTokenizer.from_pretrained("siglip2-base-patch16-512/")
        self.pcd_proj = nn.Linear(
            768, self.llm_model.config.hidden_size
        )
        self.norm_pcd = nn.LayerNorm(self.llm_model.config.hidden_size)
        #self.pos_mlp = MLP768()
        self.click_loc_encoder = LearnableFourier()
        self.click_prompt_projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.LayerNorm(768)
        )
        

        # Obj loc embedding
        #self.obj_loc_embedding = FourierPositionalEmbedding(num_frequencies=15, output_dim=4096)        

        # type embedding
        # self.img_type_embed = nn.Parameter(torch.zeros(self.llm_model.config.hidden_size), requires_grad=True)
        # self.pcd_type_embed = nn.Parameter(torch.zeros(self.llm_model.config.hidden_size), requires_grad=True)

        # LoRA
        if cfg.llm.lora.flag:
            logger.info(f"Apply LoRA with configs: {cfg.llm.lora}")
            lora_config = LoraConfig(
                r=cfg.llm.lora.rank,
                lora_alpha=cfg.llm.lora.alpha,
                target_modules=cfg.llm.lora.target_modules,
                lora_dropout=cfg.llm.lora.dropout,
                bias='none',
                modules_to_save=[],
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config=lora_config)

        self.max_context_len = cfg.llm.max_context_len
        self.max_out_len = cfg.llm.max_out_len

        # additional text x multi-modal tokens fusion
        self.clip_txt_guidance = cfg.clip_txt_guidance.flag
        if self.clip_txt_guidance:
            logger.info("Add CLIP semantics guidance")
            self.clip_model = clip.load('RN50')[0]
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
            self.clip_model.train = disabled_train
            self.clip_proj = nn.Linear(cfg.clip_txt_guidance.clip_out_dim, self.llm_model.config.hidden_size)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def count_params(self, parameters):
        tot = sum([math.prod(p.shape) for p in parameters])
        return tot

    def show_params_size(self, tot):
        if tot >= 1e9:
            return '{:.1f}B'.format(tot / 1e9)
        elif tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}k'.format(tot / 1e3)

    def get_learnable_named_params(self):
        learnable_named_params = {}
        frozen_named_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                learnable_named_params.update({n: p})
            else:
                frozen_named_params.update({n: p})
        learnable_params_size = self.count_params(learnable_named_params.values())
        frozen_params_size = self.count_params(frozen_named_params.values())
        logger.info(
            f"Build LEO with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
        logger.info(f"🧊 Frozen parameters: {list(frozen_named_params.keys())}")
        logger.info(f"🔥 Tuned parameters: {list(learnable_named_params.keys())}")

        return learnable_named_params

    def build_right_justified_sequence(self, data_dict):

        """
        Concat six sequences: `prompt_before_obj`, `prompt_middle_1`, `img_tokens`, `prompt_middle_2`, `obj_tokens`, `prompt_after_obj`.
        Return right justified sequence for causal LM: <pad>, <role/situation>, <img>, <objs>, <instruction>.
        """
        device = self.device
        bs = len(data_dict['prompt_before_obj'])
        use_obj_loc=0

        self.llm_tokenizer.padding_side = 'left'
        text_input_tokens_pre = self.llm_tokenizer(
            data_dict['prompt_before_obj'],
            return_tensors='pt',
            padding='longest'
        ).to(device)   # [PAD, BOS, tokens], (B, T1)

        text_input_tokens_mid1 = self.llm_tokenizer(
            data_dict['prompt_middle_1'],
            return_tensors='pt',
            padding='longest'
        ).to(device)

        img_tokens = data_dict['img_tokens'].to(device)
        img_masks = data_dict['img_masks'].to(device)
        img_masks = img_masks.reshape(-1, 1).repeat(1, img_tokens.size(1))

        text_input_tokens_mid2 = self.llm_tokenizer(
            data_dict['prompt_middle_2'],
            return_tensors='pt',
            padding='longest'
        ).to(device)

        obj_tokens = data_dict['scene_tokens_enhanced'].to(device)
        obj_masks = data_dict['obj_masks'].to(device)

        # additional clip fusion
        if self.clip_txt_guidance:
            with torch.no_grad():
                clip_fts = self.clip_model.encode_text(
                    clip.tokenize(data_dict['prompt_after_obj_start'], truncate=True).to(device)
                )
            clip_fts = self.clip_proj(clip_fts.to(obj_tokens.dtype))
            # B, N, C
            img_tokens = torch.einsum('bnc,bc->bnc', img_tokens, clip_fts)
            obj_tokens = torch.einsum('bnc,bc->bnc', obj_tokens, clip_fts)

        self.llm_tokenizer.padding_side = 'right'   # no need to be 'left', as padding tokens will be shifted
        self.llm_tokenizer.truncation_side = 'left'   # truncate history
        text_input_tokens_post = self.llm_tokenizer(
            data_dict['prompt_after_obj_start'],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_context_len,
        ).to(device)   # [BOS, tokens, PAD], (B, T3)

        text_input_tokens_post_end = self.llm_tokenizer(
            data_dict['prompt_after_obj_end'],
            return_tensors='pt',
            padding='longest'
        ).to(device)   # [BOS, tokens, PAD], (B, T3)

        assert text_input_tokens_mid1.attention_mask.all() and text_input_tokens_mid2.attention_mask.all(), \
               "prompt_middle should be the same and thus no padding"

        # remove bos, make "tokenize subseq and concat" equivalent to "tokenize the whole seq"
        text_input_tokens_mid1.input_ids = text_input_tokens_mid1.input_ids[:, 1:]
        text_input_tokens_mid1.attention_mask = text_input_tokens_mid1.attention_mask[:, 1:]
        text_input_tokens_mid2.input_ids = text_input_tokens_mid2.input_ids[:, 1:]
        text_input_tokens_mid2.attention_mask = text_input_tokens_mid2.attention_mask[:, 1:]
        text_input_tokens_post.input_ids = text_input_tokens_post.input_ids[:, 1:]
        text_input_tokens_post.attention_mask = text_input_tokens_post.attention_mask[:, 1:]
        text_input_tokens_post_end.input_ids = text_input_tokens_post_end.input_ids[:, 1:]
        text_input_tokens_post_end.attention_mask = text_input_tokens_post_end.attention_mask[:, 1:]
        for i in range(bs):
            if not img_masks[i].any():
                # no image input, also mask the text prompt for image tokens
                text_input_tokens_mid1.attention_mask[i].fill_(0)

        inputs_embeds_pre = self.llm_model.get_input_embeddings()(text_input_tokens_pre.input_ids)
        inputs_embeds_mid1 = self.llm_model.get_input_embeddings()(text_input_tokens_mid1.input_ids)
        inputs_embeds_mid2 = self.llm_model.get_input_embeddings()(text_input_tokens_mid2.input_ids)
        inputs_embeds_post = self.llm_model.get_input_embeddings()(text_input_tokens_post.input_ids)
        inputs_embeds_post_end = self.llm_model.get_input_embeddings()(text_input_tokens_post_end.input_ids)

        # since img_tokens, prompt_mid, obj_tokens are fixed length without padding, we concat them first
        inputs_embeds_mid = torch.cat([inputs_embeds_mid1, img_tokens, inputs_embeds_mid2, obj_tokens], dim=1)
        attn_mask_mid = torch.cat(
            [text_input_tokens_mid1.attention_mask, img_masks, text_input_tokens_mid2.attention_mask, obj_masks],
            dim=1,
        )

        post_pad_length = torch.logical_not(text_input_tokens_post.attention_mask).sum(-1)

        bs, l1, hidden_dim = inputs_embeds_pre.shape
        _, l2, _ = inputs_embeds_mid.shape
        _, l3, _ = inputs_embeds_post.shape
        _, l4, _ = inputs_embeds_post_end.shape

        total_len = l1 + l2 + (l3+use_obj_loc) + l4 # +1 for the obj loc token
        inputs_embeds = torch.zeros(bs, total_len, hidden_dim).type(inputs_embeds_pre.dtype).to(device)
        attention_mask = torch.zeros(bs, total_len).type(obj_masks.dtype).to(device)

        # assign by chunks
        for i in range(bs):
            post_pad_len = post_pad_length[i]

            if post_pad_len > 0:
                inputs_embeds[i, :post_pad_len] = inputs_embeds_post[i, -post_pad_len:]
                attention_mask[i, :post_pad_len] = 0
                # NEW: Instead of writing all the way to the end ([:]), 
                #we now write up to -l4-1, leaving the [-1-l4] position free for the location token.
                inputs_embeds[i, post_pad_len+l1+l2: -l4 -use_obj_loc] = inputs_embeds_post[i, :-post_pad_len]
                attention_mask[i, post_pad_len+l1+l2: -l4 -use_obj_loc] = 1
            else:
                # no padding
                inputs_embeds[i, -l3 -use_obj_loc -l4: -l4 -use_obj_loc] = inputs_embeds_post[i]
                attention_mask[i, -l3 -use_obj_loc -l4: -l4 -use_obj_loc] = 1

            inputs_embeds[i, post_pad_len: post_pad_len+l1] = inputs_embeds_pre[i]
            attention_mask[i, post_pad_len: post_pad_len+l1] = text_input_tokens_pre.attention_mask[i]

            inputs_embeds[i, post_pad_len+l1: post_pad_len+l1+l2] = inputs_embeds_mid[i]
            attention_mask[i, post_pad_len+l1: post_pad_len+l1+l2] = attn_mask_mid[i]

            # NEW: Append loc_token_emb: inputs_embeds[i, -1] is the last token in the sequence, the one we left free
            inputs_embeds[i, -l4: ] = inputs_embeds_post_end[i]
            attention_mask[i, -l4: ] = torch.ones(l4)


        
        ds = np.array(data_dict["ds"])
        ds_locs = np.isin(ds, ["Scan2Cap", "ScanObj"])
        if np.any(ds_locs):
            ds_locs = np.where(ds_locs)[0]
            inputs_embeds[ds_locs, -l4+1 : -l4 + 5] = data_dict["roi_feats"].to(inputs_embeds.dtype)
        '''
        inputs_embeds[:, -l4+1 : -l4 + 5] = data_dict["roi_feats"].to(inputs_embeds.dtype)
        '''
        return inputs_embeds, attention_mask


    def tokenize_text_siglip(self, data_dict):
        device = self.device
        prompt=[]
        loc_mask=[]
        loc_enc = []
        for i, item in enumerate(data_dict['prompt_after_obj_start']):
            loc = data_dict["obj_locs"][i]
            if  loc == [-100,-100,-100]:
                prompt.append(item.split("USER:")[-1])
            else:
                prompt.append(item.split("USER:")[-1].split("<loc>")[0] + "<pad>")
                loc_mask.append(i)
                loc_enc.append(loc)
        
        self.siglip_tokenizer.padding_side = 'left'
        inputs = self.siglip_tokenizer(prompt, padding="longest", return_tensors="pt")
        
        
        inputs = inputs.to("cuda")
        with torch.no_grad():
            if inputs['input_ids'].shape[1]>64:
                text_features = self.siglip_txt_model.embeddings(input_ids=inputs['input_ids'][:,:64])
            else:
                text_features = self.siglip_txt_model.embeddings(input_ids=inputs['input_ids'])

        if loc_enc!=[]:
            # loc_emb = generate_fourier_features(torch.FloatTensor(np.array(loc_enc)).to("cuda"), num_bands=128 , concat_pos=False)
            # loc_emb = self.pos_mlp(loc_emb)
            loc_emb = self.click_loc_encoder(torch.FloatTensor(np.array(loc_enc)).to("cuda"))
            loc_emb = self.click_prompt_projector(loc_emb)
            # where to update with pos encodings
            loc_mask = np.array(loc_mask)
            text_features[loc_mask,-1] = loc_emb[:,0].to(text_features.dtype)
        return text_features

    def mean_pool(self, embeddings, mask=None):
        # embeddings: (B, T, D)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (B, T, 1)
            summed = torch.sum(embeddings * mask, dim=1)
            count = torch.clamp(mask.sum(dim=1), min=1e-6)
            return summed / count
        else:
            return embeddings.mean(dim=1)
        
    def contrastive_loss(self, model_embeds, class_token_embeds, txt_mask,  temperature=0.07):
        # Step 1: Mean pool both
        model_pooled = self.mean_pool(model_embeds)  # (B, D)
        class_pooled = self.mean_pool(class_token_embeds, txt_mask)  # (B, D)
        if torch.isnan(model_pooled).any():
            print(f"For model: pooled isnan: {torch.isnan(model_pooled).any()}, norms are {torch.norm(model_embeds, dim=-1)} \nFor txt:{torch.isnan(class_pooled).any()}, norms are {torch.norm(class_token_embeds, dim=-1)}")
    
        # Step 2: Normalize (for cosine similarity)
        model_norm = F.normalize(model_pooled, dim=-1)
        class_norm = F.normalize(class_pooled, dim=-1)
    
        # Step 3: Compute similarity matrix
        logits = torch.matmul(model_norm, class_norm.T) / temperature  # (B, B)
    
        # Step 4: Labels = Identity matrix (i.e., batch item i should match only class i)
        labels = torch.arange(model_embeds.size(0), device=model_embeds.device)
    
        # Step 5: Contrastive Loss (InfoNCE)
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
    
        return loss_per_sample


    def forward_pretrain(self, data_dict):
        """
        data_dict requires keys:
        # input
        prompt_before_obj: list of str, (B,)
        prompt_middle_1: list of str, (B,)
        prompt_middle_2: list of str, (B,)
        prompt_after_obj_start: list of str, (B,)
        prompt_after_obj_end: list of str, (B,)
        obj_fts: (B, N, P, 6), xyz + rgb
        obj_masks: (B, N), 1 valid and 0 masked
        obj_locs: (B, N, 6), xyz + whd
        anchor_locs: (B, 3)
        anchor_orientation: (B, C)
        img_fts: (B, 3, H, W), rgb
        img_masks: (B, 1), 1 valid and 0 masked
        # output
        output_gt: list of str, (B,)
        """
        device = self.device
        with torch.cuda.amp.autocast(enabled=True):
            bs = len(data_dict['prompt_after_obj_start'])
            data_dict["scene_tokens"]=[[],[],[]]
            data_dict["roi_feats"]=[]

            data_dict_instance = self.pcd_encoder( {
                "coord": data_dict["coord"],
                "feat": data_dict["feat"],
                "obj_locs": data_dict["obj_locs"],
                "grid_size": 0.1,
                "scene_id":data_dict["scene_id"],
                "ds": data_dict["ds"]
            })
            #if "roi_feats" in data_dict_instance:
            #    data_dict["roi_feats"]=data_dict_instance["roi_feats"]
            for i, batch_layer in enumerate(data_dict_instance["scene_tokens"]):
                data_dict["scene_tokens"][i].append(batch_layer[0])
                    
                assert not torch.isnan(batch_layer[0]).any(), "NaN after pcd_encoder"
                

            text_token_emb = self.tokenize_text_siglip(data_dict)
            assert not torch.isnan(text_token_emb).any(), "NaN after siglip"
            text_token_emb = self.text_attention_pool(text_token_emb)
            assert not torch.isnan(text_token_emb).any(), "NaN after attention pool"
    
            pcd = self.text_3dvis_crossattent(text_embeddings=text_token_emb, 
                                                                  three_d_features=data_dict['scene_tokens'])
            assert not torch.isnan(pcd).any(), "NaN after cross-attent"
            pcd = self.pcd_proj(pcd)
            data_dict['scene_tokens_enhanced'] = self.norm_pcd(pcd)
            assert not torch.isnan(data_dict['scene_tokens_enhanced']).any(), "NaN after norm"

            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in data_dict['output_gt']],
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=self.max_out_len,
            ).to(device)
            
            with torch.no_grad():
                text_output_embeds = self.llm_model.get_input_embeddings()(text_output_tokens.input_ids)
            loss = self.contrastive_loss(data_dict['scene_tokens_enhanced'], text_output_embeds, text_output_tokens.attention_mask)
            if not torch.isfinite(loss).all():
                print(f"GT: {data_dict['output_gt']}, \n Mine: {loss}")
                print(a)
            data_dict.update({'loss': loss})
        return data_dict
            
        
    def forward(self, data_dict):
        """
        data_dict requires keys:
        # input
        prompt_before_obj: list of str, (B,)
        prompt_middle_1: list of str, (B,)
        prompt_middle_2: list of str, (B,)
        prompt_after_obj_start: list of str, (B,)
        prompt_after_obj_end: list of str, (B,)
        obj_fts: (B, N, P, 6), xyz + rgb
        obj_masks: (B, N), 1 valid and 0 masked
        obj_locs: (B, N, 6), xyz + whd
        anchor_locs: (B, 3)
        anchor_orientation: (B, C)
        img_fts: (B, 3, H, W), rgb
        img_masks: (B, 1), 1 valid and 0 masked
        # output
        output_gt: list of str, (B,)
        """
        device = self.device
        with torch.cuda.amp.autocast(enabled=True):
            bs = len(data_dict['prompt_after_obj_start'])
            if 'scene_tokens' not in data_dict:
                data_dict["scene_tokens"]=[[],[],[]]
                data_dict["roi_feats"]=[]
    
                data_dict_instance = self.pcd_encoder( {
                    "coord": data_dict["coord"],
                    "feat": data_dict["feat"],
                    "obj_locs": data_dict["obj_locs"],
                    "grid_size": 0.1,
                    "scene_id":data_dict["scene_id"],
                    "ds": data_dict["ds"]
                })
                if "roi_feats" in data_dict_instance:
                    data_dict["roi_feats"]=data_dict_instance["roi_feats"]
                
                for i, batch_layer in enumerate(data_dict_instance["scene_tokens"]):
                    data_dict["scene_tokens"][i].append(batch_layer[0])
                

            text_token_emb = self.tokenize_text_siglip(data_dict)
            text_token_emb = self.text_attention_pool(text_token_emb)
    
            pcd = self.text_3dvis_crossattent(text_embeddings=text_token_emb, 
                                                                  three_d_features=data_dict['scene_tokens'])
            pcd = self.pcd_proj(pcd)
            data_dict['scene_tokens_enhanced'] = self.norm_pcd(pcd)

    
            if len(data_dict['roi_feats']) >0:
                roi = self.pcd_proj(data_dict['roi_feats'])
                data_dict['roi_feats'] = self.norm_pcd(roi)
    
    
            # data_dict['obj_tokens'] = data_dict['obj_tokens'] + self.pcd_type_embed
    
            data_dict['img_tokens'] = self.img_proj(self.img_encoder(data_dict['img_fts']))
            # data_dict['img_tokens'] = data_dict['img_tokens'] + self.img_type_embed
    
            inputs_embeds, attention_mask = self.build_right_justified_sequence(data_dict=data_dict)
            # (B, T1+O+T2, D), (B, T1+O+T2)
            # inputs_embeds=data_dict['scene_tokens_enhanced']
            # attention_mask=torch.ones(4,128,device="cuda")
            self.llm_tokenizer.padding_side = 'right'
            self.llm_tokenizer.truncation_side = 'right'
            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in data_dict['output_gt']],
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=self.max_out_len,
            ).to(device)
            
            text_output_embeds = self.llm_model.get_input_embeddings()(text_output_tokens.input_ids)   # (B, T3, D)
            inputs_embeds = torch.cat([inputs_embeds, text_output_embeds], dim=1)   # (B, T1+O+T2+T3, D)
            attention_mask = torch.cat([attention_mask, text_output_tokens.attention_mask], dim=1)   # (B, T1+O+T2+T3)
    
            # construct targets
            targets = torch.zeros_like(attention_mask).long().fill_(-100)   # (B, T1+O+T2+T3)
    
            # only apply loss to answer tokens
            targets_idx = text_output_tokens.attention_mask.bool()
            targets[:, -targets_idx.shape[1]:][targets_idx] = text_output_tokens.input_ids[targets_idx]
    
            # do not predict bos token, regard it as condition instead
            targets[:, -targets_idx.shape[1]] = -100
            with maybe_autocast(self):
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
    
            logits = outputs.logits.float()
    
            # different from the loss inside `llm_model.forward`, here we take mean of each sequence instead of sum
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            num_tokens_for_loss = (shift_labels >= 0).int().sum(1)   # (B,)
            num_tokens_for_loss = num_tokens_for_loss.clamp(min=1)

    
            shift_logits = rearrange(shift_logits, 'b t v -> (b t) v')
            shift_labels = rearrange(shift_labels, 'b t -> (b t)')
    
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            loss = rearrange(loss, '(b t) -> b t', b=bs)
            loss = loss.sum(1) / num_tokens_for_loss   # (B,)
    
            data_dict.update({'loss': loss})
            if not torch.isfinite(loss).all():
                print(f"GT: {data_dict['output_gt']}, \n Mine: {loss} \n num_tokens_for_loss: {num_tokens_for_loss} \n CE: {F.cross_entropy(shift_logits, shift_labels, reduction='none')}")
                print(a)
        # do not average loss, average txt and eai respectively in Trainer.train_step() instead
        return data_dict

    @torch.no_grad()
    def generate(
        self,
        data_dict,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=768,
        min_length=1,
        top_p=0.9,
        repetition_penalty=3.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        """
        data_dict requires the same keys as forward() except output_gt
        """
        device = self.device
        with torch.cuda.amp.autocast(enabled=True):
            bs = len(data_dict['prompt_after_obj_start'])
            if 'scene_tokens' not in data_dict:
                data_dict["scene_tokens"]=[[],[],[]]
                data_dict["roi_feats"]=[]
    
                data_dict_instance = self.pcd_encoder( {
                    "coord": data_dict["coord"],
                    "feat": data_dict["feat"],
                    "obj_locs": data_dict["obj_locs"],
                    "grid_size": 0.1,
                    "scene_id":data_dict["scene_id"],
                    "ds": data_dict["ds"]
                })
                if "roi_feats" in data_dict_instance:
                    data_dict["roi_feats"]=data_dict_instance["roi_feats"]
                for i, batch_layer in enumerate(data_dict_instance["scene_tokens"]):
                    data_dict["scene_tokens"][i].append(batch_layer[0])
                    
            text_token_emb = self.tokenize_text_siglip(data_dict)
            text_token_emb = self.text_attention_pool(text_token_emb)
    
            pcd = self.text_3dvis_crossattent(text_embeddings=text_token_emb, 
                                                                  three_d_features=data_dict['scene_tokens'])
            pcd = self.pcd_proj(pcd)
            data_dict['scene_tokens_enhanced'] = self.norm_pcd(pcd)
    
            if len(data_dict['roi_feats']) >0:
                roi = self.pcd_proj(data_dict['roi_feats'])
                data_dict['roi_feats'] = self.norm_pcd(roi)
                
            
    
    
            # data_dict['obj_tokens'] = data_dict['obj_tokens'] + self.pcd_type_embed
    
            data_dict['img_tokens'] = self.img_proj(self.img_encoder(data_dict['img_fts']))
            # data_dict['img_tokens'] = data_dict['img_tokens'] + self.img_type_embed
    
            inputs_embeds, attention_mask = self.build_right_justified_sequence(data_dict=data_dict)
        
       
            # give bos token as condition
            bos_tokens = self.llm_tokenizer(
                [self.llm_tokenizer.bos_token] * bs,
                return_tensors='pt',
            ).to(device)
            bos_tokens_ids = bos_tokens.input_ids[:, 0:1]   # (B, 1)
            bos_tokens_attn = bos_tokens.attention_mask[:, 0:1]   # (B, 1)
    
            # prepare a `bos_token`
            bos_embeds = self.llm_model.get_input_embeddings()(bos_tokens_ids)   # (B, 1, D)
            inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)   # (B, T1+O+T2+1, D)
            attention_mask = torch.cat([attention_mask, bos_tokens_attn], dim=1)   # (B, T1+O+T2+1)
    
            with maybe_autocast(self):
                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=768,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
    
            outputs[outputs == self.llm_tokenizer.unk_token_id] = self.llm_tokenizer.eos_token_id
            # data_dict['output_tokens'] = outputs   # unable to gather variable-length tensors
    
            output_txt = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_txt = [txt.strip() for txt in output_txt]
            data_dict['output_txt'] = output_txt
        return data_dict

    
