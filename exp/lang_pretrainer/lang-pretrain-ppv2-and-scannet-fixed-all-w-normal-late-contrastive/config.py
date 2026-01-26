weight = '/home/yli7/projects/gaussian_world/GS_Transformer/exp/lang_pretrainer/base-scannet-fix-xyz-all-w-normal-contrastive-siglip2-voting/model/model_best.pth'
resume = False
evaluate = True
test_only = True
seed = 58143646
save_path = 'exp/lang_pretrainer/lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive'
num_worker = 0
batch_size = 48
batch_size_val = 48
batch_size_test = 1
epoch = 800
eval_epoch = 100
clip_grad = None
sync_bn = False
enable_amp = True
empty_cache = False
empty_cache_per_epoch = True
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0006)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(
        type='LangPretrainZeroShotSemSegEval',
        class_names=
        '/home/yli7/projects/gaussian_world/GS_Transformer_debug/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt',
        text_embeddings=
        '/home/yli7/projects/gaussian_world/GS_Transformer_debug/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2.pt',
        excluded_classes=['wall', 'floor', 'ceiling'],
        ignore_index=-1,
        vote_k=25,
        enbale_voting=True,
        confidence_threshold=0.1),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=True)
]
train = dict(type='DefaultTrainer')
test = [
    dict(
        type='ZeroShotSemSegTester',
        verbose=True,
        class_names=
        '/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs/metadata/semantic_benchmark/top100.txt',
        text_embeddings=
        '/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt',
        excluded_classes=['wall', 'floor', 'ceiling'],
        enable_voting=True,
        vote_k=25,
        confidence_threshold=0.1,
        save_feat=False,
        skip_eval=True)
]
data = dict(
    names=[
        'wall', 'ceiling', 'floor', 'table', 'door', 'ceiling lamp', 'cabinet',
        'blinds', 'curtain', 'chair', 'storage cabinet', 'office chair',
        'bookshelf', 'whiteboard', 'window', 'box', 'window frame', 'monitor',
        'shelf', 'doorframe', 'pipe', 'heater', 'kitchen cabinet', 'sofa',
        'windowsill', 'bed', 'shower wall', 'trash can', 'book', 'plant',
        'blanket', 'tv', 'computer tower', 'kitchen counter', 'refrigerator',
        'jacket', 'electrical duct', 'sink', 'bag', 'picture', 'pillow',
        'towel', 'suitcase', 'backpack', 'crate', 'keyboard', 'rack', 'toilet',
        'paper', 'printer', 'poster', 'painting', 'microwave', 'board',
        'shoes', 'socket', 'bottle', 'bucket', 'cushion', 'basket',
        'shoe rack', 'telephone', 'file folder', 'cloth', 'blind rail',
        'laptop', 'plant pot', 'exhaust fan', 'cup', 'coat hanger',
        'light switch', 'speaker', 'table lamp', 'air vent', 'clothes hanger',
        'kettle', 'smoke detector', 'container', 'power strip', 'slippers',
        'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel',
        'pot', 'clock', 'pan', 'tap', 'jar', 'soap dispenser', 'binder',
        'bowl', 'tissue box', 'whiteboard eraser', 'toilet brush',
        'spray bottle', 'headphones', 'stapler', 'marker'
    ],
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type='ScanNetPPGSDataset',
        split=('train_grid1mm_chunk6x6_stride3x3',
               'val_v1_grid1mm_chunk6x6_stride3x3', 'train_scannet_fix_xyz',
               'val_scannet_fix_xyz'),
        data_root=
        '/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs',
        sample_tail_classes=False,
        filtered_scene=[
            'c601466b77', '654a4f341b', '0f25f24a4f', '72f527a47c',
            '2c7c10379b', '5ea3e738c3', '27dd4da69e', '281ba69af1',
            '816e996553'
        ],
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.01),
            dict(
                type='ElasticDistortion',
                distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'color', 'opacity', 'quat', 'scale', 'normal',
                      'segment', 'lang_feat', 'valid_feat_mask'),
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=192000, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'lang_feat',
                      'valid_feat_mask'),
                feat_keys=('color', 'opacity', 'quat', 'scale', 'normal'))
        ],
        test_mode=False,
        loop=8),
    val=dict(
        type='ScanNetPPGSDataset',
        split='val_scannet_fix_xyz',
        data_root=
        '/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs',
        filtered_scene=[
            'c601466b77', '654a4f341b', '0f25f24a4f', '72f527a47c',
            '2c7c10379b', '5ea3e738c3', '27dd4da69e', '281ba69af1',
            '816e996553'
        ],
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'color', 'opacity', 'quat', 'scale', 'normal',
                      'segment', 'lang_feat', 'valid_feat_mask'),
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'lang_feat',
                      'valid_feat_mask'),
                feat_keys=('color', 'opacity', 'quat', 'scale', 'normal'))
        ],
        test_mode=False),
    test=[
        dict(
            type='ScanNetPPGSDataset',
            split='val',
            data_root=
            '/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs',
            transform=[
                dict(type='CenterShift', apply_z=True),
                dict(type='NormalizeColor'),
                dict(
                    type='Copy',
                    keys_dict=dict(
                        segment='origin_segment',
                        coord='origin_coord',
                        valid_feat_mask='origin_feat_mask')),
                dict(
                    type='GridSample',
                    grid_size=0.01,
                    hash_type='fnv',
                    mode='train',
                    keys=('coord', 'color', 'opacity', 'quat', 'scale',
                          'normal', 'lang_feat', 'valid_feat_mask', 'segment'),
                    return_inverse=True)
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(
                    type='GridSample',
                    grid_size=0.02,
                    hash_type='fnv',
                    mode='test',
                    keys=('coord', 'color', 'opacity', 'quat', 'scale',
                          'normal', 'lang_feat', 'valid_feat_mask'),
                    return_grid_coord=True),
                crop=None,
                post_transform=[
                    dict(type='CenterShift', apply_z=False),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'index', 'lang_feat',
                              'valid_feat_mask'),
                        feat_keys=('color', 'opacity', 'quat', 'scale',
                                   'normal'))
                ],
                aug_transform=[[{
                    'type': 'RandomRotateTargetAngle',
                    'angle': [0],
                    'axis': 'z',
                    'center': [0, 0, 0],
                    'p': 1
                }]]))
    ])
debug = 0
gpu_nums = 24
model = dict(
    type='LangPretrainer',
    backbone=dict(
        type='PT-v3m1',
        in_channels=14,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 6),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),
        dec_channels=(768, 512, 256),
        dec_num_head=(16, 16, 16),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D')),
    criteria=[
        dict(type='CosineSimilarity', reduction='mean', loss_weight=1.0),
        dict(type='L2Loss', reduction='mean', loss_weight=1.0),
        dict(
            type='AggregatedContrastiveLoss',
            temperature=0.2,
            reduction='mean',
            loss_weight=0.02,
            schedule='last_75')
    ])
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'ScanNetPPGSDataset'
data_root = '/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v2_default_fix_xyz_gs'
class_names_path = '/home/yli7/projects/gaussian_world/GS_Transformer_debug/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt'
text_embeddings_path = '/home/yli7/projects/gaussian_world/GS_Transformer_debug/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2.pt'
grid_size = 0.01
