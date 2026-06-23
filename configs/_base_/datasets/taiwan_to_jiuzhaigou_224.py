# # 6.18 成功训练的代码✔🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟
# img_norm_cfg = dict(
#     mean=[0., 0., 0.],
#     std=[10000., 10000., 10000.],
#     to_rgb=False
# )

# crop_size = (224, 224)

# source_train_pipeline = [
#     dict(type='LoadNpyImage'),

#     dict(type='LoadNpyAnnotations'),

#     dict(type='Resize', img_scale=crop_size),

#     dict(type='RandomFlip', prob=0.5),

#     dict(type='Normalize', **img_norm_cfg),

#     dict(type='DefaultFormatBundle'),

#     dict(
#         type='Collect',
#         keys=['img', 'gt_semantic_seg']
#     )
# ]

# target_train_pipeline = [
#     dict(type='LoadNpyImage'),

#     dict(type='Resize', img_scale=crop_size),

#     dict(type='RandomFlip', prob=0.5),

#     dict(type='Normalize', **img_norm_cfg),

#     dict(type='DefaultFormatBundle'),

#     dict(
#         type='Collect',
#         keys=['img']
#     )
# ]

# test_pipeline = [
#     dict(type='LoadNpyImage'),

#     dict(type='Resize', img_scale=crop_size),

#     dict(type='Normalize', **img_norm_cfg),

#     dict(type='ImageToTensor', keys=['img']),

#     dict(
#         type='Collect',
#         keys=['img']
#     )
# ]

# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=2,

#     train=dict(
#         type='UDADataset',

#         source=dict(
#             type='NpyDataset',

#             img_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/train_images.npy',

#             ann_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/train_labels.npy',

#             pipeline=source_train_pipeline
#         ),

#         target=dict(
#             type='NpyDataset',

#             img_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/train_images.npy',

#             ann_path=None,

#             pipeline=target_train_pipeline
#         ),

#         cfg=dict(
#             source=dict(data_root=''),
#             rare_class_sampling=None
#         )
#     ),

#     val=dict(
#         type='NpyDataset',

#         img_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/val_images.npy',

#         ann_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/val_labels.npy',

#         pipeline=source_train_pipeline,

#         test_mode=True
#     ),

#     test=dict(
#         type='NpyDataset',

#         img_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/val_images.npy',

#         ann_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/val_labels.npy',

#         pipeline=test_pipeline,

#         test_mode=True
#     )
# )
























# 6.20 成功测试的代码✔🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟

# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=2,

#     train=dict(
#         type='UDADataset',

#         source=dict(
#             type='NpyDataset',
#             img_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/train_images.npy',
#             ann_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/train_labels.npy',
#         ),

#         target=dict(
#             type='NpyDataset',
#             img_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/train_images.npy',
#             ann_path=None,
#         ),

#         cfg=dict(
#             source=dict(data_root=''),
#             rare_class_sampling=None
#         )
#     ),

#     val=dict(
#         type='NpyDataset',
#         img_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/val_images.npy',
#         ann_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/val_labels.npy',
#         test_mode=True
#     ),

#     test=dict(
#         type='NpyDataset',
#         img_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/val_images.npy',
#         ann_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/val_labels.npy',
#         test_mode=True
#     )
# )








# 6.23 三种格式目标域数据加载训练代码🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,

    train=dict(
        type='UDADataset',

        source=dict(
            type='NpyDataset',
            data_type='npy',
            img_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/train_images.npy',
            ann_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/train_labels.npy',
        ),

        target=dict(
            type='NpyDataset',
            data_type='npy',
            img_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/train_images.npy',
            ann_path=None,
        ),

        cfg=dict(
            source=dict(data_root=''),
            rare_class_sampling=None
        )
    ),

    val=dict(
        type='NpyDataset',
        data_type='npy',
        img_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/val_images.npy',
        ann_path='/home/Liujiali/公共/Domaindata/Taiwan_China_224/val_labels.npy',
        test_mode=True
    ),

    test=dict(
        type='NpyDataset',
        data_type='npy',
        img_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/val_images.npy',
        ann_path='/home/Liujiali/公共/Domaindata/Jiuzhaigou_China_128/val_labels.npy',
        test_mode=True
    ),
)