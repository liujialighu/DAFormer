
#6.18 成功训练的代码✔🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/taiwan_to_jiuzhaigou_224.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0

model = dict(
    pretrained=None,
    backbone=dict(
        pretrained='/home/Liujiali/公共/DAFormer-master/DAFormer-master/pretrained/mit_b5.pth'
    ),
    decode_head=dict(
        num_classes=2
    )
)

uda = dict(
    imnet_feature_dist_lambda=0.0
)

optimizer_config = None

runner = dict(
    type='IterBasedRunner',
    max_iters=20000
)

checkpoint_config = dict(
    by_epoch=False,
    interval=300
)

evaluation = dict(
    interval=300,
    metric='mIoU',
    save_best='mIoU'
)

name = 'taiwan2jiuzhaigou'
exp = 'daformer'








# 6.20 成功测试的代码✔🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟

# _base_ = [
#     '../_base_/default_runtime.py',
#     '../_base_/models/daformer_sepaspp_mitb5.py',
#     '../_base_/datasets/taiwan_to_jiuzhaigou_224.py',
#     '../_base_/uda/dacs.py',
#     '../_base_/schedules/adamw.py',
#     '../_base_/schedules/poly10warm.py'
# ]

# seed = 0

# model = dict(
#     pretrained=None,
#     backbone=dict(
#         pretrained='/home/Liujiali/公共/DAFormer-master/DAFormer-master/pretrained/mit_b5.pth'
#     ),
#     decode_head=dict(
#         num_classes=2
#     )
# )

# uda = dict(
#     imnet_feature_dist_lambda=0.0
# )

# optimizer_config = None

# runner = dict(
#     type='IterBasedRunner',
#     max_iters=20000
# )

# checkpoint_config = dict(
#     by_epoch=False,
#     interval=200
# )

# evaluation = dict(
#     interval=200,
#     metric='mIoU'
# )

# name = 'taiwan2jiuzhaigou'
# exp = 'daformer'




