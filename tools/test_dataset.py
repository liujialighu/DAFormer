from mmcv import Config

from mmseg.datasets import build_dataset

cfg = Config.fromfile(
    'configs/daformer/taiwan2jiuzhaigou_daformer_mitb5.py'
)

dataset = build_dataset(cfg.data.train)

sample = dataset[0]

print(sample.keys())

print(sample['img'].data.shape)

print(sample['gt_semantic_seg'].data.shape)

print(sample['target_img'].data.shape)