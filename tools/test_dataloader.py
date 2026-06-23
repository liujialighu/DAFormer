# tools/test_dataloader.py

from mmcv import Config

from mmseg.datasets import build_dataset
from mmseg.datasets import build_dataloader

cfg = Config.fromfile(
    'configs/daformer/taiwan2jiuzhaigou_daformer_mitb5.py'
)

dataset = build_dataset(cfg.data.train)

loader = build_dataloader(
    dataset,
    samples_per_gpu=2,
    workers_per_gpu=2,
    dist=False,
    shuffle=True
)

batch = next(iter(loader))

print(batch['img'].data[0].shape)
print(batch['gt_semantic_seg'].data[0].shape)
print(batch['target_img'].data[0].shape)