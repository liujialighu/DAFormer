import numpy as np
from ..builder import PIPELINES


# -------------------------------------------------
# Load Image (NPY version, DAFormer-compatible)
# -------------------------------------------------
@PIPELINES.register_module()
class LoadNpyImage(object):

    def __init__(self, to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):

        dataset = results.get('dataset', None)

        # idx must come from img_info
        idx = results['img_info']['filename']
        idx = int(idx)  # ensure index

        img = dataset.images[idx]  # (4, H, W)

        img = img[:3]  # (3, H, W)
        img = img.transpose(1, 2, 0)  # HWC

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = str(idx)
        results['ori_filename'] = str(idx)

        results['img'] = img

        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape

        results['scale_factor'] = 1.0
        results['flip'] = False
        results['flip_direction'] = None

        num_channels = img.shape[2]

        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False
        )

        return results


# -------------------------------------------------
# Load Annotation (NPY version, fixed)
# -------------------------------------------------
@PIPELINES.register_module()
class LoadNpyAnnotations(object):

    def __call__(self, results):

        dataset = results.get('dataset', None)

        # target domain safe guard
        if results.get('ann_info', None) is None:
            return results

        seg_map = results['ann_info'].get('seg_map', None)
        if seg_map is None:
            return results

        idx = results['img_info']['filename']
        idx = int(idx)

        gt = dataset.labels[idx]   # (1, H, W)
        gt = gt.squeeze(0)         # (H, W)
        gt = gt.astype(np.uint8)

        results['gt_semantic_seg'] = gt

        if 'seg_fields' not in results:
            results['seg_fields'] = []

        results['seg_fields'].append('gt_semantic_seg')

        return results

    def __repr__(self):
        return self.__class__.__name__