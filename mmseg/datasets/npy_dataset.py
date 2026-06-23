# 6.18 成功训练的代码✔🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟
# import numpy as np
# import torch
# import torch.nn.functional as F
# from mmcv.parallel import DataContainer as DC

# from .builder import DATASETS
# from .custom import CustomDataset


# @DATASETS.register_module()
# class NpyDataset(CustomDataset):

#     CLASSES = ('background', 'landslide')
#     PALETTE = [[0, 0, 0], [255, 0, 0]]

#     def __init__(self,
#                  img_path,
#                  ann_path=None,
#                  img_size=224,
#                  pipeline=None,
#                  test_mode=False,
#                  ignore_index=255,
#                  reduce_zero_label=False,
#                  classes=None,
#                  palette=None):

#         self.img_path = img_path
#         self.ann_path = ann_path
#         self.img_size = img_size

#         self.images = np.load(img_path)
#         self.images = self.images[:, :3, :, :]

#         self.labels = None
#         if ann_path is not None:
#             raw = np.load(ann_path)
#             self.labels = raw[:, 0, :, :]

#         self._num_samples = self.images.shape[0]

#         super().__init__(
#             pipeline=[],
#             img_dir='',
#             img_suffix='',
#             ann_dir='',
#             seg_map_suffix='',
#             split=None,
#             test_mode=test_mode,
#             ignore_index=ignore_index,
#             reduce_zero_label=reduce_zero_label,
#             classes=classes,
#             palette=palette
#         )

#     def load_annotations(self, img_dir, img_suffix, ann_dir,
#                          seg_map_suffix, split):
#         img_infos = []
#         for i in range(self._num_samples):
#             info = dict(filename=str(i), ann=dict(seg_map=str(i)))
#             img_infos.append(info)
#         return img_infos

#     def _preprocess_image(self, img):
#         img = img.astype(np.float32) / 10000.0
#         img = np.clip(img, 0.0, 1.0)
#         t = torch.from_numpy(img).unsqueeze(0)
#         t = F.interpolate(t, size=(self.img_size, self.img_size),
#                           mode='bilinear', align_corners=False)
#         return t.squeeze(0)

#     def _preprocess_label(self, label):
#         t = torch.from_numpy(label.astype(np.float32))
#         t = t.unsqueeze(0).unsqueeze(0)
#         t = F.interpolate(t, size=(self.img_size, self.img_size),
#                           mode='nearest')
#         return t.squeeze(0).long()

#     def __getitem__(self, idx):
#         img_np = self.images[idx]
#         img_tensor = self._preprocess_image(img_np)

#         img_meta = dict(
#             ori_shape=(img_np.shape[1], img_np.shape[2], 3),
#             img_shape=(self.img_size, self.img_size, 3),
#             pad_shape=(self.img_size, self.img_size, 3),
#             scale_factor=1.0,
#             flip=False,
#             filename=str(idx),
#             img_norm_cfg=dict(mean=[0.0, 0.0, 0.0],
#                               std=[1.0, 1.0, 1.0],
#                               to_rgb=False)
#         )

#         if self.test_mode:
#             # img 正常 stack 成 (B,3,H,W)
#             # img_metas 套 list，因为 inference 里访问 img_meta[0]
#             return {
#                 'img': DC(img_tensor, stack=True),
#                 'img_metas': DC([img_meta], cpu_only=True),
#             }
#         else:
#             result = {
#                 'img': DC(img_tensor, stack=True),
#                 'img_metas': DC(img_meta, cpu_only=True),
#             }
#             if self.labels is not None:
#                 label_np = self.labels[idx]
#                 label_tensor = self._preprocess_label(label_np)
#                 result['gt_semantic_seg'] = DC(label_tensor, stack=True)
#             return result

#     def prepare_test_img(self, idx):
#         original = self.test_mode
#         self.test_mode = True
#         result = self.__getitem__(idx)
#         self.test_mode = original
#         return result

#     def prepare_train_img(self, idx):
#         original = self.test_mode
#         self.test_mode = False
#         result = self.__getitem__(idx)
#         self.test_mode = original
#         return result

#     def get_gt_seg_maps(self, efficient_test=False):
#         if self.labels is None:
#             return []
#         gt_seg_maps = []
#         for i in range(len(self.labels)):
#             label = self.labels[i]
#             label_tensor = self._preprocess_label(label)
#             gt_seg_maps.append(label_tensor.squeeze(0).numpy().astype(np.uint8))
#         return gt_seg_maps

#     def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
#         from mmseg.core import mean_iou
#         import numpy as np

#         gt_seg_maps = self.get_gt_seg_maps()

#         ret_metrics = mean_iou(
#             results,
#             gt_seg_maps,
#             num_classes=len(self.CLASSES),
#             ignore_index=self.ignore_index,
#             nan_to_num=-1,
#             label_map=dict(),
#             reduce_zero_label=self.reduce_zero_label
#         )

#         class_names = self.CLASSES
#         ret_metrics_summary = dict(
#             aAcc=float(np.round(ret_metrics['aAcc'].mean() * 100, 2)),
#             mIoU=float(np.round(ret_metrics['IoU'].mean() * 100, 2)),
#             mAcc=float(np.round(ret_metrics['Acc'].mean() * 100, 2)),
#         )

#         for i, cls in enumerate(class_names):
#             ret_metrics_summary[f'IoU.{cls}'] = float(
#                 np.round(ret_metrics['IoU'][i] * 100, 2))

#         from mmcv.utils import print_log
#         print_log(ret_metrics_summary, logger=logger)

#         return ret_metrics_summary

#     def __len__(self):
#         return self._num_samples







# 6.20 成功测试的代码✔🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟

# import numpy as np
# import torch
# import torch.nn.functional as F
# from mmcv.parallel import DataContainer as DC
# from mmcv.utils import print_log

# from .builder import DATASETS
# from .custom import CustomDataset


# @DATASETS.register_module()
# class NpyDataset(CustomDataset):

#     CLASSES = ('background', 'landslide')
#     PALETTE = [[0, 0, 0], [255, 0, 0]]

#     def __init__(self,
#                  img_path,
#                  ann_path=None,
#                  img_size=224,
#                  pipeline=None,   # 接收但忽略，不走 CustomDataset pipeline
#                  test_mode=False,
#                  ignore_index=255,
#                  reduce_zero_label=False,
#                  classes=None,
#                  palette=None):

#         self.img_path = img_path
#         self.ann_path = ann_path
#         self.img_size = img_size

#         self.images = np.load(img_path)
#         self.images = self.images[:, :3, :, :]   # (N, 3, H, W)

#         self.labels = None
#         if ann_path is not None:
#             raw = np.load(ann_path)
#             self.labels = raw[:, 0, :, :]        # (N, H, W)

#         self._num_samples = self.images.shape[0]

#         # pipeline 传空列表，完全绕过 CustomDataset 的 pipeline 机制
#         super().__init__(
#             pipeline=[],
#             img_dir='',
#             img_suffix='',
#             ann_dir='',
#             seg_map_suffix='',
#             split=None,
#             test_mode=test_mode,
#             ignore_index=ignore_index,
#             reduce_zero_label=reduce_zero_label,
#             classes=classes,
#             palette=palette
#         )

#     def load_annotations(self, img_dir, img_suffix, ann_dir,
#                          seg_map_suffix, split):
#         img_infos = []
#         for i in range(self._num_samples):
#             info = dict(filename=str(i), ann=dict(seg_map=str(i)))
#             img_infos.append(info)
#         return img_infos

#     def _preprocess_image(self, img):
#         img = img.astype(np.float32) / 10000.0
#         img = np.clip(img, 0.0, 1.0)
#         t = torch.from_numpy(img).unsqueeze(0)
#         t = F.interpolate(t, size=(self.img_size, self.img_size),
#                           mode='bilinear', align_corners=False)
#         return t.squeeze(0)                      # (3, H, W)

#     def _preprocess_label(self, label):
#         t = torch.from_numpy(label.astype(np.float32))
#         t = t.unsqueeze(0).unsqueeze(0)
#         t = F.interpolate(t, size=(self.img_size, self.img_size),
#                           mode='nearest')
#         return t.squeeze(0).long()               # (1, H, W)

#     def __getitem__(self, idx):
#         img_np = self.images[idx]
#         img_tensor = self._preprocess_image(img_np)

#         img_meta = dict(
#             ori_shape=(self.img_size, self.img_size, 3),
#             img_shape=(self.img_size, self.img_size, 3),
#             pad_shape=(self.img_size, self.img_size, 3),
#             scale_factor=1.0,
#             flip=False,
#             filename=str(idx),
#             img_norm_cfg=dict(mean=[0.0, 0.0, 0.0],
#                               std=[1.0, 1.0, 1.0],
#                               to_rgb=False)
#         )

#         if self.test_mode:
#             return {
#                 'img': DC(img_tensor, stack=True),
#                 'img_metas': DC([img_meta], cpu_only=True),
#             }
#         else:
#             result = {
#                 'img': DC(img_tensor, stack=True),
#                 'img_metas': DC(img_meta, cpu_only=True),
#             }
#             if self.labels is not None:
#                 label_tensor = self._preprocess_label(self.labels[idx])
#                 result['gt_semantic_seg'] = DC(label_tensor, stack=True)
#             return result

#     def prepare_train_img(self, idx):
#         original = self.test_mode
#         self.test_mode = False
#         result = self.__getitem__(idx)
#         self.test_mode = original
#         return result

#     def prepare_test_img(self, idx):
#         original = self.test_mode
#         self.test_mode = True
#         result = self.__getitem__(idx)
#         self.test_mode = original
#         return result

#     def get_gt_seg_maps(self, efficient_test=False):
#         if self.labels is None:
#             return []
#         gt_seg_maps = []
#         for i in range(len(self.labels)):
#             label_tensor = self._preprocess_label(self.labels[i])
#             gt_seg_maps.append(
#                 label_tensor.squeeze(0).numpy().astype(np.uint8))
#         return gt_seg_maps

#     def _compute_confusion_matrix(self, results, gt_seg_maps):
#         num_classes = len(self.CLASSES)
#         confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
#         for pred, gt in zip(results, gt_seg_maps):
#             pred = pred.astype(np.int64)
#             gt = gt.astype(np.int64)
#             mask = (gt != self.ignore_index)
#             pred = pred[mask]
#             gt = gt[mask]
#             for t in range(num_classes):
#                 for p in range(num_classes):
#                     confusion[t, p] += np.sum((gt == t) & (pred == p))
#         return confusion

#     def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
#         from mmseg.core import mean_iou

#         gt_seg_maps = self.get_gt_seg_maps()

#         # 1. mIoU
#         ret_metrics = mean_iou(
#             results,
#             gt_seg_maps,
#             num_classes=len(self.CLASSES),
#             ignore_index=self.ignore_index,
#             nan_to_num=-1,
#             label_map=dict(),
#             reduce_zero_label=self.reduce_zero_label
#         )

#         ret_metrics_summary = dict(
#             aAcc=float(np.round(ret_metrics['aAcc'].mean() * 100, 2)),
#             mIoU=float(np.round(ret_metrics['IoU'].mean() * 100, 2)),
#             mAcc=float(np.round(ret_metrics['Acc'].mean() * 100, 2)),
#         )
#         for i, cls in enumerate(self.CLASSES):
#             ret_metrics_summary[f'IoU.{cls}'] = float(
#                 np.round(ret_metrics['IoU'][i] * 100, 2))

#         # 2. 混淆矩阵
#         confusion = self._compute_confusion_matrix(results, gt_seg_maps)
#         TP = confusion[1, 1]
#         FP = confusion[0, 1]
#         FN = confusion[1, 0]
#         TN = confusion[0, 0]
#         total = TP + FP + FN + TN

#         # 3. Precision / Recall / F1（滑坡类）
#         precision    = TP / (TP + FP + 1e-10)
#         recall       = TP / (TP + FN + 1e-10)
#         f1           = 2 * precision * recall / (precision + recall + 1e-10)

#         # 4. Precision / Recall / F1（背景类）
#         precision_bg = TN / (TN + FN + 1e-10)
#         recall_bg    = TN / (TN + FP + 1e-10)
#         f1_bg        = 2 * precision_bg * recall_bg / (precision_bg + recall_bg + 1e-10)

#         # 5. mean F1 / OA / Kappa
#         mean_f1 = (f1 + f1_bg) / 2.0
#         OA      = (TP + TN) / (total + 1e-10)

#         p_land_gt    = (TP + FN) / (total + 1e-10)
#         p_land_pred  = (TP + FP) / (total + 1e-10)
#         p_bg_gt      = (TN + FP) / (total + 1e-10)
#         p_bg_pred    = (TN + FN) / (total + 1e-10)
#         Pe           = p_land_gt * p_land_pred + p_bg_gt * p_bg_pred
#         kappa        = (OA - Pe) / (1 - Pe + 1e-10)

#         ret_metrics_summary.update(dict(
#             Precision_landslide  = float(np.round(precision * 100, 2)),
#             Recall_landslide     = float(np.round(recall * 100, 2)),
#             F1_landslide         = float(np.round(f1 * 100, 2)),
#             Precision_background = float(np.round(precision_bg * 100, 2)),
#             Recall_background    = float(np.round(recall_bg * 100, 2)),
#             F1_background        = float(np.round(f1_bg * 100, 2)),
#             mean_F1              = float(np.round(mean_f1 * 100, 2)),
#             OA                   = float(np.round(OA * 100, 2)),
#             Kappa                = float(np.round(kappa, 4)),
#             TP                   = int(TP),
#             FP                   = int(FP),
#             FN                   = int(FN),
#             TN                   = int(TN),
#         ))

#         print_log('=' * 60, logger=logger)
#         print_log('Evaluation Results:', logger=logger)
#         print_log('=' * 60, logger=logger)
#         for k, v in ret_metrics_summary.items():
#             print_log(f'  {k:<30s}: {v}', logger=logger)
#         print_log('=' * 60, logger=logger)

#         return ret_metrics_summary

#     def __len__(self):
#         return self._num_samples






# 6.23 三种格式目标域数据加载训练代码🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟


# import os
# import os.path as osp
# import numpy as np
# import torch
# import torch.nn.functional as F
# from mmcv.parallel import DataContainer as DC
# from mmcv.utils import print_log

# from .builder import DATASETS
# from .custom import CustomDataset


# @DATASETS.register_module()
# class NpyDataset(CustomDataset):

#     CLASSES = ('background', 'landslide')
#     PALETTE = [[0, 0, 0], [255, 0, 0]]

#     def __init__(self,
#                  # npy 格式参数
#                  img_path=None,
#                  ann_path=None,
#                  # png / tif 格式参数
#                  img_dir=None,
#                  ann_dir=None,
#                  mask_dir=None,
#                  # 通用参数
#                  data_type='npy',    # 'npy' / 'png' / 'tif'
#                  img_size=224,
#                  pipeline=None,
#                  test_mode=False,
#                  ignore_index=255,
#                  reduce_zero_label=False,
#                  classes=None,
#                  palette=None):

#         self.data_type = data_type
#         self.img_size = img_size

#         # 根据 data_type 加载数据
#         if data_type == 'npy':
#             self._load_npy(img_path, ann_path)
#         elif data_type == 'png':
#             self._load_png(img_dir, ann_dir)
#         elif data_type == 'tif':
#             self._load_tif(img_dir, mask_dir, test_mode)
#         else:
#             raise ValueError(f'不支持的 data_type: {data_type}，'
#                              f'请选择 npy / png / tif')

#         self._num_samples = len(self.images)

#         super().__init__(
#             pipeline=[],
#             img_dir='',
#             img_suffix='',
#             ann_dir='',
#             seg_map_suffix='',
#             split=None,
#             test_mode=test_mode,
#             ignore_index=ignore_index,
#             reduce_zero_label=reduce_zero_label,
#             classes=classes,
#             palette=palette
#         )

#     # ----------------------------------------------------------------
#     # 数据加载：npy
#     # ----------------------------------------------------------------
#     def _load_npy(self, img_path, ann_path):
#         assert img_path is not None, 'npy 格式需要指定 img_path'
#         raw = np.load(img_path)              # (N, 4, H, W)
#         self.images = raw[:, :3, :, :]       # (N, 3, H, W) float32

#         self.labels = None
#         if ann_path is not None:
#             raw_label = np.load(ann_path)    # (N, 1, H, W)
#             self.labels = raw_label[:, 0, :, :]  # (N, H, W) 值 0/1

#     # ----------------------------------------------------------------
#     # 数据加载：png
#     # ----------------------------------------------------------------
#     def _load_png(self, img_dir, ann_dir):
#         assert img_dir is not None, 'png 格式需要指定 img_dir'
#         import cv2

#         img_files = sorted([
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith('.png')
#         ])

#         self.images = []
#         self.labels = []

#         for fname in img_files:
#             # 读取 img，BGR → RGB，归一化到 [0,1]
#             img = cv2.imread(osp.join(img_dir, fname))  # (H, W, 3) uint8
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img.astype(np.float32) / 255.0
#             img = img.transpose(2, 0, 1)                # (3, H, W)
#             self.images.append(img)

#             if ann_dir is not None:
#                 ann_path = osp.join(ann_dir, fname)
#                 if osp.exists(ann_path):
#                     mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
#                     mask = (mask > 127).astype(np.float32)             # 0/255 → 0/1
#                     self.labels.append(mask)

#         if len(self.labels) == 0:
#             self.labels = None

#     # ----------------------------------------------------------------
#     # 数据加载：tif
#     # ----------------------------------------------------------------
#     def _load_tif(self, img_dir, mask_dir, test_mode):
#         assert img_dir is not None, 'tif 格式需要指定 img_dir'
#         try:
#             import rasterio
#         except ImportError:
#             raise ImportError('读取 tif 需要安装 rasterio: pip install rasterio')

#         img_files = sorted([
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
#         ])

#         self.images = []
#         self.labels = None

#         for fname in img_files:
#             # 读取 img，3波段 uint8
#             with rasterio.open(osp.join(img_dir, fname)) as src:
#                 img = src.read()                        # (3, H, W) uint8
#             img = img.astype(np.float32) / 255.0       # 归一化到 [0,1]
#             self.images.append(img)

#         # 测试模式才读 mask
#         if test_mode and mask_dir is not None:
#             self.labels = []
#             for fname in img_files:
#                 # mask 文件名和 img 对应
#                 mask_fname = fname
#                 mask_path = osp.join(mask_dir, mask_fname)
#                 if osp.exists(mask_path):
#                     with rasterio.open(mask_path) as src:
#                         mask = src.read(1)              # 单波段 (H, W) 值 0/1
#                     self.labels.append(mask.astype(np.float32))
#             if len(self.labels) == 0:
#                 self.labels = None

#     # ----------------------------------------------------------------
#     # 通用预处理
#     # ----------------------------------------------------------------
#     def load_annotations(self, img_dir, img_suffix, ann_dir,
#                          seg_map_suffix, split):
#         img_infos = []
#         for i in range(self._num_samples):
#             info = dict(filename=str(i), ann=dict(seg_map=str(i)))
#             img_infos.append(info)
#         return img_infos

#     def _preprocess_image(self, img):
#         """
#         img: numpy (3, H, W) float32，值已在 [0,1]
#         → tensor (3, img_size, img_size)
#         """
#         if self.data_type == 'npy':
#             # npy 数据范围 0~10000，归一化到 [0,1]
#             img = img.astype(np.float32) / 10000.0
#             img = np.clip(img, 0.0, 1.0)
#         # png 和 tif 加载时已归一化，直接用

#         t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
#         t = F.interpolate(t, size=(self.img_size, self.img_size),
#                           mode='bilinear', align_corners=False)
#         return t.squeeze(0)                              # (3, H, W)

#     def _preprocess_label(self, label):
#         """
#         label: numpy (H, W) float32，值 0/1
#         → tensor (1, H, W) long
#         """
#         t = torch.from_numpy(label.astype(np.float32))
#         t = t.unsqueeze(0).unsqueeze(0)
#         t = F.interpolate(t, size=(self.img_size, self.img_size),
#                           mode='nearest')
#         return t.squeeze(0).long()                       # (1, H, W)

#     def __getitem__(self, idx):
#         img = self.images[idx]                           # (3, H, W)
#         img_tensor = self._preprocess_image(img)         # (3, 224, 224)

#         img_meta = dict(
#             ori_shape=(self.img_size, self.img_size, 3),
#             img_shape=(self.img_size, self.img_size, 3),
#             pad_shape=(self.img_size, self.img_size, 3),
#             scale_factor=1.0,
#             flip=False,
#             filename=str(idx),
#             img_norm_cfg=dict(mean=[0.0, 0.0, 0.0],
#                               std=[1.0, 1.0, 1.0],
#                               to_rgb=False)
#         )

#         if self.test_mode:
#             return {
#                 'img': DC(img_tensor, stack=True),
#                 'img_metas': DC([img_meta], cpu_only=True),
#             }
#         else:
#             result = {
#                 'img': DC(img_tensor, stack=True),
#                 'img_metas': DC(img_meta, cpu_only=True),
#             }
#             if self.labels is not None:
#                 label_tensor = self._preprocess_label(self.labels[idx])
#                 result['gt_semantic_seg'] = DC(label_tensor, stack=True)
#             return result

#     def prepare_train_img(self, idx):
#         original = self.test_mode
#         self.test_mode = False
#         result = self.__getitem__(idx)
#         self.test_mode = original
#         return result

#     def prepare_test_img(self, idx):
#         original = self.test_mode
#         self.test_mode = True
#         result = self.__getitem__(idx)
#         self.test_mode = original
#         return result

#     def get_gt_seg_maps(self, efficient_test=False):
#         if self.labels is None:
#             return []
#         gt_seg_maps = []
#         for i in range(len(self.labels)):
#             label_tensor = self._preprocess_label(self.labels[i])
#             gt_seg_maps.append(
#                 label_tensor.squeeze(0).numpy().astype(np.uint8))
#         return gt_seg_maps

#     def _compute_confusion_matrix(self, results, gt_seg_maps):
#         num_classes = len(self.CLASSES)
#         confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
#         for pred, gt in zip(results, gt_seg_maps):
#             pred = pred.astype(np.int64)
#             gt = gt.astype(np.int64)
#             mask = (gt != self.ignore_index)
#             pred = pred[mask]
#             gt = gt[mask]
#             for t in range(num_classes):
#                 for p in range(num_classes):
#                     confusion[t, p] += np.sum((gt == t) & (pred == p))
#         return confusion

#     def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
#         from mmseg.core import mean_iou

#         gt_seg_maps = self.get_gt_seg_maps()

#         # 1. mIoU
#         ret_metrics = mean_iou(
#             results,
#             gt_seg_maps,
#             num_classes=len(self.CLASSES),
#             ignore_index=self.ignore_index,
#             nan_to_num=-1,
#             label_map=dict(),
#             reduce_zero_label=self.reduce_zero_label
#         )

#         ret_metrics_summary = dict(
#             aAcc=float(np.round(ret_metrics['aAcc'].mean() * 100, 2)),
#             mIoU=float(np.round(ret_metrics['IoU'].mean() * 100, 2)),
#             mAcc=float(np.round(ret_metrics['Acc'].mean() * 100, 2)),
#         )
#         for i, cls in enumerate(self.CLASSES):
#             ret_metrics_summary[f'IoU.{cls}'] = float(
#                 np.round(ret_metrics['IoU'][i] * 100, 2))

#         # 2. 混淆矩阵
#         confusion = self._compute_confusion_matrix(results, gt_seg_maps)
#         TP = confusion[1, 1]
#         FP = confusion[0, 1]
#         FN = confusion[1, 0]
#         TN = confusion[0, 0]
#         total = TP + FP + FN + TN

#         # 3. Precision / Recall / F1
#         precision    = TP / (TP + FP + 1e-10)
#         recall       = TP / (TP + FN + 1e-10)
#         f1           = 2 * precision * recall / (precision + recall + 1e-10)
#         precision_bg = TN / (TN + FN + 1e-10)
#         recall_bg    = TN / (TN + FP + 1e-10)
#         f1_bg        = 2 * precision_bg * recall_bg / (precision_bg + recall_bg + 1e-10)

#         # 4. mean F1 / OA / Kappa
#         mean_f1 = (f1 + f1_bg) / 2.0
#         OA      = (TP + TN) / (total + 1e-10)

#         p_land_gt   = (TP + FN) / (total + 1e-10)
#         p_land_pred = (TP + FP) / (total + 1e-10)
#         p_bg_gt     = (TN + FP) / (total + 1e-10)
#         p_bg_pred   = (TN + FN) / (total + 1e-10)
#         Pe          = p_land_gt * p_land_pred + p_bg_gt * p_bg_pred
#         kappa       = (OA - Pe) / (1 - Pe + 1e-10)

#         ret_metrics_summary.update(dict(
#             Precision_landslide  = float(np.round(precision * 100, 2)),
#             Recall_landslide     = float(np.round(recall * 100, 2)),
#             F1_landslide         = float(np.round(f1 * 100, 2)),
#             Precision_background = float(np.round(precision_bg * 100, 2)),
#             Recall_background    = float(np.round(recall_bg * 100, 2)),
#             F1_background        = float(np.round(f1_bg * 100, 2)),
#             mean_F1              = float(np.round(mean_f1 * 100, 2)),
#             OA                   = float(np.round(OA * 100, 2)),
#             Kappa                = float(np.round(kappa, 4)),
#             TP                   = int(TP),
#             FP                   = int(FP),
#             FN                   = int(FN),
#             TN                   = int(TN),
#         ))

#         print_log('=' * 60, logger=logger)
#         print_log('Evaluation Results:', logger=logger)
#         print_log('=' * 60, logger=logger)
#         for k, v in ret_metrics_summary.items():
#             print_log(f'  {k:<30s}: {v}', logger=logger)
#         print_log('=' * 60, logger=logger)

#         return ret_metrics_summary

#     def __len__(self):
#         return self._num_samples

import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class NpyDataset(CustomDataset):

    CLASSES = ('background', 'landslide')
    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self,
                 img_path=None,
                 ann_path=None,
                 img_dir=None,
                 ann_dir=None,
                 mask_dir=None,
                 data_type='npy',
                 img_size=224,
                 pipeline=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):

        self.data_type = data_type
        self.img_size = img_size

        if data_type == 'npy':
            self._load_npy(img_path, ann_path)
        elif data_type == 'png':
            self._load_png(img_dir, ann_dir)
        elif data_type == 'tif':
            self._load_tif(img_dir, mask_dir, test_mode)
        else:
            raise ValueError(f'不支持的 data_type: {data_type}，请选择 npy / png / tif')

        self._num_samples = len(self.images)

        super().__init__(
            pipeline=[],
            img_dir='',
            img_suffix='',
            ann_dir='',
            seg_map_suffix='',
            split=None,
            test_mode=test_mode,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            classes=classes,
            palette=palette
        )

    def _ensure_nchw(self, images):
        if images.ndim == 4:
            if images.shape[1] <= 10 and images.shape[-1] > 10:
                return images
            elif images.shape[-1] <= 10 and images.shape[1] > 10:
                print(f"   检测到 (N,H,W,C) 格式，自动转置为 (N,C,H,W)")
                return np.transpose(images, (0, 3, 1, 2))
            else:
                print(f"   ⚠️ 无法确定通道格式，假设为 (N,C,H,W)，请手动确认")
                return images
        return images

    def _load_npy(self, img_path, ann_path):
        assert img_path is not None, 'npy 格式需要指定 img_path'

        images = np.load(img_path).astype(np.float32)
        print(f"原始图像形状: {images.shape}, 值域: {images.min():.1f} ~ {images.max():.1f}")

        images = self._ensure_nchw(images)

        n_channels = images.shape[1]
        if n_channels < 3:
            pad = np.repeat(images[:, -1:, :, :], 3 - n_channels, axis=1)
            images = np.concatenate([images, pad], axis=1)
            print(f"   ⚠️ 通道数 {n_channels} < 3，已复制末通道补齐")
        elif n_channels > 3:
            images = images[:, :3, :, :]
            print(f"   ✅ 通道数 {n_channels} > 3，取前3通道")
        else:
            print(f"   ✅ 通道数 = 3，直接使用")

        # 统一存为 (3, H, W) 的 CHW 格式
        self.images = [images[i] for i in range(images.shape[0])]
        self.image_names = [f"sample_{i:04d}" for i in range(images.shape[0])]

        print(f"✅ 处理后图像形状: {self.images[0].shape}, dtype: {self.images[0].dtype}, "
              f"值域: {self.images[0].min():.1f} ~ {self.images[0].max():.1f}")

        self.labels = None
        if ann_path is not None and os.path.exists(ann_path):
            labels = np.load(ann_path).astype(np.float32)
            print(f"原始标签形状: {labels.shape}")

            if labels.ndim == 4:
                if labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                elif labels.shape[-1] == 1:
                    labels = labels.squeeze(-1)
                else:
                    labels = labels[:, 0, :, :]
            elif labels.ndim == 3:
                pass
            else:
                print(f"⚠️ 标签维度异常: {labels.shape}")
                self.labels = None
                return

            labels = (labels > 0).astype(np.uint8)

            if len(labels) != len(self.images):
                print(f"   ⚠️ 标签数量({len(labels)})与图像数量({len(self.images)})不一致")
                self.labels = None
            else:
                self.labels = [labels[i] for i in range(labels.shape[0])]
                print(f"✅ 标签形状: {self.labels[0].shape}, 正样本比例: {labels.mean():.4f}")
        else:
            if ann_path is not None:
                print(f"   ⚠️ 标签文件不存在: {ann_path}")
            self.labels = None

    def _load_png(self, img_dir, ann_dir):
        """加载PNG格式数据 - 添加详细打印信息"""
        assert img_dir is not None, 'png 格式需要指定 img_dir'
        import cv2

        print("=" * 60)
        print("📁 PNG数据加载开始")
        print("=" * 60)
        
        # 获取所有PNG文件
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith('.png')
        ])
        
        print(f"📊 在 {img_dir} 中找到 {len(img_files)} 个PNG图像文件")
        
        if ann_dir is not None:
            print(f"📊 标签目录: {ann_dir}")
            # 检查标签目录是否存在
            if os.path.exists(ann_dir):
                mask_files = sorted([
                    f for f in os.listdir(ann_dir)
                    if f.lower().endswith('.png')
                ])
                print(f"📊 在 {ann_dir} 中找到 {len(mask_files)} 个标签文件")
            else:
                print(f"⚠️ 标签目录不存在: {ann_dir}")

        self.images = []
        self.labels = []
        self.image_names = []
        missing_mask_count = 0

        print("\n📷 开始加载图像和标签...")
        
        for idx, fname in enumerate(img_files):
            # 加载图像
            img_path = osp.join(img_dir, fname)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"⚠️ 无法读取图像: {fname}")
                continue
                
            # BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)                 # (3, H, W)
            self.images.append(img)
            self.image_names.append(osp.splitext(fname)[0])

            # 加载对应的标签
            if ann_dir is not None:
                ann_path = osp.join(ann_dir, fname)
                if osp.exists(ann_path):
                    mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = (mask > 0).astype(np.float32)
                        self.labels.append(mask)
                    else:
                        print(f"⚠️ 无法读取标签: {fname}")
                        missing_mask_count += 1
                        # 创建全0标签作为占位
                        h, w = img.shape[1], img.shape[2]
                        self.labels.append(np.zeros((h, w), dtype=np.float32))
                else:
                    if idx < 5:  # 只打印前5个缺失的
                        print(f"⚠️ 标签文件不存在: {fname}")
                    missing_mask_count += 1
                    # 创建全0标签作为占位
                    h, w = img.shape[1], img.shape[2]
                    self.labels.append(np.zeros((h, w), dtype=np.float32))

        # 打印加载统计信息
        print("\n" + "=" * 60)
        print("✅ PNG数据加载完成")
        print("=" * 60)
        print(f"📊 加载统计:")
        print(f"   ✅ 成功加载图像: {len(self.images)} 个")
        print(f"   ✅ 成功加载标签: {len(self.labels)} 个")
        if missing_mask_count > 0:
            print(f"   ⚠️ 缺失或无效标签: {missing_mask_count} 个 (已用全0填充)")
        
        if len(self.images) > 0:
            print(f"\n📷 图像信息:")
            print(f"   形状: {self.images[0].shape}")  # (C, H, W)
            print(f"   数据类型: {self.images[0].dtype}")
            print(f"   值域: [{self.images[0].min():.3f}, {self.images[0].max():.3f}]")
            
        if len(self.labels) > 0 and self.labels is not None:
            print(f"\n🏷️ 标签信息:")
            print(f"   形状: {self.labels[0].shape}")  # (H, W)
            print(f"   数据类型: {self.labels[0].dtype}")
            print(f"   值域: [{self.labels[0].min():.1f}, {self.labels[0].max():.1f}]")
            
            # 统计正样本
            positive_samples = []
            for label in self.labels:
                if label.sum() > 0:
                    positive_samples.append(True)
                    ratio = label.mean()
                    if len(positive_samples) <= 5:  # 只打印前5个
                        print(f"   样本 {len(positive_samples)-1}: 正样本比例 {ratio:.4f}")
                else:
                    positive_samples.append(False)
            
            total_samples = len(self.labels)
            pos_count = sum(positive_samples)
            print(f"\n📈 正样本统计:")
            print(f"   包含正样本的图像数: {pos_count}/{total_samples}")
            if pos_count > 0:
                pos_ratios = [l.mean() for l in self.labels if l.sum() > 0]
                print(f"   正样本像素平均比例: {np.mean(pos_ratios):.4f}")
                print(f"   正样本像素最小比例: {np.min(pos_ratios):.4f}")
                print(f"   正样本像素最大比例: {np.max(pos_ratios):.4f}")
            else:
                print(f"   ⚠️ 所有标签均为全0，没有正样本!")
        else:
            print(f"\n⚠️ 没有加载到任何标签")
            self.labels = None
            
        print("=" * 60)

        if len(self.labels) == 0:
            self.labels = None

    def _load_tif(self, img_dir, mask_dir, test_mode):
        """加载TIF格式数据 - 添加详细打印信息"""
        assert img_dir is not None, 'tif 格式需要指定 img_dir'
        try:
            import rasterio
        except ImportError:
            raise ImportError('读取 tif 需要安装 rasterio: pip install rasterio')

        print("=" * 60)
        print("📁 TIF数据加载开始")
        print("=" * 60)

        # 获取所有TIF文件
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
        ])

        print(f"📊 在 {img_dir} 中找到 {len(img_files)} 个TIF图像文件")
        
        if mask_dir is not None:
            print(f"📊 标签目录: {mask_dir}")
            if os.path.exists(mask_dir):
                mask_files = sorted([
                    f for f in os.listdir(mask_dir)
                    if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
                ])
                print(f"📊 在 {mask_dir} 中找到 {len(mask_files)} 个标签文件")
            else:
                print(f"⚠️ 标签目录不存在: {mask_dir}")

        self.images = []
        self.labels = None
        self.image_names = []
        
        print("\n📷 开始加载图像...")
        load_errors = 0
        
        for idx, fname in enumerate(img_files):
            img_path = osp.join(img_dir, fname)
            try:
                with rasterio.open(img_path) as src:
                    img = src.read()  # (C, H, W)
                    
                    # 处理通道数
                    if img.shape[0] >= 3:
                        img = img[:3, :, :]  # 取前3通道
                    elif img.shape[0] < 3:
                        # 复制末通道补齐
                        pad = np.repeat(img[-1:, :, :], 3 - img.shape[0], axis=0)
                        img = np.concatenate([img, pad], axis=0)
                    
                    # 归一化到[0,1]
                    img = img.astype(np.float32) / 255.0
                    self.images.append(img)
                    self.image_names.append(osp.splitext(fname)[0])
                    
            except Exception as e:
                print(f"⚠️ 读取图像失败 {fname}: {e}")
                load_errors += 1
                continue

        print(f"\n📊 图像加载完成: 成功 {len(self.images)} 个, 失败 {load_errors} 个")

        # 加载标签
        if test_mode and mask_dir is not None:
            print("\n📷 开始加载标签...")
            self.labels = []
            missing_mask_count = 0
            
            for fname in img_files:
                mask_path = osp.join(mask_dir, fname)
                if osp.exists(mask_path):
                    try:
                        with rasterio.open(mask_path) as src:
                            mask = src.read(1)  # (H, W)
                        self.labels.append(mask.astype(np.float32))
                    except Exception as e:
                        print(f"⚠️ 读取标签失败 {fname}: {e}")
                        missing_mask_count += 1
                        # 创建全0标签
                        h, w = self.images[len(self.labels)].shape[1], self.images[len(self.labels)].shape[2]
                        self.labels.append(np.zeros((h, w), dtype=np.float32))
                else:
                    if len(missing_mask_count) < 5:
                        print(f"⚠️ 标签文件不存在: {fname}")
                    missing_mask_count += 1
                    # 创建全0标签
                    if len(self.images) > len(self.labels):
                        h, w = self.images[len(self.labels)].shape[1], self.images[len(self.labels)].shape[2]
                        self.labels.append(np.zeros((h, w), dtype=np.float32))

            if len(self.labels) == 0:
                self.labels = None
                print("⚠️ 没有加载到任何标签")
            else:
                print(f"✅ 成功加载标签: {len(self.labels)} 个")
                if missing_mask_count > 0:
                    print(f"⚠️ 缺失或无效标签: {missing_mask_count} 个 (已用全0填充)")

        # 打印统计信息
        print("\n" + "=" * 60)
        print("✅ TIF数据加载完成")
        print("=" * 60)
        print(f"📊 加载统计:")
        print(f"   ✅ 成功加载图像: {len(self.images)} 个")
        
        if len(self.images) > 0:
            print(f"\n📷 图像信息:")
            print(f"   形状: {self.images[0].shape}")  # (C, H, W)
            print(f"   数据类型: {self.images[0].dtype}")
            print(f"   值域: [{self.images[0].min():.3f}, {self.images[0].max():.3f}]")
            
        if self.labels is not None and len(self.labels) > 0:
            print(f"\n🏷️ 标签信息:")
            print(f"   形状: {self.labels[0].shape}")  # (H, W)
            print(f"   数据类型: {self.labels[0].dtype}")
            print(f"   值域: [{self.labels[0].min():.1f}, {self.labels[0].max():.1f}]")
            
            # 统计正样本
            positive_samples = []
            for label in self.labels:
                if label.sum() > 0:
                    positive_samples.append(True)
                else:
                    positive_samples.append(False)
            
            total_samples = len(self.labels)
            pos_count = sum(positive_samples)
            print(f"\n📈 正样本统计:")
            print(f"   包含正样本的图像数: {pos_count}/{total_samples}")
            if pos_count > 0:
                pos_ratios = [l.mean() for l in self.labels if l.sum() > 0]
                print(f"   正样本像素平均比例: {np.mean(pos_ratios):.4f}")
                print(f"   正样本像素最小比例: {np.min(pos_ratios):.4f}")
                print(f"   正样本像素最大比例: {np.max(pos_ratios):.4f}")
            else:
                print(f"   ⚠️ 所有标签均为全0，没有正样本!")
        else:
            print(f"\n⚠️ 没有加载到任何标签 (test_mode={test_mode})")
            
        print("=" * 60)

    def load_annotations(self, img_dir, img_suffix, ann_dir,
                         seg_map_suffix, split):
        img_infos = []
        for i in range(self._num_samples):
            info = dict(filename=str(i), ann=dict(seg_map=str(i)))
            img_infos.append(info)
        return img_infos

    def _preprocess_image(self, img):
        """
        img: numpy (3, H, W) float32
        npy: 值域 0~10000，归一化到 [0,1]
        png/tif: 加载时已归一化到 [0,1]，直接用
        """
        if self.data_type == 'npy':
            img = img.astype(np.float32) / 10000.0
            img = np.clip(img, 0.0, 1.0)

        t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)  # (1, 3, H, W)
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode='bilinear', align_corners=False)
        return t.squeeze(0)                              # (3, H, W)

    def _preprocess_label(self, label):
        """
        label: numpy (H, W) float32，值 0/1
        → tensor (1, H, W) long
        """
        t = torch.from_numpy(label.astype(np.float32))
        t = t.unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode='nearest')
        return t.squeeze(0).long()                       # (1, H, W)

    def __getitem__(self, idx):
        img = self.images[idx]                           # (3, H, W)
        img_tensor = self._preprocess_image(img)         # (3, img_size, img_size)

        img_meta = dict(
            ori_shape=(self.img_size, self.img_size, 3),
            img_shape=(self.img_size, self.img_size, 3),
            pad_shape=(self.img_size, self.img_size, 3),
            scale_factor=1.0,
            flip=False,
            filename=str(idx),
            img_norm_cfg=dict(mean=[0.0, 0.0, 0.0],
                              std=[1.0, 1.0, 1.0],
                              to_rgb=False)
        )

        if self.test_mode:
            return {
                'img': DC(img_tensor, stack=True),
                'img_metas': DC([img_meta], cpu_only=True),
            }
        else:
            result = {
                'img': DC(img_tensor, stack=True),
                'img_metas': DC(img_meta, cpu_only=True),
            }
            if self.labels is not None:
                label_tensor = self._preprocess_label(self.labels[idx])
                result['gt_semantic_seg'] = DC(label_tensor, stack=True)
            return result

    def prepare_train_img(self, idx):
        original = self.test_mode
        self.test_mode = False
        result = self.__getitem__(idx)
        self.test_mode = original
        return result

    def prepare_test_img(self, idx):
        original = self.test_mode
        self.test_mode = True
        result = self.__getitem__(idx)
        self.test_mode = original
        return result

    def get_gt_seg_maps(self, efficient_test=False):
        if self.labels is None:
            return []
        gt_seg_maps = []
        for i in range(len(self.labels)):
            label_tensor = self._preprocess_label(self.labels[i])
            gt_seg_maps.append(
                label_tensor.squeeze(0).numpy().astype(np.uint8))
        return gt_seg_maps

    def _compute_confusion_matrix(self, results, gt_seg_maps):
        num_classes = len(self.CLASSES)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, gt in zip(results, gt_seg_maps):
            pred = pred.astype(np.int64)
            gt = gt.astype(np.int64)
            mask = (gt != self.ignore_index)
            pred = pred[mask]
            gt = gt[mask]
            for t in range(num_classes):
                for p in range(num_classes):
                    confusion[t, p] += np.sum((gt == t) & (pred == p))
        return confusion

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        from mmseg.core import mean_iou

        gt_seg_maps = self.get_gt_seg_maps()

        ret_metrics = mean_iou(
            results,
            gt_seg_maps,
            num_classes=len(self.CLASSES),
            ignore_index=self.ignore_index,
            nan_to_num=-1,
            label_map=dict(),
            reduce_zero_label=self.reduce_zero_label
        )

        ret_metrics_summary = dict(
            aAcc=float(np.round(ret_metrics['aAcc'].mean() * 100, 2)),
            mIoU=float(np.round(ret_metrics['IoU'].mean() * 100, 2)),
            mAcc=float(np.round(ret_metrics['Acc'].mean() * 100, 2)),
        )
        for i, cls in enumerate(self.CLASSES):
            ret_metrics_summary[f'IoU.{cls}'] = float(
                np.round(ret_metrics['IoU'][i] * 100, 2))

        confusion = self._compute_confusion_matrix(results, gt_seg_maps)
        TP = confusion[1, 1]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TN = confusion[0, 0]
        total = TP + FP + FN + TN

        precision    = TP / (TP + FP + 1e-10)
        recall       = TP / (TP + FN + 1e-10)
        f1           = 2 * precision * recall / (precision + recall + 1e-10)
        precision_bg = TN / (TN + FN + 1e-10)
        recall_bg    = TN / (TN + FP + 1e-10)
        f1_bg        = 2 * precision_bg * recall_bg / (precision_bg + recall_bg + 1e-10)
        mean_f1      = (f1 + f1_bg) / 2.0
        OA           = (TP + TN) / (total + 1e-10)

        p_land_gt    = (TP + FN) / (total + 1e-10)
        p_land_pred  = (TP + FP) / (total + 1e-10)
        p_bg_gt      = (TN + FP) / (total + 1e-10)
        p_bg_pred    = (TN + FN) / (total + 1e-10)
        Pe           = p_land_gt * p_land_pred + p_bg_gt * p_bg_pred
        kappa        = (OA - Pe) / (1 - Pe + 1e-10)

        ret_metrics_summary.update(dict(
            Precision_landslide  = float(np.round(precision * 100, 2)),
            Recall_landslide     = float(np.round(recall * 100, 2)),
            F1_landslide         = float(np.round(f1 * 100, 2)),
            Precision_background = float(np.round(precision_bg * 100, 2)),
            Recall_background    = float(np.round(recall_bg * 100, 2)),
            F1_background        = float(np.round(f1_bg * 100, 2)),
            mean_F1              = float(np.round(mean_f1 * 100, 2)),
            OA                   = float(np.round(OA * 100, 2)),
            Kappa                = float(np.round(kappa, 4)),
            TP                   = int(TP),
            FP                   = int(FP),
            FN                   = int(FN),
            TN                   = int(TN),
        ))

        print_log('=' * 60, logger=logger)
        print_log('Evaluation Results:', logger=logger)
        print_log('=' * 60, logger=logger)
        for k, v in ret_metrics_summary.items():
            print_log(f'  {k:<30s}: {v}', logger=logger)
        print_log('=' * 60, logger=logger)

        return ret_metrics_summary

    def __len__(self):
        return self._num_samples