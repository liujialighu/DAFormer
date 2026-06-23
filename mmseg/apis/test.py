# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# 6.18 成功训练的代码✔🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟
# import os.path as osp
# import tempfile

# import mmcv
# import numpy as np
# import torch
# from mmcv.engine import collect_results_cpu, collect_results_gpu
# from mmcv.image import tensor2imgs
# from mmcv.runner import get_dist_info


# def np2tmp(array, temp_file_name=None, tmpdir=None):
#     """Save ndarray to local numpy file.

#     Args:
#         array (ndarray): Ndarray to save.
#         temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
#             function will generate a file name with tempfile.NamedTemporaryFile
#             to save ndarray. Default: None.
#         tmpdir (str): Temporary directory to save Ndarray files. Default: None.

#     Returns:
#         str: The numpy file name.
#     """

#     if temp_file_name is None:
#         temp_file_name = tempfile.NamedTemporaryFile(
#             suffix='.npy', delete=False, dir=tmpdir).name
#     np.save(temp_file_name, array)
#     return temp_file_name


# def single_gpu_test(model,
#                     data_loader,
#                     show=False,
#                     out_dir=None,
#                     efficient_test=False,
#                     opacity=0.5):
#     """Test with single GPU.

#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (utils.data.Dataloader): Pytorch data loader.
#         show (bool): Whether show results during inference. Default: False.
#         out_dir (str, optional): If specified, the results will be dumped into
#             the directory to save output results.
#         efficient_test (bool): Whether save the results as local numpy files to
#             save CPU memory during evaluation. Default: False.
#         opacity(float): Opacity of painted segmentation map.
#             Default 0.5.
#             Must be in (0, 1] range.
#     Returns:
#         list: The prediction results.
#     """

#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     if efficient_test:
#         mmcv.mkdir_or_exist('.efficient_test')
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, **data)

#         if show or out_dir:
#             img_tensor = data['img'][0]
#             img_metas = data['img_metas'][0].data[0]
#             imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
#             assert len(imgs) == len(img_metas)

#             for img, img_meta in zip(imgs, img_metas):
#                 h, w, _ = img_meta['img_shape']
#                 img_show = img[:h, :w, :]

#                 ori_h, ori_w = img_meta['ori_shape'][:-1]
#                 img_show = mmcv.imresize(img_show, (ori_w, ori_h))

#                 if out_dir:
#                     out_file = osp.join(out_dir, img_meta['ori_filename'])
#                 else:
#                     out_file = None

#                 model.module.show_result(
#                     img_show,
#                     result,
#                     palette=dataset.PALETTE,
#                     show=show,
#                     out_file=out_file,
#                     opacity=opacity)

#         if isinstance(result, list):
#             if efficient_test:
#                 result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
#             results.extend(result)
#         else:
#             if efficient_test:
#                 result = np2tmp(result, tmpdir='.efficient_test')
#             results.append(result)

#         batch_size = len(result)
#         for _ in range(batch_size):
#             prog_bar.update()
#     return results


# def multi_gpu_test(model,
#                    data_loader,
#                    tmpdir=None,
#                    gpu_collect=False,
#                    efficient_test=False):
#     """Test model with multiple gpus.

#     This method tests model with multiple gpus and collects the results
#     under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
#     it encodes results to gpu tensors and use gpu communication for results
#     collection. On cpu mode it saves the results on different gpus to 'tmpdir'
#     and collects them by the rank 0 worker.

#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (utils.data.Dataloader): Pytorch data loader.
#         tmpdir (str): Path of directory to save the temporary results from
#             different gpus under cpu mode. The same path is used for efficient
#             test.
#         gpu_collect (bool): Option to use either gpu or cpu to collect results.
#         efficient_test (bool): Whether save the results as local numpy files to
#             save CPU memory during evaluation. Default: False.

#     Returns:
#         list: The prediction results.
#     """

#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     rank, world_size = get_dist_info()
#     if rank == 0:
#         prog_bar = mmcv.ProgressBar(len(dataset))
#     if efficient_test:
#         mmcv.mkdir_or_exist('.efficient_test')
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)

#         if isinstance(result, list):
#             if efficient_test:
#                 result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
#             results.extend(result)
#         else:
#             if efficient_test:
#                 result = np2tmp(result, tmpdir='.efficient_test')
#             results.append(result)

#         if rank == 0:
#             batch_size = len(result)
#             for _ in range(batch_size * world_size):
#                 prog_bar.update()

#     # collect results from all ranks
#     if gpu_collect:
#         results = collect_results_gpu(results, len(dataset))
#     else:
#         results = collect_results_cpu(results, len(dataset), tmpdir)
#     return results











# 6.20 成功测试的代码✔🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟
# import os.path as osp
# import tempfile

# import mmcv
# import numpy as np
# import torch
# import cv2
# from mmcv.engine import collect_results_cpu, collect_results_gpu
# from mmcv.runner import get_dist_info


# def np2tmp(array, temp_file_name=None, tmpdir=None):
#     if temp_file_name is None:
#         temp_file_name = tempfile.NamedTemporaryFile(
#             suffix='.npy', delete=False, dir=tmpdir).name
#     np.save(temp_file_name, array)
#     return temp_file_name


# def single_gpu_test(model,
#                     data_loader,
#                     show=False,
#                     out_dir=None,
#                     efficient_test=False,
#                     opacity=0.5):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     if efficient_test:
#         mmcv.mkdir_or_exist('.efficient_test')

#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, **data)

#         if show or out_dir:
#             img_tensor = data['img'].data[0]          # (B, 3, H, W)
#             img_metas = data['img_metas'].data[0][0]  # list of dict

#             for idx in range(len(result)):
#                 img_meta = img_metas[idx]
#                 filename = img_meta['filename']

#                 # ------------------------------------------------
#                 # 1. 原图：反归一化后转 uint8
#                 # ------------------------------------------------
#                 img_np = img_tensor[idx].cpu().numpy()   # (3, H, W)
#                 img_np = img_np * 10000.0
#                 img_np = img_np / (img_np.max() + 1e-8) * 255.0
#                 img_np = img_np.clip(0, 255).astype(np.uint8)
#                 img_np = img_np.transpose(1, 2, 0)       # (H, W, 3) RGB
#                 img_bgr = img_np[:, :, ::-1].copy()      # RGB → BGR

#                 # ------------------------------------------------
#                 # 2. 预测图：二值黑白图
#                 # ------------------------------------------------
#                 pred = result[idx]                        # (H, W) 值 0/1
#                 pred_img = (pred * 255).astype(np.uint8)

#                 # ------------------------------------------------
#                 # 3. 掩膜（gt）：从 dataset 读取
#                 # ------------------------------------------------
#                 sample_idx = int(filename)
#                 if dataset.labels is not None:
#                     gt_label = dataset.labels[sample_idx]  # (H, W)
#                     # resize 到 img_size
#                     gt_tensor = torch.from_numpy(
#                         gt_label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
#                     gt_tensor = torch.nn.functional.interpolate(
#                         gt_tensor,
#                         size=(img_meta['img_shape'][0], img_meta['img_shape'][1]),
#                         mode='nearest')
#                     gt_img = (gt_tensor.squeeze().numpy() * 255).astype(np.uint8)
#                 else:
#                     # 目标域没有标签，gt 用全黑占位
#                     h, w = pred_img.shape
#                     gt_img = np.zeros((h, w), dtype=np.uint8)

#                 if out_dir:
#                     mmcv.mkdir_or_exist(out_dir)
#                     # 分别保存三张图
#                     cv2.imwrite(
#                         osp.join(out_dir, f'{filename}_image.png'), img_bgr)
#                     cv2.imwrite(
#                         osp.join(out_dir, f'{filename}_gt.png'), gt_img)
#                     cv2.imwrite(
#                         osp.join(out_dir, f'{filename}_pred.png'), pred_img)

#                 if show:
#                     cv2.imshow(f'{filename}_image', img_bgr)
#                     cv2.imshow(f'{filename}_gt', gt_img)
#                     cv2.imshow(f'{filename}_pred', pred_img)
#                     cv2.waitKey(0)

#         if isinstance(result, list):
#             if efficient_test:
#                 result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
#             results.extend(result)
#         else:
#             if efficient_test:
#                 result = np2tmp(result, tmpdir='.efficient_test')
#             results.append(result)

#         batch_size = len(result)
#         for _ in range(batch_size):
#             prog_bar.update()
#     return results


# def multi_gpu_test(model,
#                    data_loader,
#                    tmpdir=None,
#                    gpu_collect=False,
#                    efficient_test=False):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     rank, world_size = get_dist_info()
#     if rank == 0:
#         prog_bar = mmcv.ProgressBar(len(dataset))
#     if efficient_test:
#         mmcv.mkdir_or_exist('.efficient_test')

#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)

#         if isinstance(result, list):
#             if efficient_test:
#                 result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
#             results.extend(result)
#         else:
#             if efficient_test:
#                 result = np2tmp(result, tmpdir='.efficient_test')
#             results.append(result)

#         if rank == 0:
#             batch_size = len(result)
#             for _ in range(batch_size * world_size):
#                 prog_bar.update()

#     if gpu_collect:
#         results = collect_results_gpu(results, len(dataset))
#     else:
#         results = collect_results_cpu(results, len(dataset), tmpdir)
#     return results



# 6.23 三种格式目标域数据测试代码🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟🍟
import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
import cv2
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.runner import get_dist_info


def np2tmp(array, temp_file_name=None, tmpdir=None):
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'].data[0]          # (B, 3, H, W)
            img_metas = data['img_metas'].data[0][0]  # list of dict

            for idx in range(len(result)):
                img_meta = img_metas[idx]
                filename = img_meta['filename']

                img_np = img_tensor[idx].cpu().numpy()   # (3, H, W)

                # 根据数据集类型选择反归一化方式
                if hasattr(dataset, 'data_type') and dataset.data_type == 'npy':
                    img_np = img_np * 10000.0
                    img_np = img_np / (img_np.max() + 1e-8) * 255.0
                else:
                    # png / tif 已归一化到 [0,1]
                    img_np = img_np * 255.0

                img_np = img_np.clip(0, 255).astype(np.uint8)
                img_np = img_np.transpose(1, 2, 0)       # (H, W, 3)
                img_bgr = img_np[:, :, ::-1].copy()      # RGB → BGR

                pred = result[idx]
                pred_img = (pred * 255).astype(np.uint8)

                sample_idx = int(filename)
                if dataset.labels is not None:
                    gt_label = dataset.labels[sample_idx]
                    gt_tensor = torch.from_numpy(
                        gt_label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    gt_tensor = torch.nn.functional.interpolate(
                        gt_tensor,
                        size=(img_meta['img_shape'][0], img_meta['img_shape'][1]),
                        mode='nearest')
                    gt_img = (gt_tensor.squeeze().numpy() * 255).astype(np.uint8)
                else:
                    h, w = pred_img.shape
                    gt_img = np.zeros((h, w), dtype=np.uint8)

                if out_dir:
                    mmcv.mkdir_or_exist(out_dir)
                    cv2.imwrite(
                        osp.join(out_dir, f'{filename}_image.png'), img_bgr)
                    cv2.imwrite(
                        osp.join(out_dir, f'{filename}_gt.png'), gt_img)
                    cv2.imwrite(
                        osp.join(out_dir, f'{filename}_pred.png'), pred_img)

                if show:
                    cv2.imshow(f'{filename}_image', img_bgr)
                    cv2.imshow(f'{filename}_gt', gt_img)
                    cv2.imshow(f'{filename}_pred', pred_img)
                    cv2.waitKey(0)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results