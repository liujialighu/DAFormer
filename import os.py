# import os
# import numpy as np
# import cv2
# import argparse
# from collections import Counter
# import glob

# def analyze_png_dataset(images_dir, masks_dir, verbose=False):
#     """
#     分析PNG图像和掩膜数据集
#     """
    
#     print("=" * 80)
#     print("PNG数据集分析报告")
#     print("=" * 80)
    
#     # 获取所有图像文件
#     image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
#     image_files = sorted([f for f in os.listdir(images_dir) 
#                          if f.lower().endswith(image_extensions)])
    
#     if len(image_files) == 0:
#         print(f"❌ 在 {images_dir} 中没有找到图像文件")
#         return
    
#     print(f"\n📁 图像目录: {images_dir}")
#     print(f"📁 掩膜目录: {masks_dir}")
#     print(f"📊 找到 {len(image_files)} 个图像文件\n")
    
#     # 存储统计信息
#     image_shapes = []
#     image_dtypes = []
#     image_channels = []
#     image_min_vals = []
#     image_max_vals = []
#     image_mean_vals = []
#     image_std_vals = []
    
#     mask_shapes = []
#     mask_dtypes = []
#     mask_min_vals = []
#     mask_max_vals = []
#     mask_mean_vals = []
#     mask_unique_values = []
#     mask_positive_ratios = []
#     mask_values_list = []
    
#     matched_count = 0
#     mismatched_count = 0
#     missing_mask_count = 0
#     failed_mask_count = 0
    
#     # 收集所有像素用于整体统计
#     all_image_pixels = []
#     all_mask_pixels = []
    
#     # 分析每个图像
#     print("正在分析图像和掩膜...")
#     for idx, filename in enumerate(image_files):
#         if verbose:
#             print(f"\r处理进度: {idx+1}/{len(image_files)}", end='', flush=True)
        
#         # 读取图像
#         img_path = os.path.join(images_dir, filename)
#         image = cv2.imread(img_path)
        
#         if image is None:
#             print(f"\n⚠️ 无法读取图像: {filename}")
#             continue
        
#         # BGR转RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 图像统计
#         image_shapes.append(image_rgb.shape)
#         image_dtypes.append(image_rgb.dtype)
#         image_channels.append(image_rgb.shape[2] if len(image_rgb.shape) == 3 else 1)
#         image_min_vals.append(image_rgb.min())
#         image_max_vals.append(image_rgb.max())
#         image_mean_vals.append(image_rgb.mean())
#         image_std_vals.append(image_rgb.std())
        
#         # 收集图像像素
#         all_image_pixels.extend(image.flatten())
        
#         # 读取对应的掩膜
#         name_no_ext = os.path.splitext(filename)[0]
#         mask_filename = name_no_ext + '.png'
#         mask_path = os.path.join(masks_dir, mask_filename)
        
#         if os.path.exists(mask_path):
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
#             if mask is not None:
#                 matched_count += 1
                
#                 # 检查掩膜尺寸是否与图像一致
#                 if mask.shape != image_rgb.shape[:2]:
#                     mismatched_count += 1
#                     if verbose:
#                         print(f"\n⚠️ 尺寸不匹配: {filename} - 图像 {image_rgb.shape[:2]}, 掩膜 {mask.shape}")
                
#                 # 掩膜统计
#                 mask_shapes.append(mask.shape)
#                 mask_dtypes.append(mask.dtype)
#                 mask_min_vals.append(mask.min())
#                 mask_max_vals.append(mask.max())
#                 mask_mean_vals.append(mask.mean())
#                 mask_values_list.append(mask)
                
#                 # 获取唯一值
#                 unique_vals = np.unique(mask)
#                 mask_unique_values.append(unique_vals)
                
#                 # 收集掩膜像素
#                 all_mask_pixels.extend(mask.flatten())
                
#                 # 计算正样本比例
#                 if mask.max() > 0:
#                     positive_ratio = (mask > 0).sum() / mask.size
#                     mask_positive_ratios.append(positive_ratio)
#                 else:
#                     mask_positive_ratios.append(0.0)
#             else:
#                 failed_mask_count += 1
#                 if verbose:
#                     print(f"\n⚠️ 无法读取掩膜: {mask_filename}")
#         else:
#             missing_mask_count += 1
#             if verbose:
#                 print(f"\n⚠️ 掩膜文件不存在: {mask_filename}")
    
#     if verbose:
#         print("\n")
    
#     # ========== 输出统计结果 ==========
#     print("\n" + "=" * 80)
#     print("📊 统计分析结果")
#     print("=" * 80)
    
#     # 图像统计
#     print("\n【图像统计】")
#     print("-" * 40)
#     if image_shapes:
#         # 图像尺寸
#         shapes_unique = Counter(image_shapes)
#         print(f"📐 图像尺寸分布:")
#         for shape, count in shapes_unique.most_common():
#             print(f"   {shape}: {count} 张 ({count/len(image_files)*100:.1f}%)")
        
#         # 通道数
#         channels_unique = Counter(image_channels)
#         print(f"\n🎨 通道数分布:")
#         for channels, count in channels_unique.most_common():
#             print(f"   {channels} 通道: {count} 张")
        
#         # 数据类型
#         dtypes_unique = Counter(image_dtypes)
#         print(f"\n💾 数据类型分布:")
#         for dtype, count in dtypes_unique.most_common():
#             print(f"   {dtype}: {count} 张")
        
#         # 像素值范围
#         print(f"\n📊 像素值统计:")
#         print(f"   最小值范围: {min(image_min_vals):.2f} ~ {max(image_min_vals):.2f}")
#         print(f"   最大值范围: {min(image_max_vals):.2f} ~ {max(image_max_vals):.2f}")
#         print(f"   平均值范围: {min(image_mean_vals):.2f} ~ {max(image_mean_vals):.2f}")
#         print(f"   标准差范围: {min(image_std_vals):.2f} ~ {max(image_std_vals):.2f}")
        
#         # 整体统计
#         if all_image_pixels:
#             all_image_pixels = np.array(all_image_pixels)
#             print(f"\n💡 整体像素值统计 (所有图像):")
#             print(f"   最小值: {all_image_pixels.min()}")
#             print(f"   最大值: {all_image_pixels.max()}")
#             print(f"   平均值: {all_image_pixels.mean():.2f}")
#             print(f"   标准差: {all_image_pixels.std():.2f}")
#             print(f"   中位数: {np.median(all_image_pixels):.2f}")
            
#             # 直方图分布
#             print(f"\n📊 像素值分布区间:")
#             hist, bin_edges = np.histogram(all_image_pixels, bins=10)
#             for i in range(len(hist)):
#                 print(f"   [{bin_edges[i]:.0f}, {bin_edges[i+1]:.0f}): {hist[i]} 像素 ({hist[i]/len(all_image_pixels)*100:.2f}%)")
    
#     # 掩膜统计
#     print("\n【掩膜统计】")
#     print("-" * 40)
#     if mask_shapes:
#         # 掩膜尺寸
#         mask_shapes_unique = Counter(mask_shapes)
#         print(f"📐 掩膜尺寸分布:")
#         for shape, count in mask_shapes_unique.most_common():
#             print(f"   {shape}: {count} 个")
        
#         # 掩膜数据类型
#         mask_dtypes_unique = Counter(mask_dtypes)
#         print(f"\n💾 掩膜数据类型分布:")
#         for dtype, count in mask_dtypes_unique.most_common():
#             print(f"   {dtype}: {count} 个")
        
#         # 像素值范围
#         if mask_min_vals and mask_max_vals:
#             print(f"\n📊 像素值统计:")
#             print(f"   最小值范围: {min(mask_min_vals):.2f} ~ {max(mask_min_vals):.2f}")
#             print(f"   最大值范围: {min(mask_max_vals):.2f} ~ {max(mask_max_vals):.2f}")
#             print(f"   平均值范围: {min(mask_mean_vals):.2f} ~ {max(mask_mean_vals):.2f}")
        
#         # 唯一值分析
#         all_unique = set()
#         for vals in mask_unique_values:
#             all_unique.update(vals)
#         print(f"\n🔢 所有掩膜的唯一值: {sorted(all_unique)}")
        
#         # 统计每个唯一值的出现频率
#         if all_mask_pixels:
#             all_mask_pixels = np.array(all_mask_pixels)
#             unique, counts = np.unique(all_mask_pixels, return_counts=True)
#             print(f"\n📊 掩膜像素值分布:")
#             for val, count in zip(unique, counts):
#                 percentage = count / len(all_mask_pixels) * 100
#                 print(f"   值 {val}: {count} 像素 ({percentage:.2f}%)")
        
#         # 正样本统计
#         if mask_positive_ratios:
#             positive_ratios = np.array(mask_positive_ratios)
#             positive_count = sum(1 for r in mask_positive_ratios if r > 0)
#             print(f"\n📍 正样本统计:")
#             print(f"   包含正样本的图像数: {positive_count}/{len(mask_positive_ratios)} ({positive_count/len(mask_positive_ratios)*100:.1f}%)")
#             print(f"   正样本像素比例:")
#             print(f"     平均值: {positive_ratios.mean():.4f}")
#             print(f"     最小值: {positive_ratios.min():.6f}")
#             print(f"     最大值: {positive_ratios.max():.4f}")
#             print(f"     标准差: {positive_ratios.std():.4f}")
#             print(f"     中位数: {np.median(positive_ratios):.4f}")
    
#     # 数据匹配统计
#     print("\n【数据匹配统计】")
#     print("-" * 40)
#     print(f"✅ 成功匹配的图像-掩膜对: {matched_count}")
#     print(f"❌ 尺寸不匹配的对: {mismatched_count}")
#     print(f"⚠️  读取失败的掩膜: {failed_mask_count}")
#     print(f"⚠️  缺少掩膜的文件: {missing_mask_count}")
#     print(f"📊 匹配率: {matched_count/len(image_files)*100:.1f}%")
    
#     print("\n" + "=" * 80)
#     print("✅ 分析完成!")
#     print("=" * 80)
    
#     return {
#         'image_files': image_files,
#         'image_stats': {
#             'shapes': image_shapes,
#             'dtypes': image_dtypes,
#             'channels': image_channels,
#             'min_vals': image_min_vals,
#             'max_vals': image_max_vals,
#             'mean_vals': image_mean_vals,
#             'std_vals': image_std_vals
#         },
#         'mask_stats': {
#             'shapes': mask_shapes,
#             'dtypes': mask_dtypes,
#             'min_vals': mask_min_vals,
#             'max_vals': mask_max_vals,
#             'mean_vals': mask_mean_vals,
#             'unique_values': mask_unique_values,
#             'positive_ratios': mask_positive_ratios
#         }
#     }


# def main():
#     parser = argparse.ArgumentParser(description='分析PNG数据集')
#     parser.add_argument('--images_dir', type=str, 
#                        default='/home/Liujiali/DomainData/004/train/images',
#                        help='图像文件夹路径')
#     parser.add_argument('--masks_dir', type=str,
#                        default='/home/Liujiali/DomainData/004/train/masks',
#                        help='掩膜文件夹路径')
#     parser.add_argument('--verbose', action='store_true',
#                        help='显示详细处理信息')
    
#     args = parser.parse_args()
    
#     # 检查路径是否存在
#     if not os.path.exists(args.images_dir):
#         print(f"❌ 图像目录不存在: {args.images_dir}")
#         return
    
#     if not os.path.exists(args.masks_dir):
#         print(f"❌ 掩膜目录不存在: {args.masks_dir}")
#         return
    
#     # 运行分析
#     analyze_png_dataset(args.images_dir, args.masks_dir, args.verbose)


# if __name__ == "__main__":
#     main()




# #!/usr/bin/env python3
# """
# PNG数据检查脚本
# 用于查看 /home/Liujiali/DomainData/004/train/ 目录下的图像和标签数据
# """

# import os
# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt

# def check_png_data():
#     """检查PNG格式的图像和标签数据"""
    
#     # 数据路径
#     images_dir = '/home/Liujiali/DomainData/004/train/images'
#     masks_dir = '/home/Liujiali/DomainData/004/train/masks'
    
#     print("=" * 70)
#     print("PNG数据检查报告")
#     print("=" * 70)
    
#     # 检查目录是否存在
#     if not os.path.exists(images_dir):
#         print(f"❌ 图像目录不存在: {images_dir}")
#         return
#     if not os.path.exists(masks_dir):
#         print(f"❌ 标签目录不存在: {masks_dir}")
#         return
    
#     # 获取文件列表
#     image_files = sorted([f for f in os.listdir(images_dir) 
#                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
#     mask_files = sorted([f for f in os.listdir(masks_dir) 
#                         if f.lower().endswith('.png')])
    
#     print(f"📁 图像文件数量: {len(image_files)}")
#     print(f"📁 标签文件数量: {len(mask_files)}")
#     print("-" * 70)
    
#     if len(image_files) == 0:
#         print("❌ 没有找到图像文件")
#         return
    
#     # 检查前5个样本（或全部，如果少于5个）
#     num_samples = min(5, len(image_files))
    
#     for i in range(num_samples):
#         img_name = image_files[i]
#         name_no_ext = os.path.splitext(img_name)[0]
#         mask_name = name_no_ext + '.png'
        
#         print(f"\n📷 样本 {i+1}: {img_name}")
#         print("-" * 50)
        
#         # ====== 检查图像 ======
#         img_path = os.path.join(images_dir, img_name)
        
#         # 1. 使用cv2读取
#         img_cv2 = cv2.imread(img_path)
#         if img_cv2 is None:
#             print(f"   ❌ cv2无法读取图像")
#             continue
        
#         print(f"   【cv2.imread】")
#         print(f"      shape: {img_cv2.shape}")
#         print(f"      dtype: {img_cv2.dtype}")
#         print(f"      值域: [{img_cv2.min()}, {img_cv2.max()}]")
#         print(f"      通道顺序: BGR (cv2默认)")
        
#         # 2. 使用PIL读取
#         img_pil = Image.open(img_path)
#         img_pil_array = np.array(img_pil)
#         print(f"\n   【PIL.Image读取】")
#         print(f"      shape: {img_pil_array.shape}")
#         print(f"      dtype: {img_pil_array.dtype}")
#         print(f"      值域: [{img_pil_array.min()}, {img_pil_array.max()}]")
#         print(f"      通道顺序: RGB (PIL默认)")
#         print(f"      图像模式: {img_pil.mode}")
        
#         # 3. 对比cv2和PIL的差异
#         if img_cv2.shape == img_pil_array.shape:
#             # BGR vs RGB 差异检查
#             cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
#             diff = np.abs(cv2_rgb.astype(np.float32) - img_pil_array.astype(np.float32))
#             print(f"\n   【cv2(BGR→RGB) vs PIL】")
#             print(f"      最大差异: {diff.max():.1f}")
#             print(f"      平均差异: {diff.mean():.2f}")
#             if diff.max() == 0:
#                 print(f"      ✅ cv2(BGR→RGB) 与 PIL 完全一致")
#             else:
#                 print(f"      ⚠️ 存在差异，需检查通道顺序")
        
#         # ====== 检查标签 ======
#         mask_path = os.path.join(masks_dir, mask_name)
#         if not os.path.exists(mask_path):
#             print(f"\n   ⚠️ 标签文件不存在: {mask_name}")
#             continue
        
#         # 1. 使用cv2读取标签
#         mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask_cv2 is None:
#             print(f"   ❌ cv2无法读取标签")
#             continue
        
#         print(f"\n   【标签 - cv2.imread(IMREAD_GRAYSCALE)】")
#         print(f"      shape: {mask_cv2.shape}")
#         print(f"      dtype: {mask_cv2.dtype}")
#         print(f"      值域: [{mask_cv2.min()}, {mask_cv2.max()}]")
#         unique_values = np.unique(mask_cv2)
#         print(f"      唯一值: {unique_values}")
        
#         # 统计类别分布
#         total_pixels = mask_cv2.size
#         for val in unique_values:
#             count = np.sum(mask_cv2 == val)
#             ratio = count / total_pixels * 100
#             print(f"         值 {val}: {count} 像素 ({ratio:.2f}%)")
        
#         # 2. 使用PIL读取标签
#         mask_pil = Image.open(mask_path)
#         mask_pil_array = np.array(mask_pil)
#         print(f"\n   【标签 - PIL.Image读取】")
#         print(f"      shape: {mask_pil_array.shape}")
#         print(f"      dtype: {mask_pil_array.dtype}")
#         print(f"      值域: [{mask_pil_array.min()}, {mask_pil_array.max()}]")
#         unique_values_pil = np.unique(mask_pil_array)
#         print(f"      唯一值: {unique_values_pil}")
        
#         # 3. 检查图像和标签尺寸是否一致
#         print(f"\n   【尺寸匹配检查】")
#         print(f"      图像尺寸: {img_cv2.shape[:2]}")
#         print(f"      标签尺寸: {mask_cv2.shape}")
#         if img_cv2.shape[:2] == mask_cv2.shape:
#             print(f"      ✅ 图像和标签尺寸一致")
#         else:
#             print(f"      ❌ 尺寸不一致! 需要resize")
        
#         # 4. 检查标签是否需要二值化
#         if len(unique_values) == 2:
#             if set(unique_values) == {0, 1}:
#                 print(f"\n   ✅ 标签已经是二值化 (0/1)")
#             elif set(unique_values) == {0, 255}:
#                 print(f"\n   ⚠️ 标签值是 0/255，需要二值化 (mask > 127)")
#             else:
#                 print(f"\n   ⚠️ 标签值: {unique_values}，需要检查")
#         else:
#             print(f"\n   ⚠️ 标签不是二值，有 {len(unique_values)} 个不同的值")
#             print(f"      可能是多分类标签，需要确认")
    
#     # ====== 统计所有样本的标签类别 ======
#     print("\n" + "=" * 70)
#     print("所有样本标签统计")
#     print("=" * 70)
    
#     all_unique_values = set()
#     for mask_name in mask_files[:min(10, len(mask_files))]:
#         mask_path = os.path.join(masks_dir, mask_name)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is not None:
#             all_unique_values.update(np.unique(mask).tolist())
    
#     print(f"所有标签中出现的唯一值: {sorted(all_unique_values)}")
    
#     if all_unique_values == {0, 1}:
#         print("✅ 所有标签都是 0/1 二值化格式")
#     elif all_unique_values == {0, 255}:
#         print("⚠️ 所有标签都是 0/255 格式，需要二值化 (mask > 0 或 mask > 127)")
#     elif all_unique_values == {0, 255, 127} or len(all_unique_values) > 2:
#         print("⚠️ 标签包含多个值，可能是多分类或标注问题")
    
#     # ====== 总结建议 ======
#     print("\n" + "=" * 70)
#     print("📝 总结与建议")
#     print("=" * 70)
    
#     print("\n【代码加载逻辑对比】")
#     print("  代码1 (NpyDataset):")
#     print("    - cv2.imread → BGR转RGB → transpose(2,0,1) → (3,H,W)")
#     print("    - 归一化: /255.0")
#     print("    - 标签: (mask > 127).astype(float32)")
#     print("\n  代码2 (LandslideTarget):")
#     print("    - cv2.imread → BGR转RGB → 保持 (H,W,3)")
#     print("    - 归一化: ToTensor() (/255.0)")
#     print("    - 标签: (mask > 0).astype(uint8)")
    
#     print("\n⚠️ 需要你根据上述数据检查结果判断:")
#     print("  1. 图像的通道顺序是否正确 (BGR vs RGB)?")
#     print("  2. 标签的真实值是什么 (0/1 还是 0/255)?")
#     print("  3. 是否需要转置为 (C,H,W) 还是保持 (H,W,C)?")
    
#     print("\n" + "=" * 70)
#     print("✅ 检查完成")
#     print("=" * 70)

# if __name__ == "__main__":
#     check_png_data()









import os
import shutil
from sklearn.model_selection import train_test_split

# 设置路径
img_dir = '/home/Liujiali/DomainData/palu/img'
mask_dir = '/home/Liujiali/DomainData/palu/mask'
output_dir = '/home/Liujiali/DomainData/palu'

# 创建输出目录
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

# 获取所有图像文件和mask文件
img_files = sorted([f for f in os.listdir(img_dir) 
                    if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])

print(f"找到 {len(img_files)} 个图像文件")

# 检查对应的mask是否存在
valid_files = []
for f in img_files:
    mask_path = os.path.join(mask_dir, f)
    if os.path.exists(mask_path):
        valid_files.append(f)
    else:
        print(f"⚠️ 警告: {f} 没有对应的mask文件，已跳过")

print(f"有效配对文件数: {len(valid_files)}")

# 划分数据集: 8:1:1
train_files, temp_files = train_test_split(
    valid_files, 
    test_size=0.2,  # 20%留给验证集和测试集
    random_state=42
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,  # 10%测试集, 10%验证集
    random_state=42
)

print(f"\n📊 划分结果:")
print(f"  训练集: {len(train_files)} 个文件 (80%)")
print(f"  验证集: {len(val_files)} 个文件 (10%)")
print(f"  测试集: {len(test_files)} 个文件 (10%)")

# 复制文件函数（使用copy2保留元数据）
def copy_files(file_list, split_name):
    for fname in file_list:
        # 复制图像到新目录
        src_img = os.path.join(img_dir, fname)
        dst_img = os.path.join(output_dir, split_name, 'images', fname)
        shutil.copy2(src_img, dst_img)  # copy2保留文件元数据
        
        # 复制对应的mask到新目录
        src_mask = os.path.join(mask_dir, fname)
        dst_mask = os.path.join(output_dir, split_name, 'masks', fname)
        shutil.copy2(src_mask, dst_mask)
    
    print(f"✅ {split_name}: 复制了 {len(file_list)} 对文件")

# 执行复制
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("\n" + "="*60)
print("✅ 数据集划分完成！")
print("="*60)
print(f"📁 原始数据保留在:")
print(f"   {img_dir}")
print(f"   {mask_dir}")
print(f"\n📁 划分后的数据保存在:")
print(f"   {output_dir}/train/images/  (训练集图像)")
print(f"   {output_dir}/train/masks/   (训练集标签)")
print(f"   {output_dir}/val/images/    (验证集图像)")
print(f"   {output_dir}/val/masks/     (验证集标签)")
print(f"   {output_dir}/test/images/   (测试集图像)")
print(f"   {output_dir}/test/masks/    (测试集标签)")
print("="*60)