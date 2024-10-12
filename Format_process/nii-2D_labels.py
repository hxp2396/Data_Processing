#coding=utf-8
# 2、针对多个Nii的切片
import SimpleITK as sitk
import skimage.io as io
import sys, os
import numpy as np
from tqdm import tqdm

# 例子：BraTS_2018数据集（MRI）中的标签：0是背景、1是坏疽、2是浮肿、4是增强肿瘤区
# 选择标签值，输出相应的标签切片图
LABEL_NUM =1

# CT 加窗调对比度（hu值）
IF_WINDOWS = 1  # 0是不加窗  ； 1 是加窗
WINDOW_WEIGHT =1500
WINDOW_LONGTH = -500

# nii路径'  './HGG'
NII_IN_DIR_PATH = './test/test_images'
# NII_IMAGES_IN_DIR_PATH =
# NII_LABELS_IN_DIR_PATH = './HGG'

# 图片输出路径
IMAGE_OUT_PATH = './test/Images'
LABEL_OUT_PATH = './test/Masks'
os.makedirs(IMAGE_OUT_PATH,exist_ok=True)
os.makedirs(LABEL_OUT_PATH,exist_ok=True)

np.set_printoptions(threshold=np.inf)


# index = -1

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data


def find_label_image(label_data, all_one_array, torch_CT, niidir_one):
    index = -1
    print(label_data.shape)
    for i in tqdm(range(label_data.shape[0])):  # 遍历所有切片
        true_sum = np.sum(label_data[i] == LABEL_NUM * all_one_array[i])  # 找到有标签为label_lum的切片。
        if true_sum >= 0:  # 没有打该标签的不生成图片，避免类别不平衡问题。
            index += 1
            file_dir1 = os.path.join(LABEL_OUT_PATH, niidir_one + '_' + str(index) + '.png')  # 标签图片的路径
            file_dir2 = os.path.join(IMAGE_OUT_PATH, niidir_one + '_' + str(index) + '.png')  # 数据图片的路径niniinii-2D_labels.py:46-2D_labels.py:46inii-2D_labels.py:46-2D_labels.py:46
            # print(i)

            # 二值化label
            label_img = np.zeros(label_data[i].shape, dtype=np.uint8)
            for x in range(label_img.shape[0]):
                for y in range(label_img.shape[1]):
                    if label_data[i][x, y] >0 :
                        label_img[x, y] = 255;

            # 归一化image到[0,1]范围内,否则无法io.imsave(报错:Images of type float must be between -1 and 1)
            a = 0
            b = 1
            Ymax = np.max(torch_CT[i])
            Ymin = np.min(torch_CT[i])
            k = (b - a) / (Ymax - Ymin)
            norY = a + k * (torch_CT[i] - Ymin)

            # 保存带有label的image.png
            io.imsave(file_dir1, label_img)
            io.imsave(file_dir2, norY)


def _img_transfor(itk):
    img_arr = sitk.GetArrayFromImage(itk).astype(np.float32)
    if IF_WINDOWS:
        img_arr = window_normalize(img_arr, WW=1600, WL=-600)
    # torch_itk = torch.from_numpy(img_arr)
    return img_arr


def window_normalize(img, WW, WL, dst_range=(0, 1)):
    """
    WW: window width
    WL: window level
    dst_range: normalization range
    """
    src_min = WL - WW / 2
    src_max = WL + WW / 2
    outputs = (img - src_min) / WW * (dst_range[1] - dst_range[0]) + dst_range[0]
    outputs[img >= src_max] = 1
    outputs[img <= src_min] = 0
    return outputs


def main():
    # 判断是否有输入数据目录，没有则报错
    assert os.path.exists(NII_IN_DIR_PATH)

    # 判断输出路径是否存在，若不存在，则创建一个目录
    if not os.path.exists(IMAGE_OUT_PATH):
        os.mkdir(IMAGE_OUT_PATH)
    if not os.path.exists(LABEL_OUT_PATH):
        os.mkdir(LABEL_OUT_PATH)

    # 遍历所有样例
    niidir_list = os.listdir(NII_IN_DIR_PATH)  # 获取子文件（包括目录）
    # print(niidir_list)
    niidir_list_len = len(niidir_list)  # 获取子文件（包括目录）的数量
    niidir_list_count = 0
    for niidir_one in tqdm(niidir_list):
        # print(niidir_one)
        nii_label_path =(NII_IN_DIR_PATH.replace("test_images","test_labels")+ '/'+niidir_one)
        nii_image_path = NII_IN_DIR_PATH + '/' + niidir_one
        # print(nii_label_path)
        # nii_image_path = NII_IN_DIR_PATH + '/' + niidir_one
        #
        label_filename = nii_label_path
        train_filename = nii_image_path

        label_data = read_img(label_filename)
        train_data = read_img(train_filename)

        # 生成与label维度一样的array，且元素都为1.
        like_label_one_array = np.ones_like(label_data)

        itk_gt = sitk.ReadImage(label_filename)
        itk_CT = sitk.ReadImage(train_filename)

        torch_CT = _img_transfor(itk_CT)

        find_label_image(label_data, like_label_one_array, torch_CT, niidir_one)


if __name__ == "__main__":
    main()