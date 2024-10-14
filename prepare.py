# coding = utf - 8
from __future__ import division
from glob import glob
import cv2
import h5py
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import SimpleITK as sitk
import skimage.io as io
import sys, os


def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x;
def prepare():
    spacing_list = []

    for file in tqdm(os.listdir('img')):

        ct = sitk.ReadImage(os.path.join('img', file))
        data=sitk.GetArrayFromImage(ct)

        # 将灰度值在阈值之外的截断掉
        data_clipped = np.clip(data, -125, 275)
        #归一化
        # data = (data - np.mean(data)) / np.std(data)  # 图像归一化，减去均值除于方差(-1,1)
        max=np.max(data)
        min=np.min(data)
        data =MaxMinNormalization(data,max,min)
        new_ct=sitk.GetImageFromArray(data)
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())
        new_ct.SetDirection(ct.GetDirection())
        sitk.WriteImage(new_ct,os.path.join('new_ct', file))
def To_h5():
    # 像素映射
    # dicts={'0':0,'1':1,'2':2,'3':3,'4':4,'5':0,'6':5,'7':6,'8':7,'9':0,'10':0,'11':8,'12':0,'13':0}
    dir='ACDC/test/test_images'
    dir1 = 'ACDC/test/test_labels'
    ct_list=os.listdir(dir)
    print(ct_list)
    for ct_file in ct_list:
        ct=sitk.ReadImage(os.path.join(dir,ct_file))
        ct_array=sitk.GetArrayFromImage(ct)
        mask=sitk.ReadImage(os.path.join(dir1,ct_file))
        mask_array=sitk.GetArrayFromImage(mask)
        # c,h,w=mask_array.shape
        # for i in range(0,c):
        #     for j in range(0, h):
        #         for k in range(0, w):
        #             mask_array[i][j][k]=dicts[str(mask_array[i][j][k])]

        # print(mask_array.shape)
        # print(os.path.join('label',ct_file.replace('img','label')))
        print(np.unique(mask_array))
        f=h5py.File(ct_file.replace('.nii.gz','.npy.h5'),'w')
        f.create_dataset('image', data=ct_array, compression="gzip")
        f.create_dataset('label', data=mask_array, compression="gzip")
        f.close()

def To_Slice():

    dir='ACDC/train/train_images/'
    slice_list=os.listdir(dir)
    # print(slice_list)
    for img_file in slice_list:
        img_path=os.path.join(dir,img_file)
        label_path=os.path.join('label',img_file)
        # 读取image和label数据
        img_data = cv2.imread(img_path,0)  # 将图片读为数组
        # img_data = cv2.resize(img_data, (64, 64))  # 调整图像大小
        label_data=  cv2.imread(label_path,0)# 将图片标签与数值相对应，这里的标签名就是文件夹的名字
        # analyze(label_data)
        max_pix = np.amax(label_data)
        if max_pix!=0:
            print(max_pix)
            label_data = (label_data /max_pix)# 归一化
            analyze(label_data)
        # label_train = label_train * 255
        # 存储image和label数据
        np.savez(img_file.replace('.png',''),image=img_data,label=label_data)
def gene_txt():
    npz_list=glob('ACDC/h5/*.h5')
    print(npz_list)
    with open('train.txt',"w") as f:
        for npz_filename in npz_list:
            f.write(npz_filename.replace('.h5','')+'\n')




import scipy, numpy, shutil, os, nibabel
import sys, getopt
import imageio
import numpy as np
import SimpleITK as sitk
'''
SimpleITK 和 Nibabel 区别在于：（nii图像可以看成2维，也可以看成三维）
SimpleITK读取数据是（X，Y，Z）显示，Nibabel读取图像是（Z，Y，X）显示，也就是Nibabel加载的图像会旋转90°，其中X表示通道数，即切片层数。详情
想要批量抽取MRI数据的几个切片，借鉴了上述代码的Python版本
，原始代码抽取了全部切片。本次实验主要是抽取sMRI的横断面（axial）方向的其中30张切片，
数量根据自己需求可自行设置。'''
def niito2D():
    dicts = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 0, '6': 5, '7': 6, '8': 7, '9': 0, '10': 0, '11': 8, '12': 0,
             '13': 0}
    dir='ACDC/train/train_images/'
    dir1='ACDC/train/train_labels/'
    image_list = os.listdir(dir)  # 遍历文件夹数据
    print(image_list)
    for img_file in image_list:
        ct=sitk.ReadImage(os.path.join(dir,img_file))
        image_array=sitk.GetArrayFromImage(ct)
        label = sitk.ReadImage(os.path.join(dir1,img_file))
        label_array = sitk.GetArrayFromImage(label)
        # print(label_array.shape)
        # c, h, w = label_array.shape
        # for i in range(0, c):
        #     for j in range(0, h):
        #         for k in range(0, w):
        #             label_array[i][j][k] = dicts[str(label_array[i][j][k])]
        # print(np.unique(label_array))
        # image_array = nibabel.load(os.path.join(dir,img_file)).get_data()  # 数据读取
        # label_array=nibabel.load(os.path.join('label',img_file.replace('img','label'))).get_data()  # 数据读取

        total_slices = image_array.shape[0]  # 总切片数
        slice_counter = 0  # 从第几个切片开始

        # iterate through slices
        for current_slice in range(0,  total_slices):
            # alternate slices
            if (slice_counter % 1) == 0:
                image_data = image_array[current_slice,:, : ]  # 保存该切片，可以选择不同方向。
                label_data=label_array[current_slice,:, :]
                # print(label_data.shape)
                print(np.unique(label_data))

                # alternate slices and save as png
                if (slice_counter % 1) == 0:
                    print('Saving image...')
                    # 切片命名
                    image_name = img_file+str(current_slice)+'.png'
                    labe_name=image_name
                    # 保存
                    # data=  data.astype(np.uint8)

                    np.savez(img_file.split('.')[0]+'_'+str(current_slice), image=image_data, label=label_data)
                    # imageio.imwrite('image/'+image_name, image_data)
                    # imageio.imwrite('label/'+labe_name, label_data)
                    print('Saved.')
                    slice_counter += 1
    # print('Finished converting images')
def analyze(img_arr):
    arr=img_arr
    shap=arr.shape


        # 查看标签内像素种类

    num=[]
    for i in range(0, shap[0]):
        for j in range(0, shap[1]):
                if arr[i][j] not in num:
                    num.append(arr[i][j])
                    # print(num)

    print(len(num))
def find_class():
    from PIL import Image

    # dict={'0':0,'227'}
    filelist=glob('lb/*.png')
    num = []
    print(filelist)
    for file in filelist:
        arr=cv2.imread(file,0)
        # print(arr.shape)
        # max_pix = np.amax(arr)
        # label_train = arr / max_pix  # 归一化
        # label = Image.fromarray(label_train * 255)
        #
        #
        # label = label.convert("L")
        # print(label.size)

        # print(arr.shape)

        shape=arr.shape

        #查看标签内像素种类
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                if arr[i][j] not in num:
                    num.append(arr[i][j])
                    print(num)
        # imageio.imwrite(file.replace('label','lb'), arr)
    print(num)
if __name__=="__main__":



    # prepare()
    # To_h5()
    # niito2D()
    # To_Slice()
    gene_txt()

    # find_class()
    
    
#     imgpath='label/label0036.nii.gz'
#     arr = sitk.GetArrayFromImage(sitk.ReadImage(imgpath,sitk.sitkInt16))
#     print(arr.shape)
#     shap=arr.shape
#
#
#     # 查看标签内像素种类
#
#     num=[]
#     for i in range(0, shap[0]):
#         for j in range(0, shap[1]):
#             for k in range(0, shap[2]):
#                 if arr[i][j][k] not in num:
#                     num.append(arr[i][j][k])
#                     print(num)
#
# print(num)


