import argparse
import time

import h5py
import numpy
import numpy as np
import cv2
import scipy.misc
from PIL import Image
#--------------------------------将图片更换格式存储---------------------------------
# from PIL import Image
# import os
# from glob import glob
#
# filelist=glob(r'ultra_nerve/Masks/*.tif')
# for name in filelist:
#     img=cv2.imread(name)
#     filename=os.path.split(name)[-1].replace('tif','png')
#     cv2.imwrite(filename,img)
    # print(filename)

# path=r'C:\Users\hxp\Desktop\DataCollection\TCGA-2Z-A9J9-01A-01-TS1.png'
# img=cv2.imread(path)
# cv2.imwrite('TCGA-2Z-A9J9-01A-01-TS1.png',img)
# img=Image.open(path)
# img.show()
# # print(np.unique(img))


#--------------------------------------提取图片轮廓--------------------------------------
# from PIL.ImageShow import show
#
# img = cv2.imread(r'C:\Users\hxp\Desktop\DataCollection\study_0255_0031.png')
#
# kernel = np.ones((5, 5), dtype=np.uint8)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 1)
# ss = np.hstack((img, opening))
#
#
# cv2.imwrite('seg_smooth.png', opening)
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
#
#
# img = cv2.imread(r'C:\Users\hxp\Desktop\DataCollection\study_0255_0031.png')
# kernel = np.ones((3, 3), dtype=np.uint8)
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# ss = np.hstack((img, gradient))
# cv2.imwrite('seg_gr.png', ss)

# ----------------------------------npy转图片与切片--------------------------------------
import numpy as np
import os
import cv2
from glob import glob

# path='Synapse/train_npz/'
# save_img_path='Synapse/pic/Images/'
# save_label_path='Synapse/pic/Masks/'
# os.makedirs(save_img_path,exist_ok=True)
# os.makedirs(save_label_path,exist_ok=True)
# filelists=glob(path+'*.npz')
# for file in filelists:
#     name=os.path.split(file)[-1].replace('npz','png')
#     dd=np.load(file)
#     img,label=dd['image'],dd['label']
#     print(save_img_path+name)
#     scipy.misc.imsave(save_img_path+name,img)
#     cv2.imwrite(save_label_path+name, label)


# path=r'C:\Users\hxp\Desktop\DataCollection\Synapse\pic\Masks\case0040_slice120.png'
# img=cv2.imread(path)
# print(np.unique(img))
#--------------------------------------3d切片----------------------------------------------------------
# import numpy as np
# import os
# import cv2
# from glob import glob
#
# path='Synapse/test_vol_h5/'
# save_img_path='Synapse/test/Images/'
# save_label_path='Synapse/test/Masks/'
# os.makedirs(save_img_path,exist_ok=True)
# os.makedirs(save_label_path,exist_ok=True)
# filelists=glob(path+'*.h5')
# # print(filelists)
# for file in filelists:
#     name=os.path.split(file)[-1].replace('.npy.h5','')
#     # print(name)
#     h5=h5py.File(file,'r')
#     img,label=h5['image'].value,h5['label'].value
#     assert img.shape==label.shape
#     c,h,w=img.shape
#     for i in range(0,c):
#         im=img[i,:,:]
#         la=label[i,:,:]
#         new_name=name+'_slice_'+str(i)+'.png'
#         # print(new_name)
#         save_img_pth=save_img_path+new_name
#         save_label_pth = save_label_path + new_name
#         scipy.misc.imsave(save_img_pth,im)
#         cv2.imwrite(save_label_pth, la)


    # print(save_img_path+name)
    # scipy.misc.imsave(save_img_path+name,img)
    # cv2.imwrite(save_label_path+name, label)
# -------------------------颜色反转-----------------------------------------------------------------------------------

# import cv2
# import numpy as np
# path=r'radiopaedia_7_85703_0.nii.gz_32.png'
# img=cv2.imread(path,0)
# h,w=img.shape
# for i in range(0,h):
#     for j in range(0,w):
#         if img[i][j]==0:
#             img[i][j] = 255
#         elif img[i][j]==255:
#             img[i][j] = 0
#         else:
#             pass
# cv2.imwrite('new.png',img)


# print(np.unique(img))

# --------------------绘制散点图，柱状图，折线图---------------------------------------------------------------------------
# import numpy as np
# from matplotlib import pyplot as plt
#
# plt.figure(figsize=(12, 9))
# n = 4
# X = np.arange(n) + 1
#
# # X是1,2,3,4,5,6,7,8,柱的个数
# # numpy.random.uniform(low=0.0, high=1.0, size=None), normal
# # uniform均匀分布的随机数，normal是正态分布的随机数，0.5-1均匀分布的数，一共有n个
# Y1 = [1,2,3,4]
#
# Y2 = np.random.uniform(0.5, 1.0, n)
# Y3 = np.random.uniform(0.5, 1.0, n)
# plt.bar(X, Y1, width=0.25, facecolor='lightskyblue', edgecolor='white')
# # width:柱的宽度
# plt.bar(X + 0.25, Y2, width=0.25, facecolor='yellowgreen', edgecolor='white')
# plt.bar(X + 0.5, Y3, width=0.25, facecolor='purple', edgecolor='white')
# # 水平柱状图plt.barh，属性中宽度width变成了高度height
# # 打两组数据时用+
# # facecolor柱状图里填充的颜色
# # edgecolor是边框的颜色
# # 想把一组数据打到下边，在数据前使用负号
# # plt.bar(X, -Y2, width=width, facecolor='#ff9999', edgecolor='white')
# # 给图加text
# for x, y in zip(X, Y1):
#     plt.text(x + 0.0, y + 0.01, '%.2f' % y, ha='center', va='bottom')
#
# for x, y in zip(X, Y2):
#     plt.text(x + 0.25, y + 0.01, '%.2f' % y, ha='center', va='bottom')
# for x, y in zip(X, Y3):
#     plt.text(x + 0.5, y + 0.01, '%.2f' % y, ha='center', va='bottom')
# plt.ylim(0, +1.25)
# plt.show()


# 1
# (mean=(0.3589057922363281,),std=(0.220982164144516,))#train
# (mean=(0.35954394936561584,), std=(0.22119909524917603,))#valid
# (mean=(0.35319095849990845,), std=(0.21809816360473633,))#test
# 2
# (mean=(0.56125474,), std=(0.23397951,))  # train
# (mean=(0.5627737,), std=(0.23380543,))  # valid
# (mean=(0.5587516,), std=(0.23418126,))  # test

#-------------------标签显化-------------------------------------------------------------

# import os
# from glob import glob
# path=r'C:\Users\hxp\Desktop\DataCollection\TCGA-2Z-A9J9-01A-01-TS1.png'
# filelist=glob(r'E:\DataCollection\Synapse\test\Masks\*.png')
# # print(filelist)
# for name in filelist:
#     img=cv2.imread(name)
#     img=img*31.875
#     filename=os.path.split(name)[-1]
#     cv2.imwrite('test_mask/'+filename,img)
#     # print(filename)
#
# path=r'E:\DataCollection\pred\case0001_slice_75.png'
# img=cv2.imread(path,0)
# print(np.unique(img))
# --------------------#计算Dice-----------------------------------------------------------------

# def cal_dice(seg, gt, classes=9, background_id=0):
#     channel_dice = []
#     a=np.array(np.unique(seg))
#     b=np.array(np.unique(gt))
#     c=(a==b)
#
#     if len(np.unique(seg))==len(np.unique(gt)) and len(np.unique(seg))==1 and c:
#         channel_dice.append(1)
#     else:
#
#         for i in range(classes):
#
#
#             if i == background_id:
#                 continue
#             if i not in np.unique(seg) and i not in np.unique(gt):
#                 continue
#             else:
#                 cond = i ** 2
#                 # 计算相交部分
#                 inter = len(np.where(seg * gt == cond)[0])
#                 total_pix = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0])
#                 if total_pix == 0:
#                     dice = 0
#                 else:
#                     dice = (2 * inter) / total_pix
#             # print(dice)
#             channel_dice.append(dice)
#     res = np.array(channel_dice).mean()
#     print(res)
#     return res
# filelist=glob(r'E:\DataCollection\test_mask\*.png')
# dice=0.
# # print(filelist)
# for name in filelist:
#     name1=name
#     name2=name.replace('test_mask','pred')
#     img1=cv2.imread(name1,0)
#     img1=cv2.resize(img1,(224,224),interpolation=cv2.INTER_NEAREST)
#     # print(np.unique(img1))
#     img2=cv2.imread(name2,0)
#     # print(np.unique(img2))
#     dice+=cal_dice(img1,img2)
#
# print(dice/1568)
# -----------------------------------------labemme标签批量处理-------------------------------------
from matplotlib import image as mpimg
# from pyExcelerator.Workbook import Workbook
from scipy.ndimage import io

from skimage.color import gray2rgb


'''
windows下批量将json转化为标签，先进入只存放json的目录，然后进入cmd,输入for /r %i in (*) do labelme_json_to_dataset %i
cd D:\json
for /r %i in (*) do labelme_json_to_dataset %i

'''
# 调用labelme库中原有的 labelme_json_to_dataset 为核心
# 批量将文件夹中的json文件转换，并抽取对应图片至各自文件夹
#
# import os
# import shutil
# import argparse
#
#
# def GetArgs():
#     parser = argparse.ArgumentParser(description='将labelme标注后的json文件批量转换为图片')
#     parser.add_argument('--input', '-i', default='Ultr/ultra_new/', help='json文件目录')
#     parser.add_argument('--out-masks', '-m', default='Ultr/masks/', help='mask图存储目录')
#     parser.add_argument('--out-img', '-r',default='Ultr/image/', help='json文件中提取出的原图存储目录')
#     parser.add_argument('--out-viz', '-v', default='Ultr/viz/',help='mask与原图合并viz图存储目录')
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     _args = GetArgs()
#     _jsonFolder = _args.input
#
#     input_files = os.listdir(_jsonFolder)
#     for sfn in input_files:  # single file name
#         if (os.path.splitext(sfn)[1] == ".json"):  # 是否为json文件
#
#             # 调用labelme_json_to_dataset执行转换,输出到 temp 文件夹
#             os.system("labelme_json_to_dataset %s -o temp" % (_jsonFolder + '/' + sfn))
#
#             # 复制json文件中提取出的原图到存储目录
#             if _args.out_img:
#                 if not os.path.exists(_args.out_img):  # 文件夹是否存在
#                     os.makedirs(_args.out_img)
#
#                 src_img = "temp\img.png"
#                 dst_img = _args.out_img + '/' + os.path.splitext(sfn)[0] + ".png"
#                 shutil.copyfile(src_img, dst_img)
#
#             # 复制mask图到存储目录
#             if _args.out_mask:
#                 if not os.path.exists(_args.out_mask):  # 文件夹是否存在
#                     os.makedirs(_args.out_mask)
#
#                 src_mask = "temp\label.png"
#                 dst_mask = _args.out_mask + '/' + os.path.splitext(sfn)[0] + ".png"
#                 shutil.copyfile(src_mask, dst_mask)
#
#             # 复制viz图到存储目录
#             if _args.out_viz:
#                 if not os.path.exists(_args.out_viz):  # 文件夹是否存在
#                     os.makedirs(_args.out_viz)
#
#                 src_viz = "temp\label_viz.png"
#                 dst_viz = _args.out_viz + '/' + os.path.splitext(sfn)[0] + ".png"
#                 shutil.copyfile(src_viz, dst_viz)

# -----------------------------------labelme 标签批量转化，注意这个函数文件夹里必须只含有JSON文件-----------
# import json
# import os
# import os.path as osp
# import warnings
#
# import PIL.Image
# import yaml
#
# from labelme import utils
# import base64
#
# import numpy as np
# from skimage import img_as_ubyte
#
# def main():
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-o', '--out', default='Ultr/out/')
#     args = parser.parse_args()
#     json_file = r'E:\DataCollection\Ultr\ultra_new'
#
#     # freedom
#     list_path = os.listdir(json_file)
#     print('freedom =', json_file)
#     for i in range(0, len(list_path)):
#         path = os.path.join(json_file, list_path[i])
#         if os.path.isfile(path):
#
#             data = json.load(open(path))
#             img = utils.img_b64_to_arr(data['imageData'])
#             lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
#
#             captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
#
#             lbl_viz = utils.draw_label(lbl, img, captions)
#             out_dir = osp.basename(path).replace('.', '_')
#             save_file_name = out_dir
#             out_dir = osp.join(osp.dirname(path), out_dir)
#
#             if not osp.exists(json_file + '\\' + 'labelme_json'):
#                 os.mkdir(json_file + '\\' + 'labelme_json')
#             labelme_json = json_file + '\\' + 'labelme_json'
#
#             out_dir1 = labelme_json + '\\' + save_file_name
#             if not osp.exists(out_dir1):
#                 os.mkdir(out_dir1)
#
#             PIL.Image.fromarray(img).save(out_dir1 + '\\' + save_file_name + '_img.png')
#             PIL.Image.fromarray(lbl).save(out_dir1 + '\\' + save_file_name + '_label.png')
#
#             PIL.Image.fromarray(lbl_viz).save(out_dir1 + '\\' + save_file_name +
#                                               '_label_viz.png')
#
#             if not osp.exists(json_file + '\\' + 'mask_png'):
#                 os.mkdir(json_file + '\\' + 'mask_png')
#             mask_save2png_path = json_file + '\\' + 'mask_png'
#             ################################
#             # mask_pic = cv2.imread(out_dir1+'\\'+save_file_name+'_label.png',)
#             # print('pic1_deep:',mask_pic.dtype)
#
#             mask_dst = img_as_ubyte(lbl)  # mask_pic
#             print('pic2_deep:', mask_dst.dtype)
#             cv2.imwrite(mask_save2png_path + '\\' + save_file_name + '_label.png', mask_dst)
#             ##################################
#
#             with open(osp.join(out_dir1, 'label_names.txt'), 'w') as f:
#                 for lbl_name in lbl_names:
#                     f.write(lbl_name + '\n')
#
#             warnings.warn('info.yaml is being replaced by label_names.txt')
#             info = dict(label_names=lbl_names)
#             with open(osp.join(out_dir1, 'info.yaml'), 'w') as f:
#                 yaml.safe_dump(info, f, default_flow_style=False)
#
#             print('Saved to: %s' % out_dir1)
#
# if __name__ == '__main__':
#     main()
#
# ---------------------------------------------------------将标签以mask的形式画在原图-----------------------------------
# import cv2
# import os
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.image as mpimg
# import numpy as np
#
# image = mpimg.imread('1.jpg')
# plt.imshow(image)
# plt.axis('off') # 不显示坐标轴
# plt.show()
# image.flags.writeable = True  # 将数组改为读写模式`
# Image.fromarray(np.uint8(image))
# masks = mpimg.imread('1_.jpg')
# plt.imshow(masks)
# plt.axis('off') # 不显示坐标轴
# plt.show()
# print(masks.shape)
# image[:,:,:][masks[:,:,:]>0] = 255
# img=Image.fromarray(image)
# img.save('sert.png')
# cv2.imwrite('test.png',image)


# -------------------------------------------------------方法2
# import numpy as np
# import cv2
# import os
#
# def _draw_to_overlay_image(labimgs, name, srcimgs):
#     num_examples = 626 * 547####图像的大小
#     labimg = cv2.imread(labimgs)
#     srcimg = cv2.imread(srcimgs)
#     srcimg = np.reshape(srcimg, [num_examples, 3]) # //原图
#     labimg = np.reshape(labimg, [num_examples, 3])  #//标签图
#     label_mage=np.zeros([num_examples, 3], np.uint8) #//定义融合后的新图矩阵
#     colors=[[255, 193, 37],[0,0,250]]   #//每一个类别对应一种颜色
#     # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [139, 0, 0], [0, 0, 139], [139, 0, 139], [144, 238, 144],
#     #           [0, 139, 139], [155, 48, 288], [255, 62, 150], [255, 165, 0],
#     #           [255, 211, 155], [255, 193, 37], [255, 255, 0], [192, 255, 62], [0, 255, 255], [153, 50, 204],
#     #           [255, 162, 0], [50, 205, 50], [0, 255, 255],
#     #           [47, 79, 79], [119, 136, 153], [25, 25, 112], [123, 104, 238], [135, 206, 250], [0, 100, 0],
#     #           [173, 255, 47], [188, 143, 143], [250, 128, 114],
#     #           [205, 102, 29], [205, 51, 51], [205, 16, 118]]  # //每一个类别对应一种颜色
#     # lab_pix = [33, 37, 35, 36, 34, 165, 166, 167, 50, 66, 164, 40, 163, 39, 38, 168, 49, 65, 162, 161, 67, 81, 82, 83,
#     #            84, 85, 86, 97, 98, 99, 100, 113] #// 每个类别对应的像素值
#     lab_pix=[0,255]   #//每个类别对应的像素值
#
#     yuv_from_rgb = np.array([[0.299, 0.587, 0.114],
#                              [-0.14714119, -0.28886916, 0.43601035],
#                              [0.61497538, -0.51496512, -0.10001026]])   #//YUV和RGB空间转换的参数矩阵
#     rgb_from_yuv = np.linalg.inv(yuv_from_rgb)     #//求转置矩阵
#     for i in range(0,num_examples):
#         for a,b in enumerate(lab_pix):
#             if labimg[i][0] == b:
#                 label_mage[i] = colors[a]    #//比对标签图中的像素值所属类别，然后下面进行颜色空间的转换
#         Y = srcimg[i].dot(yuv_from_rgb[0].T.copy())
#         U = label_mage[i].dot(yuv_from_rgb[1].T.copy())
#         V = label_mage[i].dot(yuv_from_rgb[2].T.copy())
#         rgb = np.array([Y, U, V]).dot(rgb_from_yuv.T.copy())
#         if rgb[0] > 255: rgb[0] = 255   #//超出像素值255的全部设置为255，下同
#         if rgb[1] > 255: rgb[1] = 255
#         if rgb[2] > 255: rgb[2] = 255
#         if rgb[0] < 0: rgb[0] = 0
#         if rgb[1] < 0: rgb[1] = 0
#         if rgb[2] < 0: rgb[2] = 0
#         label_mage[i] = rgb
#     rimg = np.reshape(label_mage, [547, 626, 3])#图像大小
#     Image.fromarray(rimg).save('dd.jpg')
#     cv2.imwrite(name, rimg)
#
#
# if __name__ == '__main__':
#     _draw_to_overlay_image('1_.jpg', 'test.png', '1.jpg')
#     # src_dir=os.walk('~/anaysis apollo/srcimg')  #//需要可视化数据的原图路径
#     # i=1
#     # for path,b,img_list in src_dir:
#     #     for srcimg in img_list:
#     #         if srcimg.endswith('jpg'):
#     #             labimg=path+'/'+srcimg.split('.')[0]+'_bin.png'  #//找到对应的标签图
#     #             if os.path.exists(labimg):
#     #                 name = '~/anaysis apollo/mage/' + srcimg.split('.')[0] + '_mage.jpg'
#     #                 srcimg= path + '/' + srcimg
#     #                 _draw_to_overlay_image(labimg,name,srcimg)
#     #                 print ('process %d' %i)
#     #                 i += 1
#     # print ('finish!!!')
# -------------------------------根据mask扣出原图对应区域------------------------

# import cv2
# from PIL import Image
# import numpy as np
#
# yuantu = "1.jpg"  # image是.jpg格式的
# mask_path = "1_.jpg"  # mask是.png格式的，
#
# # 使用opencv叠加图片
# img = cv2.imread(yuantu)
# masks = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
#
# # 将image的相素值和mask像素值相加得到结果
# masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), masks=masks)
# cv2.imwrite("masks.jpg", masked)

# -------------------------批量在rgb原图上画出轮廓线---------------------------、
# import cv2
# import os
# import numpy as np
#
# def union_image_mask(image_path, mask_path, image_name, color = (255, 0, 255)):
#     image = cv2.imread(image_path)
#     chang = image.shape[1]
#     kuan = image.shape[0]
#     mask_2d = cv2.imread(mask_path,0)
#     mask_2d = cv2.resize(mask_2d,(chang, kuan))
#
#     coef = 255 if np.max(image)<3 else 1
#     image = (image * coef).astype(np.float32)
#     contours, _ = cv2.findContours(mask_2d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     print(image.shape)
#     cv2.drawContours(image, contours, -1, color, 1)
#     # cv2.imwrite(os.path.join('Add', image_name),image)
#     cv2.imwrite(image_name,image)
#
#
# def change_image(path):
#     for img_name in os.listdir(path):
#         print(img_name)
#         img_path = os.path.join(path, img_name)
#         img = cv2.imread(img_path)
#
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         img[img >0 ] = 255
#         cv2.imwrite(img_path, img)
#
#
# if __name__ == '__main__':
#     #union_image_mask('1.jpg', '1_.jpg', 'masks.jpg')
    # images = os.listdir('dataset/11') # 原图文件夹
    # masks = os.listdir('output') # mask文件夹
    # for image_name in images:
    #     image_path = os.path.join('dataset/11', image_name)
    #     mask_path = os.path.join('output', image_name)
    #     union_image_mask(image_path, mask_path,image_name)
    #change_image('output')

# ------------合并原图与标签，并设置透明度、
# from PIL import Image
# img = Image.open("1.jpg")
# img2 = Image.open("1_.jpg")
# merge = Image.blend(img, img2, 0.3)
# merge.save("my.jpg")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------生成mask覆盖图,,,注意图片格式不能为png-------
# attr_colors = {
#     'pigment_network': (0, 107, 176),
#     'negative_network': (239, 169, 13),
#     'streaks': (29, 24, 21),
#     'milia_like_cyst': (5, 147, 65),
#     'globules': (220, 47, 31)
# }
#
# def put_predict_image(origin_image_np, test_mask, attr, alpha):
#     '''
#     将predict图片以apha透明度覆盖到origin图片中
#     :param origin_image:
#     :param predict_image:
#     :param RGB:
#     :param alpha:
#     :return:
#     '''
#     test_mask_RGB = Image.fromarray(test_mask.astype('uint8')).convert("RGB") # 将原始二值化图像转换成RGB
#
#     test_mask_np = np.asarray(test_mask_RGB,dtype=np.int) # 将二值化图像转换成三维数组
#     height, width, channels = test_mask_np.shape  # 获得图片的三个纬度
#     # 转换预测图像的颜色
#     origin_image_np.flags.writeable=True
#     test_mask_np.flags.writeable = True
#     for row in range(height):
#         for col in range(width):
#             # 上色，这里将mask图像中白色部分转换为我们想要的颜色，
#             if test_mask_np[row, col, 0] == 255 and test_mask_np[row, col, 1] == 255 and test_mask_np[row, col, 2] == 255:
#                 test_mask_np[row, col, 0] = attr_colors[attr][0]
#                 test_mask_np[row, col, 1] = attr_colors[attr][1]
#                 test_mask_np[row, col, 2] = attr_colors[attr][2]
#             # 这里对我们关心的白色区域，将这一步分像素按照比例相加。
#             if test_mask_np[row, col, 0] != 0 or test_mask_np[row,col, 1] != 0 or test_mask_np[row, col, 2] != 0:
#                 origin_image_np[row,col,0] = alpha*origin_image_np[row,col,0] + (1-alpha)*test_mask_np[row, col, 0]
#                 origin_image_np[row,col,1] = alpha*origin_image_np[row,col,1] + (1-alpha)*test_mask_np[row, col, 1]
#                 origin_image_np[row,col,2] = alpha*origin_image_np[row,col,2] + (1-alpha)*test_mask_np[row, col, 2]
#     img = Image.fromarray(origin_image_np)
#     img.save('test.png')
#     return origin_image_np
# if __name__ == '__main__':
#     import matplotlib.image as mpimg
#
#     import matplotlib.pyplot as pil
#     import numpy as np
#
#     image = mpimg.imread('case0005_slice035.bmp')
#     if len(image.shape)!=3:
#         image=gray2rgb(image)
#     # print(image.shape)
#     # origin=cv2.imread('1.png')
#     test=cv2.imread('ncase0005_slice035.png',0)
#     test[test>0]=255
#     put_predict_image(image,test,'globules',0.5)

# ---------------------------------------------------检索同一目录下所有相同后缀的文件,并重命名-------------------------------------
def rename():
    import os
    def findAllFilesWithSpecifiedSuffix(target_dir, target_suffix="dcm"):
        find_res = []
        target_suffix_dot = "." + target_suffix
        walk_generator = os.walk(target_dir)
        for root_path, dirs, files in walk_generator:
            if len(files) < 1:
                continue
            for file in files:
                file_name, suffix_name = os.path.splitext(file)
                if suffix_name == target_suffix_dot:
                    find_res.append(os.path.join(root_path, file))
        return find_res
    a=findAllFilesWithSpecifiedSuffix("E:/DataCollection/NSCLC/NSCLC/volume", "gz")
    leng=len(a)
    print(leng)
    for filename in a:
        pre=os.path.split(filename)[0].split('volume\\')[1]
        surname=os.path.split(filename)[-1]
        newname=pre+'_'+surname
        # print(pre)
        # print(newname)
        try:
            os.rename(filename,filename.replace(surname,newname))
        except Exception as e:
            print(e)
            print('rename file fail\r\n')
        else:
            print('rename file success\r\n')

    # print(sop)
    # --------------------------------------------读取dicom文件元信息-------------------------------------------------------
# import pydicom
# from pydicom.data import get_testdata_file
# for filename in a:
#     pre=os.path.split(filename)[0]
#     surname=os.path.split(filename)[-1].split('.')[0]
#     dataset=pydicom.dcmread(filename, force=True)
#     str1=str(dataset)
#     sop=str1.split('SOP Instance UID                    UI:')[1].split('(')[0].strip()
#     newname=os.path.join(pre,sop+'.dcm')
#     # print(newname)
#     try:
#         os.rename(filename,newname)
#     except Exception as e:
#         print(e)
#         print('rename file fail\r\n')
#     else:
#         print('rename file success\r\n')


# ---------------------------------------------------------------------多类分割将不同的标签用不同颜色覆盖到原图上------------------------------------
attr_colors = {
    '0':(255, 0, 0),
'1':(0, 255, 0),
'2':(0, 0, 255),
'3':(139, 0, 0),
'4':(0, 0, 139),
'5':(139, 0, 139),
'6':(144, 238, 144),
'7':(0, 139, 139),
'8':(155, 48, 288),
'9':(255, 62, 150),
'10':(255, 165, 0),
'11':(255, 211, 155),
'12':(255, 193, 37),
'13':(255, 255, 0),
}

def put_predict_image(origin_image_np, test_mask, attr, alpha):
    '''
    将predict图片以apha透明度覆盖到origin图片中
    :param origin_image:
    :param predict_image:
    :param RGB:
    :param alpha:
    :return:
    '''
    test_mask_RGB = Image.fromarray(test_mask.astype('uint8')).convert("RGB") # 将原始二值化图像转换成RGB

    test_mask_np = np.asarray(test_mask_RGB,dtype=np.int) # 将二值化图像转换成三维数组
    height, width, channels = test_mask_np.shape  # 获得图片的三个纬度
    # 转换预测图像的颜色
    origin_image_np.flags.writeable=True
    test_mask_np.flags.writeable = True
    for row in range(height):
        for col in range(width):
            # 上色，这里将mask图像中白色部分转换为我们想要的颜色，
            if test_mask_np[row, col, 0] == 255 and test_mask_np[row, col, 1] == 255 and test_mask_np[row, col, 2] == 255:
                test_mask_np[row, col, 0] = attr_colors[attr][0]
                test_mask_np[row, col, 1] = attr_colors[attr][1]
                test_mask_np[row, col, 2] = attr_colors[attr][2]
            # 这里对我们关心的白色区域，将这一步分像素按照比例相加。
            if test_mask_np[row, col, 0] != 0 or test_mask_np[row,col, 1] != 0 or test_mask_np[row, col, 2] != 0:
                origin_image_np[row,col,0] = alpha*origin_image_np[row,col,0] + (1-alpha)*test_mask_np[row, col, 0]
                origin_image_np[row,col,1] = alpha*origin_image_np[row,col,1] + (1-alpha)*test_mask_np[row, col, 1]
                origin_image_np[row,col,2] = alpha*origin_image_np[row,col,2] + (1-alpha)*test_mask_np[row, col, 2]
    img = Image.fromarray(origin_image_np)
    # img.save('test.png')
    return origin_image_np
#
# if __name__ == '__main__':
#     import matplotlib.image as mpimg
#     import PIL.Image as Image
#     import numpy as np
#
#     origin = mpimg.imread('case0005_slice035.jpg')####不能为png格式
#     if len(origin.shape)!=3:
#         origin=gray2rgb(origin)
#     pred = cv2.imread('ncase0005_slice035.png',0)
#     pixellist=np.unique(pred)
#     # print(pixellist)
#     # print(type(pixellist))
#     for iters in pixellist:
#         if iters==0:
#             continue
#         img=np.zeros_like(pred)
#         # img[pred==iters]=255
#         # img[pred<255] =0
#         h,w=img.shape
#         for i in range(0,h):
#             for j in range(0,w):
#                 if pred[i][j]==iters:
#                     img[i][j]=255
#                 else:
#                     img[i][j]=0
#         # print(origin.shape)
#         # print(img.shape)
#         # print(numpy.unique(img))
#         origin=put_predict_image(origin, img, str(iters), 0.7)

# ----------------------------------------------二值分割标签转化为TP TN FP FN标签---------------------------------
def transfer():
    import cv2
    import numpy as np

    # label='U/DDUnet/1.nii.gz_11.png'
    # pred='U/new/1.nii.gz_11.png'
    #
    # lab_array=cv2.imread(label,0)
    # pred_array=cv2.imread(pred,0)
    # # print(lab_array.shape)
    # h,w=lab_array.shape
    # new_pred=np.zeros_like(lab_array)
    # # for i in range(0,h):
    # #         for j in range(0,w):
    # #             if lab_array[i][j]==255 and pred_array[i][j]==255:               #TP
    # #                 # print("tp")
    # #                 new_pred[i][j] = 3
    # #             if lab_array[i][j] == 255 and pred_array[i][j] == 0:            #FN
    # #                 # print("fn")
    # #                 new_pred[i][j] = 2
    # #             if lab_array[i][j] == 0 and pred_array[i][j] == 255:              #FP
    # #                 # print("fp")
    # #                 new_pred[i][j] = 1
    # #             if lab_array[i][j] == 0 and pred_array[i][j] == 0:             #TN
    # #                 # print("tn")
    # #                 new_pred[i][j]=0
    # # cv2.imwrite('new.jpg',new_pred*80)

    label_path='U/new'
    pred_path='U/Attention Unet'
    save_path='U/Transfer'
    os.makedirs(save_path,exist_ok=True)
    file_list=glob(label_path+'/*.png')
    for file in file_list:
        labelpath=file
        predpath=file.replace(label_path,pred_path)
        filename=os.path.split(file)[-1]
        # print(filename)
        lab_array = cv2.imread(labelpath, 0)
        pred_array = cv2.imread(predpath, 0)
        # print(lab_array.shape)
        h, w = lab_array.shape
        new_pred = np.zeros_like(lab_array)
        for i in range(0,h):
                for j in range(0,w):
                    if lab_array[i][j]==255 and pred_array[i][j]==255:               #TP
                        # print("tp")
                        new_pred[i][j] = 3
                    if lab_array[i][j] == 255 and pred_array[i][j] == 0:            #FN
                        # print("fn")
                        new_pred[i][j] = 2
                    if lab_array[i][j] == 0 and pred_array[i][j] == 255:              #FP
                        # print("fp")
                        new_pred[i][j] = 1
                    if lab_array[i][j] == 0 and pred_array[i][j] == 0:             #TN
                        # print("tn")
                        new_pred[i][j]=0
        cv2.imwrite(os.path.join(save_path,filename.replace('png','jpg')),new_pred)

# -------------------------------------------------将tp tn fp fn 标签在原图覆盖---------------------------
def tomerge():
    import matplotlib.image as mpimg
    import PIL.Image as Image
    import numpy as np

    label_path='U/Transfer'
    img_path='U/newimg'
    save_path='U/merge'
    os.makedirs(save_path,exist_ok=True)
    file_list = glob(label_path + '/*.jpg')
    # print(file_list)
    for file in file_list:
        labelpath = file
        imgpath = file.replace(label_path, img_path)
        filename = os.path.split(file)[-1]

        origin = mpimg.imread(imgpath)####不能为png格式
        if len(origin.shape)!=3:
            origin=gray2rgb(origin)
        pred = cv2.imread(labelpath,0)
        pixellist=np.unique(pred)
        # print(pixellist)
        # print(type(pixellist))
        for iters in pixellist:
            if iters==0:
                continue
            img=np.zeros_like(pred)
            # img[pred==iters]=255
            # img[pred<255] =0
            h,w=img.shape
            for i in range(0,h):
                for j in range(0,w):
                    if pred[i][j]==iters:
                        img[i][j]=255
                    else:
                        img[i][j]=0
            # print(origin.shape)
            # print(img.shape)
            # print(numpy.unique(img))
            origin=put_predict_image(origin, img, str(iters), 0.4)
        new_img = Image.fromarray(origin)
        new_img.save(os.path.join(save_path,filename))
# -----------------------------------------------------------多类别分割图以不同颜色表示------------------
def givecolor(labepath,savepath=None):
    filename=os.path.split(labepath)[-1]
    label=labepath
    labe=cv2.imread(label,0)
    img=np.zeros_like(labe)
    cv2.imwrite('temp.jpg',img)
    origin = mpimg.imread('temp.jpg')####不能为png格式
    if len(origin.shape)!=3:
        origin=gray2rgb(origin)
    pixellist=np.unique(labe)
    for iters in pixellist:
        if iters==0:
            continue
        img=np.zeros_like(labe)
        # img[pred==iters]=255
        # img[pred<255] =0
        h,w=img.shape
        for i in range(0,h):
            for j in range(0,w):
                if labe[i][j]==iters:
                    img[i][j]=255
                else:
                    img[i][j]=0
        # print(origin.shape)
        # print(img.shape)
        # print(numpy.unique(img))
        origin=put_predict_image(origin, img, str(iters), 0.4)
    new_img = Image.fromarray(origin)
    if savepath is not None:
        save_path=os.path.join(savepath,filename)
    else:
        save_path=labepath

    new_img.save(save_path)
    if os.path.exists('temp.jpg'):
        os.remove('temp.jpg')

def color():
    label_path='U/Transfer'
    filelist=glob(label_path + '/*.jpg')
    for file in filelist:
        givecolor(file)

# transfer()
# tomerge()
# color()


# -----------------------------------------------写入excel文件----------------------------------
def writecsv():
#!/usr/bin/python3
# -*- coding: utf-8 -*-

    # 导入CSV安装包
    import csv

    # 1. 创建文件对象
    f = open('文件名.csv','a',encoding='utf-8')

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    csv_writer.writerow(["姓名","年龄","性别"])

    # 4. 写入csv文件内容
    csv_writer.writerow(["l",'18','男'])
    csv_writer.writerow(["c",'20','男'])
    csv_writer.writerow(["w",'22','女'])

    # 5. 关闭文件
    f.close()
    # # 1. 创建文件对象
    #             checkpoint_dir = os.path.join(args.save_model_path, args.stage)
    #             os.makedirs(checkpoint_dir, exist_ok=True)
    #             filename=os.path.join(checkpoint_dir,'indicator.csv')
    #             f = open(filename, 'a', encoding='utf-8',newline='')
    #             # 2. 基于文件对象构建 csv写入对象
    #             csv_writer = csv.writer(f)
    #             if not os.path.getsize(filename):
    #                 # 3. 构建列表头
    #                 csv_writer.writerow([" ", 'Dice', 'Acc', 'jaccard', 'Sensitivity', 'Specificity'])
    #             # 4. 写入csv文件内容
    #             csv_writer.writerow([args.net_work, str(Dice[0]), str(Acc[0]), str(jaccard[0]), str(Sensitivity[0]), str(Specificity[0])])
    #             # 5. 关闭文件
    #             f.close()

# --------------------------------------------------------灰度-rgb转换-----------------------------------------
def grey_rgb(imgpath,saveroot='/',classnum=2,colorlist=[]):
    label = Image.open(imgpath)
    filename=os.path.split(imgpath)[-1]
    label = label.convert("RGBA")

    width = label.size[0]  # 长度
    height = label.size[1]  # 宽度
    name = os.path.join(saveroot,filename)
    print(name)
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = label.getpixel((i, j))  # i,j表示像素点
            for k in range(0,classnum):
                if (data[0] == k and data[1] == k and data[2] == k):
                    label.putpixel((i, j), colorlist[k])  # 颜色改变
    #         # elif (data >= 22):
    #         #     label.putpixel((i, j), (0, 255, 200, 50))
    label.save(name)
def example():
    colorlist=[(255,255,255),(255,0,0),(0,255,0),(0, 0, 255), (139, 0, 0), (0, 0, 139), (139, 0, 139), (144, 238, 144),
                (0, 139, 139), (155, 48, 288)]
    imapath='ncase0005_slice035.png'
    grey_rgb(imapath,'U/',9,colorlist)

# ----------------------------------------------------将3d数据调窗处理----------------------------------------------------
def setwindow():
    import SimpleITK as sitk
    import numpy as np
    import os
    from glob import glob


    def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
        newImg = sitk.GetImageFromArray(savedImg)
        newImg.SetOrigin(origin)
        newImg.SetDirection(direction)
        newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
        sitk.WriteImage(newImg, saved_name)


    def window_transform(ct_array, windowWidth, windowCenter, normal=False):
        """
        return: trucated image according to window center and window width
        and normalized to [0,1]
        """
        minWindow = float(windowCenter) - 0.5 * float(windowWidth)
        newimg = (ct_array - minWindow) / float(windowWidth)
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        if not normal:
            newimg = (newimg * 255).astype('uint8')
        return newimg


    # ct_path = 'Head Scan'
    # saved_path = 'Head Scan/seg'
    name_list=glob('NSCLC/NSCLC/volume/*.gz')
    # name_list = ['volume-covid19-A-0031.nii']
    for name in name_list:
        ct = sitk.ReadImage(os.path.join(name))
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        xyz_thickness = ct.GetSpacing()
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(name.replace('volume', 'masks'))))
        seg_bg = seg_array == 0
        seg_liver = seg_array >= 1

        ct_bg = ct_array * seg_bg
        ct_liver = ct_array * seg_liver


        liver_min = ct_liver.min()
        liver_max = ct_liver.max()


        # by liver
        liver_wide = liver_max - liver_min
        liver_center = (liver_max + liver_min) / 2
        liver_wl = window_transform(ct_array, liver_wide, liver_center, normal=True)
        saved_name = os.path.join(name)
        saved_preprocessed(liver_wl, origin, direction, xyz_thickness, saved_name)
# setwindow()
# --------------------------------------------------3d数据切片-----------------------------------------

 # coding=utf-8
# 2、针对多个Nii的切片
import SimpleITK as sitk
import skimage.io as io
import sys, os
import numpy as np
from tqdm import tqdm
def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data
def find_label_image(label_data, all_one_array, torch_CT, niidir_one):
    index = -1
    print(label_data.shape)
    for i in tqdm(range(label_data.shape[0])):  # 遍历所有切片
        true_sum = np.sum(label_data[i] == LABEL_NUM * all_one_array[i])  # 找到有标签为label_lum的切片。
        if true_sum > 0:  # 没有打该标签的不生成图片，避免类别不平衡问题。
            index += 1
            file_dir1 = os.path.join(LABEL_OUT_PATH, niidir_one + '_' + str(index) + '.png')  # 标签图片的路径
            file_dir2 = os.path.join(IMAGE_OUT_PATH, niidir_one + '_' + str(
                index) + '.png')  # 数据图片的路径niniinii-2D_labels.py:46-2D_labels.py:46inii-2D_labels.py:46-2D_labels.py:46
            # print(i)
            # 二值化label
            label_img = np.zeros(label_data[i].shape, dtype=np.uint8)
            for x in range(label_img.shape[0]):
                for y in range(label_img.shape[1]):
                    if label_data[i][x, y] == LABEL_NUM:
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
def _img_transfor(itk,ww,wl):
    img_arr = sitk.GetArrayFromImage(itk).astype(np.float32)
    if IF_WINDOWS:
        img_arr = window_normalize(img_arr, WW=ww, WL=wl)
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
    # 例子：BraTS_2018数据集（MRI）中的标签：0是背景、1是坏疽、2是浮肿、4是增强肿瘤区
    # 选择标签值，输出相应的标签切片图
    LABEL_NUM = 255

    # CT 加窗调对比度（hu值）
    IF_WINDOWS = 0  # 0是不加窗  ； 1 是加窗
    WINDOW_WEIGHT =1
    WINDOW_LONGTH = 0.1
    # nii路径'  './HGG'
    NII_IN_DIR_PATH = 'NSCLC/volume'
    # NII_IMAGES_IN_DIR_PATH =
    # NII_LABELS_IN_DIR_PATH = './HGG'
    # 图片输出路径
    IMAGE_OUT_PATH = './NSCLC/Image'
    LABEL_OUT_PATH = './NSCLC/Label'
    os.makedirs(IMAGE_OUT_PATH, exist_ok=True)
    os.makedirs(LABEL_OUT_PATH, exist_ok=True)
    np.set_printoptions(threshold=np.inf)
    # index = -1
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
        nii_label_path = (NII_IN_DIR_PATH.replace("volume", "masks") + '/' + niidir_one)
        nii_image_path = NII_IN_DIR_PATH + '/' + niidir_one
        # print(nii_label_path)
        # nii_image_path = NII_IN_DIR_PATH + '/' + niidir_one
        label_filename = nii_label_path
        train_filename = nii_image_path
        label_data = read_img(label_filename)
        print(np.unique(label_data))
        train_data = read_img(train_filename)
        # 生成与label维度一样的array，且元素都为1.
        like_label_one_array = np.ones_like(label_data)
        itk_gt = sitk.ReadImage(label_filename)
        itk_CT = sitk.ReadImage(train_filename)
        torch_CT = _img_transfor(itk_CT,WINDOW_WEIGHT,WINDOW_LONGTH)
        find_label_image(label_data, like_label_one_array, torch_CT, niidir_one)




# ---------------------------------------------------------对图像进行小波变换------------------------------------
# import cv2
# import numpy as np
# from pywt import dwt2, idwt2
#
# # 读取灰度图
# img = cv2.imread('image/000004.png', 0)
#
# # 对img进行haar小波变换：
# cA, (cH, cV, cD) = dwt2(img, 'haar')
#
# # 小波变换之后，低频分量对应的图像：
# cv2.imwrite('lena.png', np.uint8(cA / np.max(cA) * 255))
# # 小波变换之后，水平方向高频分量对应的图像：
# cv2.imwrite('lena_h.png', np.uint8(cH / np.max(cH) * 255))
# # 小波变换之后，垂直平方向高频分量对应的图像：
# cv2.imwrite('lena_v.png', np.uint8(cV / np.max(cV) * 255))
# # 小波变换之后，对角线方向高频分量对应的图像：
# cv2.imwrite('lena_d.png', np.uint8(cD / np.max(cD) * 255))
#
# # 根据小波系数重构回去的图像
# rimg = idwt2((cA, (cH, cV, cD)), 'haar')
# cv2.imwrite('rimg.png', np.uint8(rimg))
# ----------------------------------------------------提取标签图像的边界线--------------------------------------

#  -*- coding: utf-8 -*-
import cv2
import os

def Edge_Extract(root):
    img_root = os.path.join(root,'img_masks')			# 修改为保存图像的文件名
    edge_root = os.path.join(root,'img_edge')			# 结果输出文件

    if not os.path.exists(edge_root):
        os.mkdir(edge_root)

    file_names = os.listdir(img_root)
    img_name = []

    for name in file_names:
        if not name.endswith('.png'):
            assert "This file %s is not PNG"%(name)
        img_name.append(os.path.join(img_root,name[:-4]+'.png'))

    index = 0
    for image in img_name:
        img = cv2.imread(image,0)
        cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,30,100))
        index += 1
    return 0

#
# if __name__ == '__main__':
#     root = 'Masks/'	# 修改为你对应的文件路径
#     Edge_Extract(root)




# -----------------------------------------膨胀腐蚀----------------------------------------------------------
# import cv2 as cv
# import numpy as np
#
#
# def erode_demo(image):
#     # print(image.shape)
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     # cv.imshow("binary", binary)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))  # 定义结构元素的形状和大小
#     dst = cv.erode(binary, kernel)  # 腐蚀操作
#     cv.imshow("erode_demo", dst)
#
#
# def dilate_demo(image):
#     # print(image.shape)
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     # cv.imshow("binary", binary)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 定义结构元素的形状和大小
#     dst = cv.dilate(binary, kernel)  # 膨胀操作
#     cv.imshow("dilate_demo", dst)
#
#
# src = cv.imread("Masks/img_edge/10.png")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# erode_demo(src)
# dilate_demo(src)
#
# cv.waitKey(0)
#
# cv.destroyAllWindows()
# ---------------------------------------------------------------根据边界线填充闭合区域-------------------------------------------
import cv2
import numpy as np

'''
图像说明：
图像为二值化图像，255白色为目标物，0黑色为背景
要填充白色目标物中的黑色空洞
'''


def FillHole(imgPath, SavePath):
    im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    print(im_in)

    # 复制 im_in 图像
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break
    # 得到im_floodfill
    cv2.floodFill(im_floodfill, mask, seedPoint, 255);

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    # 保存结果
    cv2.imwrite(SavePath, im_out)
# from glob import glob
# filelist=glob('Ultra_HC/label/*.png')
# # print(filelist)
# save_dir='Ultra_HC/new_label'
# os.makedirs(save_dir,exist_ok=True)
# for file in filelist:
#     FillHole(file,file.replace('label','new_label'))


# FillHole('Ultra_HC/label/1_HC.png','teast.png')

# -----------------------------将一张图的多个标签合并成一个图---------------------------------
from glob import glob
import os
import shutil
# filecollect=glob('TCIA_SegPC/validation/y/*.bmp')
# print(filecollect)
# for file in filecollect:
#     filename=file.split('\\')[1]
#     name=file.split('\\')[1].split('_')[0]
#     print(name)
#     os.makedirs(os.path.join('TCIA_SegPC/validation/valid_mask/',name),exist_ok=True)
#     shutil.copy(file,(os.path.join('TCIA_SegPC/validation/valid_mask/',name,filename)))
# -*----------------------------------------------分别遍历一个目录中的所有子目录和所有文件
# import os
# import traceback
#
# file = []
# dir = []
# dir_res = []
# give_path = 'TCIA_SegPC/validation/valid_mask'
#
# def list_dir(start_dir):
#     dir_res = os.listdir(start_dir)
#     for path in dir_res:
#         temp_path = start_dir + '/' + path
#         if os.path.isfile(temp_path):
#             file.append(temp_path)
#         if os.path.isdir(temp_path):
#             dir.append(temp_path)
#             list_dir(temp_path)
#
# if __name__ == '__main__':
#     try:
#         list_dir(give_path)
#         # print("file：", file)
#         # print("dir：", dir)
#         for di in dir:
#             files=glob(di+'/*.bmp')
#             savepath=di+'/'+di.split('/')[-1]+'.bmp'
#             h,w=cv2.imread(files[0]).shape[0],cv2.imread(files[0]).shape[1]
#             img_array=np.zeros((h, w))
#             for file in files:
#                 arr=cv2.imread(file,0)
#                 img_array=img_array+arr
#             cv2.imwrite(savepath,img_array)
#             print(savepath)
#
#
#
#     except Exception as e:
#         print(traceback.format_exc(), flush=True)

# ---------------------------------------将多标签合并并调整大小-----------------------------------------------------------------------------------------
# import os
# from glob import glob
#
# import cv2
# import numpy as np
# from tqdm import tqdm
#
#
# def main():
#     img_size = 96
#
#     paths = glob('inputs/data-science-bowl-2018/stage1_train/*')
#
#     os.makedirs('inputs/dsb2018_%d/images' % img_size, exist_ok=True)
#     os.makedirs('inputs/dsb2018_%d/masks' % img_size, exist_ok=True)
#
#     for i in tqdm(range(len(paths))):
#         path = paths[i]
#         img = cv2.imread(os.path.join(path, 'images',
#                          os.path.basename(path) + '.png'))
#         mask = np.zeros((img.shape[0], img.shape[1]))
#         for mask_path in glob(os.path.join(path, 'masks', '*')):
#             mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
#             mask[mask_] = 1
#         if len(img.shape) == 2:
#             img = np.tile(img[..., None], (1, 1, 3))
#         if img.shape[2] == 4:
#             img = img[..., :3]
#         img = cv2.resize(img, (img_size, img_size))
#         mask = cv2.resize(mask, (img_size, img_size))
#         cv2.imwrite(os.path.join('inputs/dsb2018_%d/images' % img_size,
#                     os.path.basename(path) + '.png'), img)
#         cv2.imwrite(os.path.join('inputs/dsb2018_%d/masks' % img_size,
#                     os.path.basename(path) + '.png'), (mask * 255).astype('uint8'))
#
#
# if __name__ == '__main__':
#     main()
