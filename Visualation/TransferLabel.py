import os
from glob import glob
import cv2
import numpy as np
# rootpath=r'E:\DataCollection\ultra_nerve\Masks\1_1.png'
# array=cv2.imread(rootpath,0)
# print(np.unique(array))
from tqdm import tqdm

files=glob('tubule/label/*.bmp')
# print(files)
with tqdm(total=len(files)) as bar: # total表示预期的迭代次数
    for file in files:
        array=cv2.imread(file,0)
        h,w=array.shape
        # for i in range(h):
        #     for j in range(w):
        #         if array[i][j]==60:
        #             array[i][j]=40
        #         if array[i][j]==80:
        #             array[i][j]=40
        print(np.unique(array))
        # array[array>0]=255
        # array[array < 254] = 0
        # cv2.imwrite(file,array)
        bar.update()

#     # ---------------------------------删除无病灶图片
# import os
# from glob import glob
# import cv2
# import numpy as np
# rootpath=r'E:\DataCollection\spleen\masks'
# files=glob(rootpath+'/*.jpg')
# # print(files)
# for file in files:
#     array=cv2.imread(file)
#     if len(np.unique(array))==1:
#         os.remove(file)
#     print(np.unique(array))

# ------------------------------多类别分割图以不同颜色表示------------------
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
from matplotlib import image as mpimg
from skimage.color import gray2rgb

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

# -------------------------------------------------------------病理图像裁剪切片--------------------------------------------
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools

def check_and_creat_dir(file_url):
    '''
    判断文件是否存在，文件路径不存在则创建文件夹
    :param file_url: 文件路径，包含文件名
    :return:
    '''
    file_gang_list = file_url.split('/')
    # print("okk",file_gang_list )
    if len(file_gang_list) > 1:
        [fname, fename] = os.path.split(file_url)
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            return None
        # 还可以直接创建空文件
    else:
        return None

"""
病理图像数据集扩充，
大尺寸病理图像剪切，包括：原图、上下翻转、左右翻转、旋转180、小角度随机旋转
"""
def cut_image_2part(image):
    width, height = image.size
    # print("image.size", image.size,image)
    item_width =299
    box_list = []
    for m in range(0,4):
        #创造剪切图片的随机起点
        x_start = np.random.randint(0, high=42, size=1, dtype='l').item()
        y_start = np.random.randint(0, high=101, size=1, dtype='l').item()
        for i in range(0, 1):
            for j in range(0, 2):
                box = (x_start+j * item_width, y_start+i * item_width, x_start+(j + 1) * item_width, y_start+(i + 1) * item_width)
                box_list.append(box)
                print( box)
                # plt.imshow(image.crop(box))
                # plt.show()
    image_list = [image.crop(box) for box in box_list]
    image_LR= image.transpose(Image.FLIP_LEFT_RIGHT)
    image_TB= image.transpose(Image.FLIP_TOP_BOTTOM)
    image_ROT= image.rotate(180)
    image_list = image_list + [image_LR.crop(box) for box in box_list]+\
                 [image_TB.crop(box) for box in box_list]+\
                 [image_ROT.crop(box) for box in box_list]
    cut_box_list = []
    x_start =20
    y_start =90
    for i in range(0, 1):
        for j in range(0, 2):
            box = (x_start+j * item_width, y_start+i * item_width, x_start+(j + 1) * item_width, y_start+(i + 1) * item_width)
            cut_box_list.append(box)
    print("cut_box_list",cut_box_list)
    rot1 = np.random.randint(0, high=20, size=1, dtype='l').item()
    rot2 = np.random.randint(-20, high=0,size=1, dtype='l').item()
    image_list = image_list + [image.rotate(rot1).crop(box) for box in cut_box_list]+\
                 [image.rotate(rot2).crop(box) for box in cut_box_list]+\
                 [image_LR.rotate(rot1).crop(box) for box in cut_box_list]+\
                 [image_LR.rotate(rot2).crop(box) for box in cut_box_list]+\
                 [image_TB.rotate(rot1).crop(box) for box in cut_box_list]+\
                 [image_TB.rotate(rot2).crop(box) for box in cut_box_list]+\
                 [image_ROT.rotate(rot1).crop(box) for box in cut_box_list]+\
                 [image_ROT.rotate(rot2).crop(box) for box in cut_box_list]
    return image_list

def cut_image(image):
    width, height = image.size
    # print("image.size", image.size,image)
    item_width = 299
    box_list = []
    # # (left, upper, right, lower)
    for i in range(0, 1):  # 两重循环，生成9张图片基于原图的位置
        for j in range(0, 1):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)
            # plt.imshow(image.crop(box))
            # plt.show()
    image2 = image.rotate(100, expand=0)
    # plt.imshow(image2)
    # plt.show()

    image_list = [image.crop(box) for box in box_list]
    # if rotate:
    #     image_list = image_list + [image2.crop(box) for box in box_list]
    return image_list

# 保存在list输出到图片中
def save_images(file_path, image_list):
    index = 1
    for image in image_list:
        save_path="E:/DataCollection/XXX/cut/" + file_path+"_" + str(index) + '.tif'
        print(save_path)
        check_and_creat_dir(save_path)
        image.save(save_path)
        index += 1

# if __name__ == '__main__':
#     root ="E:/DataCollection/XXX/MonuSeg/Training"
#     p = os.listdir(root)
#     print(p)
#     for file_path in p:
#         file_root=os.path.join(root,file_path)
#         p2 = os.listdir(file_root)
#         # print(p2)
#         for file_path2 in p2:
#             image_path=os.path.join(file_root,file_path2)
#             # print(image_path)
#             image = Image.open(image_path)
#             # plt.imshow(image)
#             # plt.show()
#             image = image.resize((1024,1024))
#             image_list = cut_image_2part(image)
#             # print(file_path+"/"+file_path2)
#             save_images(file_path + "/" + file_path2, image_list)
