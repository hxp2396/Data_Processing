# import cv2
# def image_label_cut(image_path, label_5_path, label_3_path=None):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     print(image.shape)
#     label_5 = cv2.imread(label_5_path, cv2.IMREAD_COLOR)
#     label_3 = cv2.imread(label_3_path, cv2.IMREAD_COLOR)
#     count = 120
#     for i in range(0,10):
#         a = 60 + 5 * i
#         b = 252 + 5 * i
#         c = 200 + 5 * i
#         d = 392 + 5* i
#         img = image[a:b, c:d, :]
#         New_label_5 = label_5[a:b, c:d, :]
#         # New_label_3 = label_3[a:b, c:d, :]
#
#         cv2.imwrite(('%d_sat.jpg' % count), img)
#         cv2.imwrite(('%d_label5.png' % count), New_label_5)
#         # cv2.imwrite(("F:\\" + '%d_label3.png' % count), New_label_3)
#
#         count += 1
#
# img_path='1.jpg'
# labelpath='1s.jpg'
# image_label_cut(img_path,labelpath)
# - -----------------------------------遍历一个文件夹中所有图片并把它们一分为二----------------------------------------------------------------------------------------------------
#
# from PIL import Image
# import os
#
# path = r'C:\Users\hxp\Desktop\crop\save'   #文件目录
# #path这个目录处理完之后需要手动更改
# path_list = os.listdir(path)
# #path_list.remove('.DS_Store')   #macos中的文件管理文件，默认隐藏，这里可以忽略，如果是mac可能需要加回这行（笔者没有用过mac）
# print(path_list)
#
# for i in path_list: #截左半张图片
#     a = open(os.path.join(path,i),'rb')
#     img = Image.open(a)
#     w = img.width       #图片的宽
#     h = img.height      #图片的高
#     print('正在处理图片',i,'宽',w,'长',h)
#
#     box = (0,0,w*0.5,h) #box元组内分别是 所处理图片中想要截取的部分的 左上角和右下角的坐标
#     img = img.crop(box)
#     print('正在截取左半张图...')
#     img.save('L'+i) #这里需要对截出的图加一个字母进行标识，防止名称相同导致覆盖
#     print('L-',i,'保存成功')
#
# for i in path_list: #截取右半张图片
#     a = open(os.path.join(path,i),'rb')
#     img = Image.open(a)
#     w = img.width       #图片的宽
#     h = img.height      #图片的高
#     print('正在处理图片',i,'宽',w,'长',h)
#
#     box = (w*0.5,0,w,h)
#     img = img.crop(box)
#     print('正在截取右半张图...')
#     img.save('R'+i)
#     print('R-',i,'保存成功')
#
# print("'{}'目录下所有图片已经保存到本文件目录下。".format(path))
from glob import glob
# -----------------------------------------------------------------------------------------------
import numpy as  np
import cv2

path='2.jpg'
image=cv2.imread(path,0)
print(np.unique(image))

files=glob('cell/Masks/*.png')
print(files)
for file in files:
    array=cv2.imread(file,0)
    print(np.unique(array))
    # array[array>180]=255
    # array[array < 254] = 0
    cv2.imwrite(file,array)

# ----------------------------------------------膨胀腐蚀------------------------------------------
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
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))  # 定义结构元素的形状和大小
#     dst = cv.dilate(binary, kernel)  # 膨胀操作
#     cv.imshow("dilate_demo", dst)
#
#
# src = cv.imread("cell/Masks/R1-1.png")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# erode_demo(src)
# dilate_demo(src)
#
# cv.waitKey(0)
#
# cv.destroyAllWindows()

# -----------------------------------遍历一个文件夹中所有图片并把它们一分为十-------------------------------------
from PIL import Image
import os

path = 'X:\\文件\\拼图a4'   #文件目录
#path这个目录截完之后需要手动更改
path_list = os.listdir(path)
print(path_list)

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (0,0,w*0.1,h)
    img = img.crop(box)
    print('正在截取第一部分...')
    img.save('A'+i)
    print('A-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.1,0,w*0.2,h)
    img = img.crop(box)
    print('正在截取第二部分...')
    img.save('B'+i)
    print('B-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.2,0,w*0.3,h)
    img = img.crop(box)
    print('正在截取第三部分...')
    img.save('C'+i)
    print('C-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.3,0,w*0.4,h)
    img = img.crop(box)
    print('正在截取第四部分...')
    img.save('D'+i)
    print('D-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.4,0,w*0.5,h)
    img = img.crop(box)
    print('正在截取第五部分...')
    img.save('E'+i)
    print('E-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.5,0,w*0.6,h)
    img = img.crop(box)
    print('正在截取第六部分...')
    img.save('F'+i)
    print('F-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.6,0,w*0.7,h)
    img = img.crop(box)
    print('正在截取第七部分...')
    img.save('G'+i)
    print('G-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.7,0,w*0.8,h)
    img = img.crop(box)
    print('正在截取第八部分...')
    img.save('H'+i)
    print('H-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.8,0,w*0.9,h)
    img = img.crop(box)
    print('正在截取第九部分...')
    img.save('I'+i)
    print('I-',i,'保存成功')

for i in path_list:
    a = open(os.path.join(path,i),'rb')
    img = Image.open(a)
    w = img.width       #图片的宽
    h = img.height      #图片的高
    print('正在处理图片',i,'宽',w,'长',h)

    box = (w*0.9,0,w,h)
    img = img.crop(box)
    print('正在截取第十部分...')
    img.save('J'+i)
    print('J-',i,'保存成功')

print("'{}'目录下所有图片已经保存到本文件目录下。".format(path))

# ------------------------.遍历一个文件夹中所有图片并把它们按宽度旋转-----------------------------
from PIL import Image
import os

path = 'X:\\文件\\新建文件夹'  # 文件目录
# path这个目录截完之后需要手动更改
path_list = os.listdir(path)
print(path_list)

for i in path_list:
    a = open(os.path.join(path, i), 'rb')
    img = Image.open(a)
    w = img.width  # 图片的宽
    h = img.height  # 图片的高
    print('正在处理图片', i, '宽', w, '长', h)
    if h > w:
        img.rotate(270, expand=True).save('0' + i)  # 这里具体去看pillow里的rotate方法
        print('旋转成功')

print("'{}'目录下所有图片已经保存到本文件目录下。".format(path))


