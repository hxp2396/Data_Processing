# encoding=utf-8

# function: 更改图片尺寸大小

from PIL import Image
import os.path
import glob
'''
filein: 输入图片
fileout: 输出图片
width: 输出图片宽度
height:输出图片高度
type:输出图片类型（png, gif, jpeg...）

这个函数img.resize((width, height),Image.ANTIALIAS)
第二个参数：
Image.NEAREST ：低质量
Image.BILINEAR：双线性
Image.BICUBIC ：三次样条插值
Image.ANTIALIAS：高质量
'''
# ---------------------------------------------同时对图片和标签调整大小-------------------------------

def ResizeImage(filein, fileout, width, height, type):
    img = Image.open(filein)
    out = img.resize((width, height), Image.ANTIALIAS)  # resize image with high-quality
    out.save(fileout, type)


def convertjpg(jpgfile,outdir,width=512,height=512):
    # print(jpgfile)
    label=jpgfile.replace('Data','Ground')
    # print(label)
    img=Image.open(jpgfile)
    new_img=img.resize((width,height),Image.NEAREST)
    new_img.save(os.path.join(outdir,'Data',os.path.basename(jpgfile)))
    labels = Image.open(label)
    new_label = labels.resize((width, height), Image.NEAREST)
    new_label.save(os.path.join(outdir, 'Ground', os.path.basename(jpgfile)))

def convertimage_only(jpgfile,outdir,width=512,height=512):
    img = Image.open(jpgfile)
    new_img = img.resize((width, height), Image.NEAREST)
    new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))

####批量修改图片大小
# for jpgfile in glob.glob("XXX/Ultra2/*.bmp"):
#     savepath="XXX/Ultnewimg/"
#     os.makedirs(savepath,exist_ok=True)
#     convertimage_only(jpgfile,savepath )
    # convertjpg(jpgfile,r"E:\DataCollection\XXX\Ultnewimg")

    # ---------------------------------------------只对图片或者标签调整大小-------------------------------




 # ---------------- 等比例缩放图像大小# -------------------------------------------------------------------------------------------------------------
import cv2
def imgToSize(img):
    ''' imgToSize()
    # ----------------------------------------
    # Function:   将图像等比例缩放到 512x512 大小
    #             根据图像长宽不同分为两种缩放方式
    # Param img:  图像 Mat
    # Return img: 返回缩放后的图片
    # Example:    img = imgToSize(img)
    # ----------------------------------------
    '''
    # 测试点
    # cv2.imshow('metaImg.jpg', img)

    imgHeight, imgWidth = img.shape[:2]

    # cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src 原图像，dsize 输出图像的大小，
    # img = cv2.resize(img, (512,512))
    zoomHeight = 512
    zoomWidth = int(imgWidth*512/imgHeight)
    img = cv2.resize(img, (zoomWidth,zoomHeight))

    # 测试点
    # cv2.imshow('resizeImg', img)

    # 如果图片属于 Width<Height，那么宽度将达不到 512
    if imgWidth >= imgHeight:
        # 正常截取图像
        w1 = (zoomWidth-512)//2
        # 图像坐标为先 Height，后 Width
        img = img[0:512, w1:w1+512]
    else:
        # 如果宽度小于 512，那么对两侧边界填充为全黑色
        # 根据图像的边界的像素值，向外扩充图片，每个方向扩充50个像素，常数填充：
        # dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]])
        # dst = cv2.copyMakeBorder(img,50,50,50,50, cv2.BORDER_CONSTANT,value=[0,255,0])
        # 需要填充的宽度为 512-zoomWidth
        left = (512-zoomWidth)//2
        # 避免余数取不到
        right = left+1
        img = cv2.copyMakeBorder(img, 0,0,left,right, cv2.BORDER_CONSTANT, value=[0,0,0])
        img = img[0:512, 0:512]

    # 测试点
    # cv2.imshow('size512', img)

    return img
# def re():
from glob import glob
filelist=glob('XXX/Ultra2/*.bmp')
print(filelist)
for name in filelist:
    file=os.path.split(name)[-1]
    img=cv2.imread(name)
    img=imgToSize(img)
    cv2.imwrite('XXX/ultra_new_mask/'+file,img)



# img=cv2.imread()
#
# img=imgToSize(img)
# cv2.imwrite('new.png',img)
###图像翻转
# encoding:utf-8

# import cv2
# image = cv2.imread("14.jpg")
#
# # Flipped Horizontally 水平翻转
# h_flip = cv2.flip(image, 1)
# cv2.imwrite("girl-h.jpg", h_flip)
#
# # Flipped Vertically 垂直翻转
# v_flip = cv2.flip(image, 0)
# cv2.imwrite("girl-v.jpg", v_flip)
#
# # Flipped Horizontally & Vertically 水平垂直翻转
# hv_flip = cv2.flip(image, -1)
# cv2.imwrite("girl-hv.jpg", hv_flip)
#
# if __name__ == "__main__":
#     filein = r'0.jpg'
#     fileout = r'testout.png'
#     width = 6000
#     height = 6000
#     type = 'png'
#     ResizeImage(filein, fileout, width, height, type)