#coding=utf-8
import scipy, numpy, shutil, os, nibabel
import sys, getopt
import imageio
import numpy as np
'''
SimpleITK 和 Nibabel 区别在于：（nii图像可以看成2维，也可以看成三维）
SimpleITK读取数据是（X，Y，Z）显示，Nibabel读取图像是（Z，Y，X）显示，也就是Nibabel加载的图像会旋转90°，其中X表示通道数，即切片层数。详情
想要批量抽取MRI数据的几个切片，借鉴了上述代码的Python版本
，原始代码抽取了全部切片。本次实验主要是抽取sMRI的横断面（axial）方向的其中30张切片，
数量根据自己需求可自行设置。'''
def niito2D(filepath):
    inputfiles = os.listdir(filepath)  # 遍历文件夹数据
    outputfile = 'result\\'  # 输出文件夹
    print('Input file is ', inputfiles)
    print('Output folder is ', outputfile)
    for inputfile in inputfiles:
        image_array = nibabel.load(filepath + inputfile).get_data()  # 数据读取
        print(image_array.shape)#(256,256,192)
        print(len(image_array.shape))#3
        all_slice=image_array.shape[2]
        # set destination folder
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)  # 不存在输出文件夹则新建
            print("Created ouput directory: " + outputfile)
        print('Reading NIfTI file...')
        total_slices = all_slice  # 总切片数
        slice_counter = 0  # 从第几个切片开始
        # iterate through slices
        for current_slice in range(0,  total_slices):
            # alternate slices
            if (slice_counter % 1) == 0:
                data = image_array[:, :, current_slice]  # 保存该切片，可以选择不同方向。
                # alternate slices and save as png
                if (slice_counter % 1) == 0:
                    print('Saving image...')
                    # 切片命名
                    image_name = inputfile[:-4] + "{:0>3}".format(str(current_slice + 1)) + ".png"
                    # 保存
                    # data=  data.astype(np.uint8)
                    imageio.imwrite(image_name, data)
                    print('Saved.')
                    # move images to folder
                    print('Moving image...')
                    src = image_name
                    shutil.move(src, outputfile)
                    slice_counter += 1
                    print('Moved.')
    print('Finished converting images')

if __name__ == '__main__':
    niito2D('./volume/')