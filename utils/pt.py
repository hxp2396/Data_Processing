from PIL import Image
import os
import os.path

rootdir = 'validation/labelcol/'  # 指明被遍历的文件夹
for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print('the fulll name of the file is :' + currentPath)

        im = Image.open(currentPath)
        # 将图片重新设置尺寸
        out = im.resize((128,128))
        newname = r"new" + '\\' + filename
        out.save(newname)
