##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil

def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    # rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    # picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    picknumber =1000
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        # shutil.move(fileDir + name, tarDir + name)###选择剪切过去
        shutil.copy(fileDir + name, tarDir + name)  ###选择复制过去
    return


if __name__ == '__main__':
    fileDir = "E:/qata-covid19/QaTa-COV19/Control_Group/Control_Group_II/CHESTXRAY-14/Train/"  # 源图片文件夹路径
    tarDir = './result/'  # 移动到新的文件夹路径
    os.makedirs(tarDir,exist_ok=True)
    moveFile(fileDir)