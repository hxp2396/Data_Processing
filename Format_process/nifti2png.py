import SimpleITK as sitk
import numpy as np
import cv2
import os
def mha2jpg(mhaPath, outFolder, windowsCenter=0, windowsSize=0):
    """
    The function can output a group of jpg files by a specified mha file.
    Args:
        mhaPath:mha file path.
        outfolder:The folder that the jpg files are saved.
        windowsCenter:the CT windows center.
        windowsSize:the CT windows size.
    Return:void
    """
    name=os.path.basename(mhaPath).split('.')[0]#带后缀的文件名
    image = sitk.ReadImage(mhaPath)
    img_data = sitk.GetArrayFromImage(image)
    channel = img_data.shape[0]

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    low = windowsCenter - windowsSize / 2
    high = windowsCenter + windowsSize / 2

    for s in range(channel):
        slicer = img_data[s, :, :]
        if windowsCenter != 0 and windowsSize != 0:
            slicer[slicer < low] = low
            slicer[slicer > high] = high
            slicer = slicer - low
        img = cv2.normalize(slicer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(outFolder, name+str(s) + '.png'), img)


def main():
    mha1 = r'D:\code\Data_Processing\dataset\predata\folder5\ADC'
    mha2 = r'D:\code\Data_Processing\dataset\predata\folder5\ADC_ss'
    mha3 = r'D:\code\Data_Processing\dataset\predata\folder5\LABEL'
    wc = 0
    ws = 0
    for file in os.listdir(mha1):
        file=os.path.join(mha1,file)
        out=mha1.replace('predata', 'PNG')
        os.makedirs(mha1.replace('predata', 'PNG'), exist_ok=True)
        mha2jpg(file, out, wc, ws)
    for file in os.listdir(mha2):
        file=os.path.join(mha2,file)
        out=mha2.replace('predata', 'PNG')
        os.makedirs(mha2.replace('predata', 'PNG'), exist_ok=True)
        mha2jpg(file, out, wc, ws)
    for file in os.listdir(mha3):
        file=os.path.join(mha3,file)
        out=mha3.replace('predata', 'PNG')
        os.makedirs(mha3.replace('predata', 'PNG'), exist_ok=True)
        mha2jpg(file, out, wc, ws)

if __name__ == "__main__":
    main()
