from glob import glob
import os
import shutil
image_path='AATTCT-IDS/Image/Extracted/'
images=glob(image_path+'*/*.jpg')
for file in images:
    name=os.path.split(file)[-1]
    patientid=os.path.split(file)[0].split('\\')[-1]
    img_savepath='./IDS_mergejson/Image/'
    label_path='./IDS_mergejson/Label/'
    os.makedirs(img_savepath, exist_ok=True)
    os.makedirs(label_path,exist_ok=True)
    shutil.copy(file,os.path.join(img_savepath,patientid+'_'+name))
    shutil.copy(file.replace('Image','Label').replace('Extracted','Vis&Sub_json').replace('jpg','json'), os.path.join(label_path, patientid + '_' + name.replace('jpg','json')))
    print(patientid)
# print(images)