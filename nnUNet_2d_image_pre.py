
# 2D   TO   3D
import os
import random
from tqdm import tqdm
import SimpleITK as sitk
import cv2
import numpy as np
def SplitDataset(img_path, train_percent=0.9):
    data = os.listdir(img_path)
    train_images = []
    test_images = []
    num = len(data)
    train_num = int(num * train_percent)
    indexes = list(range(num))
    train = random.sample(indexes, train_num)
    for i in indexes:
        if i in train:
            train_images.append(data[i])
        else:
            test_images.append(data[i])
    return train_images, test_images


def conver(img_path, save_dir, mask_path=None, select_condition=None, mode="trian"):
    os.makedirs(save_dir, exist_ok=True)
    if mode == "train":
        savepath_img = os.path.join(save_dir, 'imagesTr')
        savepath_mask = os.path.join(save_dir, 'labelsTr')
    elif mode == "test":
        savepath_img = os.path.join(save_dir, 'imagesTs')
        savepath_mask = os.path.join(save_dir, 'labelsTs')
    os.makedirs(savepath_img, exist_ok=True)
    if mask_path is not None:
        os.makedirs(savepath_mask, exist_ok=True)

    ImgList = os.listdir(img_path)
    with tqdm(ImgList, desc="conver") as pbar:
        for name in pbar:
            if select_condition is not None and name not in select_condition:
                continue
            Img = cv2.imread(os.path.join(img_path, name))
            if mask_path is not None:
                Mask = cv2.imread(os.path.join(mask_path, name), 0)
                Mask = (Mask / 255).astype(np.uint8)
                if Img.shape[:2] != Mask.shape:
                    Mask = cv2.resize(Mask, (Img.shape[1], Img.shape[0]))
            Img_Transposed = np.transpose(Img, (2, 0, 1))
            Img_0 = Img_Transposed[0].reshape(1, Img_Transposed[0].shape[0], Img_Transposed[0].shape[1])
            Img_1 = Img_Transposed[1].reshape(1, Img_Transposed[1].shape[0], Img_Transposed[1].shape[1])
            Img_2 = Img_Transposed[2].reshape(1, Img_Transposed[2].shape[0], Img_Transposed[2].shape[1])
            if mask_path is not None:
                Mask = Mask.reshape(1, Mask.shape[0], Mask.shape[1])

            Img_0_name = name.split('.')[0] + '_0000.nii.gz'
            Img_1_name = name.split('.')[0] + '_0001.nii.gz'
            Img_2_name = name.split('.')[0] + '_0002.nii.gz'
            if mask_path is not None:
                Mask_name = name.split('.')[0] + '.nii.gz'

            Img_0_nii = sitk.GetImageFromArray(Img_0)
            Img_1_nii = sitk.GetImageFromArray(Img_1)
            Img_2_nii = sitk.GetImageFromArray(Img_2)
            if mask_path is not None:
                Mask_nii = sitk.GetImageFromArray(Mask)

            sitk.WriteImage(Img_0_nii, os.path.join(savepath_img, Img_0_name))
            sitk.WriteImage(Img_1_nii, os.path.join(savepath_img, Img_1_name))
            sitk.WriteImage(Img_2_nii, os.path.join(savepath_img, Img_2_name))
            if mask_path is not None:
                sitk.WriteImage(Mask_nii, os.path.join(savepath_mask, Mask_name))

from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)
if __name__ == "__main__":
    # train_percent = 0.9
    # img_path = r"KMC/Image"
    # mask_path = r"KMC/label"
    # output_folder = r"./out"
    # os.makedirs(output_folder, exist_ok=True)
    # train_images, test_images = SplitDataset(img_path, train_percent)
    # conver(img_path, output_folder, mask_path, train_images, mode="train")
    # conver(img_path, output_folder, mask_path, test_images, mode="test")
    output_file='out/data.json'
    imagesTr_dir='out/imagesTr'
    imagesTs_dir='out/imagesTs'
    modalities=('T1', 'T2', 'FLAIR')
    labels={0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    dataset_name='KMC'
    generate_dataset_json(output_file,imagesTr_dir,imagesTs_dir,modalities,labels,dataset_name)