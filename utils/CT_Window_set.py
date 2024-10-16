#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 9:51
# @File : CT_Window_set.py
# @annotation:
import SimpleITK as sitk
import os
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

if __name__ == '__main__':
    name_list = ['volume-covid19-A-0031.nii']
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
