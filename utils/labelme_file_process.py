#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 9:16
# @File : labelme_file_process.py
# @annotation:
# -----------------------------------labelme 标签批量转化，注意这个函数文件夹里必须只含有JSON文件-----------
import json
import os
import os.path as osp
import warnings
import PIL.Image
import cv2
import yaml
from labelme import utils
from skimage import img_as_ubyte
def process_labelme_file(json_dir):
    list_path = os.listdir(json_dir)
    print('freedom =', json_dir)
    for i in range(0, len(list_path)):
        path = os.path.join(json_dir, list_path[i])
        if os.path.isfile(path):
            data = json.load(open(path))
            img = utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = utils.draw_label(lbl, img, captions)
            out_dir = osp.basename(path).replace('.', '_')
            save_file_name = out_dir
            out_dir = osp.join(osp.dirname(path), out_dir)
            if not osp.exists(json_dir + '\\' + 'labelme_json'):
                os.mkdir(json_dir + '\\' + 'labelme_json')
            labelme_json = json_dir + '\\' + 'labelme_json'
            out_dir1 = labelme_json + '\\' + save_file_name
            if not osp.exists(out_dir1):
                os.mkdir(out_dir1)
            PIL.Image.fromarray(img).save(out_dir1 + '\\' + save_file_name + '_img.png')
            PIL.Image.fromarray(lbl).save(out_dir1 + '\\' + save_file_name + '_label.png')
            PIL.Image.fromarray(lbl_viz).save(out_dir1 + '\\' + save_file_name +
                                              '_label_viz.png')
            if not osp.exists(json_dir + '\\' + 'mask_png'):
                os.mkdir(json_dir + '\\' + 'mask_png')
            mask_save2png_path = json_dir + '\\' + 'mask_png'
            mask_dst = img_as_ubyte(lbl)  # mask_pic
            print('pic2_deep:', mask_dst.dtype)
            cv2.imwrite(mask_save2png_path + '\\' + save_file_name + '_label.png', mask_dst)
            with open(osp.join(out_dir1, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')
            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir1, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)
            print('Saved to: %s' % out_dir1)

if __name__ == '__main__':
    json_dir = '../data/labelme'
    process_labelme_file(json_dir)