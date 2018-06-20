import random

import numpy as np
import cv2
import os
import scipy.io
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)


def mask_origin_image(input_dir, mask_dir, out_dir):
    for parent, dir_names, file_names in os.walk(input_dir):
        for origin_image in file_names:
            name = origin_image.split('.')[0]
            mask_file_name = os.path.join(mask_dir, name + 'mask.mat')
            image = cv2.imread(os.path.join(input_dir, origin_image))
            mask = scipy.io.loadmat(mask_file_name)['BW']

            mask_image = cv2.bitwise_and(np.array(image), np.array(image), mask=np.array(mask))
            cv2.imwrite(os.path.join(out_dir, name + '.jpg'), mask_image)
            # break


def list_shape(input_dir):
    for parent, dir_names, file_names in os.walk(input_dir):
        for origin_image in file_names:
            image = cv2.imread(os.path.join(input_dir, origin_image))
            print(image.shape)


def gen_density_map(input_dir, out_dir):
    for parent, dir_names, file_names in os.walk(input_dir):
        for txt_label in file_names:
            name = txt_label.split('.')[0]
            # h = 120
            # w = 160
            h = 60
            w = 80
            density_arr = np.zeros((h, w))

            with open(os.path.join(input_dir, txt_label), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    x = int(line.strip().split('	')[0]) // 8
                    y = int(line.strip().split('	')[1]) // 8
                    density_arr[y, x] = 1

            density = np.zeros(density_arr.shape, dtype=np.float32)

            gt_count = np.count_nonzero(density_arr)
            if gt_count == 0:
                return density

            mask = cv2.GaussianBlur(density_arr, (11, 11), sigmaX=3.0, sigmaY=3.0)
            print('car num is : {} and density num is {}'.format(len(lines), np.sum(mask)))
            # cv2.imwrite(os.path.join(out_dir, name + 'densitymap.jpg'), mask)
            np.save(os.path.join(out_dir, name + 'gt.npy'), mask)


def gen_train_test_txt(img_dir):
    for _, _, file_names in os.walk(img_dir):
        random.shuffle(file_names)
        train_num = int(0.7 * len(file_names))
        train_imgs = file_names[: train_num]
        test_imgs = file_names[train_num:]
    with open('train.txt', 'a') as train_file:
        for item in train_imgs:
            train_file.write(item + ' ' + item.split('.')[0] + 'gt.npy\n')
    with open('test.txt', 'a') as train_file:
        for item in test_imgs:
            train_file.write(item + ' ' + item.split('.')[0] + 'gt.npy\n')


if __name__ == '__main__':
    # mask_origin_image('D:\data\TRANCOS_v3\images\image',
    #                   'D:\data\TRANCOS_v3\images\mask',
    #                   'D:\data\TRANCOS_v3\images\masked_image')
    # list_shape('D:\data\TRANCOS_v3\images\masked_image')
    gen_density_map('D:\data\TRANCOS_v3\images\\txt',
                    'D:\data\TRANCOS_v3\images\\gt')
    # gen_train_test_txt('D:\data\TRANCOS_v3\images\masked_image')

