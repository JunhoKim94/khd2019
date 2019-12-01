import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np
import tensorflow as tf
import keras
from multiprocessing import Pool
from tqdm import tqdm
from keras_preprocessing import image as ksimg


def image_preprocessing(image, target_resolution, normalize):
# def image_preprocessing(image, rescale, resize_factor):
    
    if image.shape[:2] != target_resolution:
        res = cv2.resize(image,
                         target_resolution,
                         interpolation=cv2.INTER_AREA)

    if normalize == True:
        res = (res / 255.).astype(np.float32)


    return res


def Label2Class(label):
    # one hot encoding (0-3 --> [., ., ., .])
    if label == 'AMD':
        cls = 1
    elif label == 'RVO':
        cls = 2
    elif label == 'DMR':
        cls = 3
    else:
        cls = 0
    
    resvec = [0, 0, 0, 0]
    resvec[cls] = 1
    return resvec


def map_fn(x):
    return image_preprocessing(cv2.imread(x, 3), (390, 307), True)


def dataset_loader(img_path,
                   target_resolution,
                   normalize):
    t1 = time.time()

    
    ## 이미지 읽기
    p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path) for f in files if all(s in f for s in ['.jpg'])]
    p_list.sort()

    with Pool(32) as p:
        images = p.map(map_fn, tqdm(p_list))
    
    print("Mapping_Finished")
    
    labels = []
    for i, p in enumerate(p_list):
        labels.append(Label2Class(p.split('/')[-2]))

    images = np.array(images).astype(np.float32)
    labels = np.array(labels)

    t2 = time.time()
    print('Dataset prepared for' ,t2 -t1 ,'sec')
    print('Images:' ,images.shape ,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, ' among 0-3 classes')

    return images, labels