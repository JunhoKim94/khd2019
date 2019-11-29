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


def image_preprocessing(image, target_resolution, normalize):
    if image.shape[:2] != target_resolution:
        res = cv2.resize(image,
                         target_resolution,
                         interpolation=cv2.INTER_AREA)

    if normalize == True:
        res = res / 255.

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


def dataset_loader(img_path,
                   target_resolution,
                   normalize):
    t1 = time.time()

    ## 이미지 읽기
    p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path) for f in files if all(s in f for s in ['.jpg'])]
    p_list.sort()

    images = []
    labels = []
    for i, p in enumerate(tqdm(p_list)):
        im = cv2.imread(p, 3)
        im = image_preprocessing(im,
                                 target_resolution,
                                 normalize)
        images.append(im)
        
        # label 데이터 생성
        l = Label2Class(p.split('/')[-2])
        labels.append(l)

    images = np.array(images)
    labels = np.array(labels)

    t2 = time.time()
    print('Dataset prepared for' ,t2 -t1 ,'sec')
    print('Images:' ,images.shape ,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, ' among 0-3 classes')

    return images, labels


def cutmix(x, y):
    x = x.copy()
    y = y.copy()
    
    batch_size, height, width, channel = x.shape
    
    lmbda = np.random.uniform()
    sqrt_lmbda = np.sqrt(lmbda)
    rand_index = np.random.permutation(batch_size)

    offset_x = int(width * sqrt_lmbda)
    offset_y = int(height * sqrt_lmbda)

    left = np.random.randint(0, width-offset_x+1)
    right = left + offset_x
    down = np.random.randint(0, height-offset_y+1)
    up = down + offset_y
    
    # input reconstruction
    permutation = np.random.permutation(batch_size)
    x[:, down:up, left:right] = x[permutation, down:up, left:right]

    # label reconstruction
    lmbda = (right-left)*(up-down)/(height*width)
    
    y = (1-lmbda)*y + lmbda*y[permutation]
    
    return x, y


def make_dataset(dataset, batch_size=32):
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(batch_size=batch_size,
                            drop_remainder=True)
    dataset = dataset.map(map_func=cutmix,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
    

def cutmix_gen(x, y, batch_size=32):
    total = x.shape[0]
    xs, ys = [], []
    
    for i in range(total//batch_size):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]
        yield cutmix(x_batch, y_batch)

def apply_cutmix(x, y, batch_size=32):
    total = x.shape[0]
    xs, ys = [], []

    for i in range(total//batch_size):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]
        xs.append(x_batch)
        ys.append(y_batch)
    
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class CutmixGen(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size :
                        (idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size :
                        (idx+1)*self.batch_size]

        return cutmix(batch_x, batch_y)

    

if __name__ == '__main__':
    data_size = 4
    x = np.zeros([data_size, 6, 6, 1])
    y = np.random.randint(4, size=(data_size,))
    for i, _y in enumerate(y):
        x[i] = _y
    y = np.eye(4)[y]

    inputs = tf.keras.layers.Input((x.shape[1:]))
    _x = tf.keras.layers.Dense(10, activation='relu')(inputs)
    _x = tf.keras.layers.GlobalMaxPool2D()(_x)
    _x = tf.keras.layers.Dense(4, activation='softmax')(_x)
    model = tf.keras.Model(inputs=inputs, outputs=_x)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam())
       
    gen = CutmixGen(x, y, batch_size=4)
    model.fit(gen, epochs=10)
