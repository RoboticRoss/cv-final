# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of creating DexYCB datasets."""

import json
from dex_ycb_toolkit.factory import get_dataset
import h5py

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from models import YourModel, VGGModel, SegmentationModel

import os
import sys
import argparse
import re
from datetime import datetime

import hyperparameters as hp
from models import YourModel, VGGModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries



def extract_paths(split): 
  """
  extract the images from their filepaths
  """

  setup = 's0'
  name =  name = '{}_{}'.format(setup, split)
  print('Dataset name: {}'.format(name))
      
  dataset = get_dataset(name)

  depth_image_paths = np.array([x['depth_file'] for x in dataset])
  label_image_paths = np.array([x['label_file'] for x in dataset])

  return depth_image_paths, label_image_paths


def load_npz(npz):
   """
   For loading in npz files as the actual segmentation map image array
   """
   npz = np.load(npz.numpy().decode(('utf-8)')))
   return npz['seg']

@tf.function
def data_preprocess(image_path, label_path):
    """
    Preprocesses the data to be used in Tensorflow datasets
    """
    depth_image = tf.io.read_file(image_path)
    depth_image = tf.image.decode_png(depth_image, channels=1)
    # depth_image shape: [480, 640, 1]

    label_image = tf.py_function(func=load_npz, inp=[label_path], Tout=tf.float32)
    label_image = tf.expand_dims(label_image, axis=2)
    label_image = tf.ensure_shape(label_image, [480, 640, 1])
   # label_image = tf.image.grayscale_to_rgb(label_image)
    label_image = tf.image.resize(label_image, [244, 244])
    label_image = tf.cast(label_image > 21, tf.uint8)

    # normalize !!
    depth_image = tf.image.convert_image_dtype(depth_image, tf.float32)
    # depth_image = keras.applications.vgg19.preprocess_input(depth_image)
    depth_image = tf.image.grayscale_to_rgb(depth_image)
    depth_image = tf.image.resize(depth_image, [244, 244])

    return depth_image, label_image


def train(model, train_data, validation_data, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # # Keras callbacks for training
    # callback_list = [
    #     keras.callbacks.TensorBoard(
    #         log_dir=logs_path,
    #         update_freq='batch',
    #         profile_batch=0),
    #     ImageLabelingLogger(logs_path, train_data),
    #     CustomModelSaver(checkpoint_path, 1, hp.max_num_weights)
    # ]

    # Begin training
    model.fit(
        x=train_data,
        validation_data=validation_data,
        epochs=20,
        batch_size=32,            
        
        initial_epoch=init_epoch,
    )

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def show_mask(dataset, batch_size):
    """
    KERAS visualization code - https://keras.io/examples/vision/fully_convolutional_network/
    """
    images, masks = next(iter(dataset))
    random_idx = keras.random.uniform([], minval=0, maxval=batch_size, seed=10)

    test_image = images[int(random_idx)].numpy().astype("float")
    test_mask = masks[int(random_idx)].numpy().astype("float")

    # Overlay segmentation mask on top of image.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].set_title("Image")
    ax[0].imshow(test_image / 255.0)

    ax[1].set_title("Image with segmentation mask overlay")
    ax[1].imshow(test_image / 255.0)
    ax[1].imshow(
        test_mask,
        cmap="inferno",
        alpha=0.6,
    )
    plt.show()

def main():
    
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    train_depth_paths, train_label_paths = extract_paths('train')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_depth_paths, train_label_paths))

    # depth_path, label_path = next(iter(train_dataset))
    # data_preprocess(depth_path, label_path)


    train_dataset = train_dataset.map(data_preprocess, num_parallel_calls=tf.data.AUTOTUNE) # allows the images to be processed in parallel!

    batch_size = 32
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) # uses a background thread (speeds up pipeline)

    # show_mask(train_dataset, batch_size)

    test_depth_paths, test_label_paths = extract_paths('test')
    test_dataset = tf.data.Dataset.from_tensor_slices((test_depth_paths, test_label_paths))
    test_dataset = test_dataset.map(data_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    test_dataset = test_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = SegmentationModel()
    model(keras.Input(shape=(224, 224, 3)))

    model.vgg19.summary()
    model.head.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])


    checkpoint_path = "checkpoints" + os.sep + \
            "segmentation_model" + os.sep + timestamp + os.sep
    
    logs_path = "logs" + os.sep + "segmentation_model" + \
            os.sep + timestamp + os.sep
    
    train(model, train_dataset, test_dataset, checkpoint_path, logs_path, init_epoch)






if __name__ == '__main__':
  main()