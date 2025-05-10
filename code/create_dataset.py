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

import cv2

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


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        choices=['1', '2', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), examining your model with LIME (2), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.weights.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .weights.h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--lime-image',
        default='test/Bedroom/image_0003.jpg',
        help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


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

    label_image = tf.image.grayscale_to_rgb(label_image)
    label_image = tf.image.resize(label_image, [224, 224])
    label_image = tf.cast(label_image > 21, tf.uint8)

    # normalize !! (comment out normalization to get good visuals when graphing)
    # depth_image = tf.image.convert_image_dtype(depth_image, tf.float32)
    depth_image = tf.image.grayscale_to_rgb(depth_image)
    depth_image = tf.image.resize(depth_image, [224, 224])
    depth_image = keras.applications.vgg19.preprocess_input(depth_image)

    return depth_image, label_image



def setup_model():
    
    input_layer = keras.Input(shape=(224, 224, 3))
    
    # uses PRE-TRAINED imagenet weights
    vgg_model = keras.applications.vgg19.VGG19(include_top=True, weights="imagenet", name="vgg19")

    backbone = keras.models.Model(
        inputs=vgg_model.layers[1].input,
        outputs=[
            vgg_model.get_layer("block5_pool").output 
           # for block_name in ["block3_pool", "block4_pool", "block5_pool"] # output at 3 stages to get more information
        ]
    )

    backbone.trainable = False
    x = backbone(input_layer)


    # replace dense layers with Conv2D layers
    dense_convs = []

    dense_conv_one = keras.layers.Conv2D(
        filters=4096,
        kernel_size=(7, 7),
        strides=(1, 1), 
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_initializer= keras.initializers.Constant(1.0)
    )

    dense_convs.append(dense_conv_one)
    dropout_layer_one = keras.layers.Dropout(0.5)
    dense_convs.append(dropout_layer_one)

    dense_conv_two = keras.layers.Conv2D(
        filters=4096,
        kernel_size=(1,1),
        strides=(1,1),
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_initializer=keras.initializers.Constant(1.0)
    )
    
    dense_convs.append(dense_conv_two)
    dropout_layer_two = keras.layers.Dropout(0.5)
    dense_convs.append(dropout_layer_two)

    dense_convs = keras.Sequential(dense_convs)
    dense_convs.trainable = False

    x[-1] = dense_convs(x[-1])

    pool = keras.layers.Conv2D(filters=2, kernel_size=(1,1), padding="same", strides=(1, 1), activation="relu", name="pooling")
    softmax = keras.layers.Conv2D(filters=2, kernel_size=(1,1), padding="same", strides=(1, 1), activation="softmax", name="softmax")
    upsample = keras.layers.UpSampling2D(size=(32, 32), data_format=keras.backend.image_data_format(), interpolation="bilinear", name="upsample_one")

    head_one = pool(x[0])
    head_two = softmax(head_one)
    final_output = upsample(head_two)

    seg_model = keras.Model(inputs = input_layer, outputs=final_output)

    # load weights into the dense_conv layers
    weights1 = vgg_model.get_layer("fc1").get_weights()[0]
    weights2 = vgg_model.get_layer("fc2").get_weights()[0]
    weights1 = weights1.reshape(7, 7, 512, 4096)
    weights2 = weights2.reshape(1, 1, 4096, 4096)
    dense_convs.layers[0].set_weights([weights1])
    dense_convs.layers[2].set_weights([weights2])

    return seg_model


def train(model, train_data, validation_data, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # dataset1 = train_data.map(lambda x: {'data': x, 'task': 'task1'})
    # dataset2 = validation_data.map(lambda x: {'data': x, 'task': 'task2'})
    
    dataset = train_data.concatenate(validation_data)

    # Keras callbacks for training
    callback_list = [
        keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, train_data, validation_data),
        CustomModelSaver(checkpoint_path, 1, hp.max_num_weights)
    ]

    # Begin training
    model.fit(
        x=train_data,
        validation_data=validation_data,
        epochs=20,
        steps_per_epoch=100, 
        validation_steps=100,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        steps= 50,
        verbose=1,
    )

def visualize_prediction(test_dataset, batch_size, model):
    # images, masks = next(iter(test_dataset))
    # random_idx = keras.random.uniform([], minval=0, maxval=batch_size, seed=10)

    # # Get random test image and mask
    # test_image = images[int(random_idx)].numpy().astype("float")
    # test_mask = masks[int(random_idx)].numpy().astype("float")

    # # depth_image = tf.image.convert_image_dtype(test_image, tf.float32)
    # #depth_image = tf.image.grayscale_to_rgb(depth_image)
    # # depth_image = tf.image.resize(depth_image, [224, 224])
    # # depth_image = np.expand_dims(test_image, axis=0)
    # # depth_image = keras.applications.vgg19.preprocess_input(depth_image)

    # # Perform inference on FCN-32S
    # pred_mask = model(test_image)


    image_path = '/Volumes/Random Stuff/dex-ycb-20210415/20200813-subject-02/20200813_154304/836212060125/aligned_depth_to_color_000000.png'
    mask_path = '/Volumes/Random Stuff/dex-ycb-20210415/20200813-subject-02/20200813_154304/836212060125/labels_000000.npz'

    image, mask = data_preprocess(image_path, mask_path)
    image = np.expand_dims(image, axis=0)

    
    pred_mask = model(image)

    pred_mask = model.predict(image)


    model.evaluate(pred_mask, mask)
    
    


    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0, ...]
    # gets rid of batch dimension!


    # Plot all results
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))

    # ax[0, 0].set_title("Image")
    # ax[0, 0].imshow(test_image)

    # ax[0, 1].set_title("Image with ground truth overlay")
    # ax[0, 1].imshow(test_image / 255.0)
    # ax[0, 1].imshow(
    #     test_mask,
    #     cmap="inferno",
    #     alpha=0.6,
    # )

    # ax[1, 0].set_title("Image with Predicted Mask overlay")
    # ax[1, 0].imshow(test_image)
    # ax[1, 0].imshow(pred_mask, cmap="inferno", alpha=0.6)

    # plt.show()


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
    batch_size = 32

    # SETUP DATASETS
    train_depth_paths, train_label_paths = extract_paths('train')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_depth_paths, train_label_paths))
    train_dataset = train_dataset.map(data_preprocess, num_parallel_calls=tf.data.AUTOTUNE) # allows the images to be processed in parallel!
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) # uses a background thread (speeds up pipeline)

    val_depth_paths, val_label_paths = extract_paths('val')
    val_dataset = tf.data.Dataset.from_tensor_slices((val_depth_paths, val_label_paths))
    val_dataset = val_dataset.map(data_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_depth_paths, test_label_paths = extract_paths('test')
    test_dataset = tf.data.Dataset.from_tensor_slices((test_depth_paths, test_label_paths))
    test_dataset = test_dataset.map(data_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # # Get random test image and mask
    # test_image = images[int(10)].numpy().astype("float")
    # test_mask = masks[int(10)].numpy().astype("float")

    # plt.imshow(test_image)
    # plt.imshow(test_mask)


    # SETUP MODEL
    seg_model = setup_model() 
    seg_optimizer = keras.optimizers.Adam(learning_rate = 1e-3, weight_decay = 1e-4)
    seg_loss = keras.losses.SparseCategoricalCrossentropy()

    seg_model.compile(
        optimizer=seg_optimizer,
        loss=seg_loss, 
        metrics=[
            keras.metrics.SparseCategoricalAccuracy()
        ]
    )
    
    seg_model.summary()
    
    checkpoint_path = "checkpoints" + os.sep + \
            "segmentation_model" + os.sep + timestamp + os.sep
    
    logs_path = "logs" + os.sep + "segmentation_model" + \
            os.sep + timestamp + os.sep
    
    # Make checkpoint directory if needed
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    seg_model.load_weights('/Users/isabella/Desktop/CSCI1430_Projects/checkpoints/segmentation_model/050925-164744/sege000-acc0.9871.weights.h5')
    visualize_prediction(test_dataset, batch_size, seg_model)
    
    # train(seg_model, train_dataset, val_dataset, checkpoint_path, logs_path, init_epoch)

    # test(seg_model, test_dataset)


ARGS = parse_args()
main()