"""
Homework 5 - CNNs
CSCI1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel, VGGModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

from PIL import Image
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def classify(image_path, model):
    image = image_path
    
    original = Image.open(image)
    img = original.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = keras.applications.vgg16.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)

    print("SHAPE", np.shape(img))

    # depth_image shape: [480, 640, 1]

    # {0: 'closed', 1: 'open'}
    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis=1)
    
    if (prediction):
        pred = 'open'
    else: 
        pred = 'closed'
    
    plt.imshow(original)
    plt.title(pred)
    plt.show()


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.weights.h5)"

        print(regex)
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of main.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # Run script from location of main.py
    os.chdir(sys.path[0])

    model = VGGModel()
    checkpoint_path = "checkpoints" + os.sep + \
        "vgg_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "vgg_model" + \
        os.sep + timestamp + os.sep
    model(tf.keras.Input(shape=(224, 224, 3)))

    # Print summaries for both parts of the model
    model.vgg16.summary()
    model.head.summary()

    # Load base of VGG model
    model.vgg16.load_weights(ARGS.load_vgg)
    model.head.load_weights('/Users/isabella/Desktop/CSCI1430_Projects/cv-final/code/checkpoints/vgg_model/050625-211806/vgg.e018-acc0.9679.weights.h5')

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    image = '/Users/isabella/Desktop/CSCI1430_Projects/cv-final/data/test/open/segmentada6.png'
    
    classify(image, model)

# Make arguments global
main()