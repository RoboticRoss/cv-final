import freenect
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

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth.astype(np.uint16)

