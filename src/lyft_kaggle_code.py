
## This is KAGGLE Code -- mostly taken as is from the links mentioned below as SOURCE Links 
## SOURCE -1 -->> https://www.kaggle.com/code/tarunpaparaju/lyft-competition-understanding-the-data#Visualizing-the-data


import os , gc , numpy as np , pandas as pd
#import gc
# import numpy as np
# import pandas as pd

import json , math , sys , time 
# import math
# import sys
# import time
from datetime import datetime
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image

from matplotlib.axes import Axes
from matplotlib import animation, rc
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot, init_notebook_mode
import plotly.figure_factory as ff

#init_notebook_mode(connected=True)

import seaborn as sns
from pyquaternion import Quaternion
from tqdm import tqdm

from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from pathlib import Path

import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import copy


DATA_PATH = './input_dir/3d-object-detection-for-autonomous-vehicles/'
train = pd.read_csv(DATA_PATH + 'train.csv')
#sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')"