import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, Reshape
tf.config.run_functions_eagerly(True)

from functools import reduce  # Required in Python 3
import operator

from tqdm import tqdm
from tqdm.keras import TqdmCallback
import time

#from itertools import product
import datetime

import matplotlib.pyplot as plt

#from sklearn import metrics
from scipy.optimize import fsolve
from scipy.stats import norm

#import io
import os
import platform



gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)