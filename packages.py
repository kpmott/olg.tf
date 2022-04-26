import numpy as np
import csv
import tensorflow as tf
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
import operator
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import time
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm
import os

print(os.system("which python3"))
print(os.system("python3 --version"))