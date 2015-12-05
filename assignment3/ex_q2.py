
# A bit of setup

import numpy as np
import matplotlib.pyplot as plt
from time import time

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# NOTE: The full TinyImageNet dataset will take up about 490MB of disk space, 
# and loading the full TinyImageNet-100-A dataset into memory will use about 2.8GB of memory.
from cs231n.data_utils import load_tiny_imagenet

tiny_imagenet_a = 'cs231n/datasets/tiny-imagenet-100-A'
        
class_names, X_train, y_train, X_val, y_val, X_test, y_test = load_tiny_imagenet(tiny_imagenet_a)

# Zero-mean the data
mean_img = np.mean(X_train, axis=0)
X_train -= mean_img
X_val -= mean_img
X_test -= mean_img
