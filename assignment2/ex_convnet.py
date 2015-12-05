#Train a ConvNet!
# As usual, a bit of setup

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.classifiers.convnet import *
# from load_mnist import load_data

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    # X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    x_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
# X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
# print 'Train data shape: ', X_train.shape
# print 'Train labels shape: ', y_train.shape
# print 'Validation data shape: ', X_val.shape
# print 'Validation labels shape: ', y_val.shape
# print 'Test data shape: ', X_test.shape
# print 'Test labels shape: ', y_test.shape


##########Sanity check loss
# model = init_three_layer_convnet()

# X = np.random.randn(100, 3, 32, 32)
# y = np.random.randint(10, size=100)

# loss, _ = three_layer_convnet(X, model, y, reg=0)

# # Sanity check: Loss should be about log(10) = 2.3026
# print 'Sanity check loss (no regularization): ', loss

# # Sanity check: Loss should go up when you add regularization
# loss, _ = three_layer_convnet(X, model, y, reg=1)
# print 'Sanity check loss (with regularization): ', loss
########for super layer
# model = init_super_convnet()
# X = np.random.randn(100, 3, 32, 32)
# y = np.random.randint(10, size=100)
# loss,_ = super_convnet(X,model,y,reg=0)
# # Sanity check: Loss should be about log(10) = 2.3026
# print 'Sanity check loss (no regularization): ', loss
# loss, _ = super_convnet(X, model, y, reg=1)
# print 'Sanity check loss (with regularization): ', loss

##########Gradient check
##########for super convnet
num_inputs = 2
input_shape = (3, 16, 16)
reg = 0.0
num_classes = 10
X = np.random.randn(num_inputs, *input_shape)
y = np.random.randint(num_classes, size=num_inputs)

model = init_super_convnet(num_filters=3, filter_size=3, input_shape=input_shape)
loss, grads = super_convnet(X, model, y)
for param_name in sorted(grads):
    f = lambda _: super_convnet(X, model, y)[0]
    param_grad_num = eval_numerical_gradient(f, model[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
##############for three_layer##########
# num_inputs = 2
# input_shape = (3, 16, 16)
# reg = 0.0
# num_classes = 10
# X = np.random.randn(num_inputs, *input_shape)
# y = np.random.randint(num_classes, size=num_inputs)

# model = init_three_layer_convnet(num_filters=3, filter_size=3, input_shape=input_shape)
# loss, grads = three_layer_convnet(X, model, y)

# for param_name in sorted(grads):
#     f = lambda _: three_layer_convnet(X, model, y)[0]
#     param_grad_num = eval_numerical_gradient(f, model[param_name], verbose=False, h=1e-6)
#     e = rel_error(param_grad_num, grads[param_name])
#     print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
#####OK############UP#############
####Overfit small data and train visual
# Use a two-layer ConvNet to overfit 50 training examples.
# model = init_two_layer_convnet(filter_size=7)
# trainer = ClassifierTrainer()
# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#           X_train[:50], y_train[:50], X_test, y_test, model, two_layer_convnet,
#           reg=0.001, momentum=0.9, learning_rate=0.0001, batch_size=50, num_epochs=1,
#           acc_frequency=50, verbose=True)
# plt.subplot(2, 1, 1)
# plt.plot(loss_history)
# plt.xlabel('iteration')
# plt.ylabel('loss')

# plt.subplot(2, 1, 2)
# plt.plot(train_acc_history)
# plt.plot(val_acc_history)
# plt.legend(['train', 'val'], loc='upper left')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()

# from cs231n.vis_utils import visualize_grid

# grid = visualize_grid(best_model['W1'].transpose(0, 2, 3, 1))
# plt.imshow(grid.astype('uint8'))

# plt.show()