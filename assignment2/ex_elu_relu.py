# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.Elu_Relu import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Create some toy data to check your implementations
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  model = {}
  model['W1'] = np.linspace(-0.2, 0.6, num=input_size*hidden_size).reshape(input_size, hidden_size)
  model['b1'] = np.linspace(-0.3, 0.7, num=hidden_size)
  model['W2'] = np.linspace(-0.4, 0.1, num=hidden_size*num_classes).reshape(hidden_size, num_classes)
  model['b2'] = np.linspace(-0.5, 0.9, num=num_classes)
  return model

def init_toy_data():
  X = np.linspace(-0.2, 0.5, num=num_inputs*input_size).reshape(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y


# model = init_toy_model()
# X,y = init_toy_data()

# scores = elu(X,model)
# correct_scores = [[-0.5328368, 0.20031504, 0.93346689],
#  [-0.59412164, 0.15498488, 0.9040914 ],
#  [-0.67658362, 0.08978957, 0.85616275],
#  [-0.77092643, 0.01339997, 0.79772637],
#  [-0.89110401, -0.08754544, 0.71601312]]

# # the difference should be very small. We get 3e-8
# print 'Difference between your scores and correct scores:'
# print np.sum(np.abs(scores - correct_scores))



# loss and scores differ large ,but grads is ok....
# 
# reg = 0.1
# loss, _ = relu(X, model, y, reg)
# correct_loss = 1.38191946092

# # should be very small, we get 5e-12
# print 'Difference between your loss and correct loss:'
# print np.sum(np.abs(loss - correct_loss))

# elu & relu are ok
# from cs231n.gradient_check import eval_numerical_gradient

# loss, grads = elu(X, model, y, reg)

# # these should all be less than 1e-8 or so
# for param_name in grads:
#   param_grad_num = eval_numerical_gradient(lambda W: elu(X, model, y, reg)[0], model[param_name], verbose=False)
#   print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
###########loading datasets
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=9000, num_validation=1000, num_test=1000):
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
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test



X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

from cs231n.classifier_trainer import ClassifierTrainer

####don't forget init model every train nn
####
model = init_elu_relu_model(32*32*3, 100, 10) # input size, hidden size, number of classes

trainer = ClassifierTrainer()
ne = 20
_, loss_relu,train_acc,val_acc= trainer.train(X_train, y_train, X_val, y_val,
                                             model, relu,
                                             num_epochs=ne, reg=1.0,update='rmsprop',
                                             momentum=0.9, learning_rate_decay = 0.95,batch_size=80,
                                             learning_rate=1e-5, verbose=True)
model = init_elu_relu_model(32*32*3, 100, 10)
_, loss_leaky,train_acc_1,val_acc_1= trainer.train(X_train, y_train, X_val, y_val,
                                             model, leaky,
                                             num_epochs=ne, reg=1.0,update='rmsprop',batch_size=80,
                                             momentum=0.9, learning_rate_decay = 0.95,
                                             learning_rate=1e-5, verbose=True)
model = init_elu_relu_model(32*32*3, 100, 10)
_, loss_elu,train_acc_2,val_acc_2= trainer.train(X_train, y_train, X_val, y_val,
                                             model, elu,
                                             num_epochs=ne, reg=1.0,update='rmsprop',batch_size=80,
                                             momentum=0.9, learning_rate_decay = 0.95,
                                             learning_rate=1e-5, verbose=True)


# best validation accuracy: 0.447000
#######Debug the training
# Plot the loss function and train / validation accuracies
plt.style.use('ggplot')
plt.subplot(3, 1, 1)
plt.plot(loss_relu,'r')
plt.plot(loss_leaky,'y')
plt.plot(loss_elu,'b')
plt.legend(['-relu', '-leaky','-elu'], loc='upper right')
plt.title('Loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(3, 1, 2)
plt.plot(train_acc,'r-')
plt.plot(train_acc_1,'b-')
plt.plot(train_acc_2,'y-')
plt.legend(['T_relu', 'T_leaky','T_elu'], loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')

plt.subplot(3, 1, 3)
plt.plot(val_acc_1,'b-')
plt.plot(val_acc,'r-')
plt.plot(val_acc_2,'y-')
plt.legend(['V_relu','V_leaky','V_elu'], loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')

plt.show()
