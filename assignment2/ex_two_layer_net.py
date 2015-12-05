# A bit of setup

import numpy as np
import matplotlib.pyplot as plt

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
# X, y = init_toy_data()
##################################
###############compute scores
###################################
# from cs231n.classifiers.neural_net import two_layer_net

# scores = two_layer_net(X, model)
# # print scores
# correct_scores = [[-0.5328368, 0.20031504, 0.93346689],
#  [-0.59412164, 0.15498488, 0.9040914 ],
#  [-0.67658362, 0.08978957, 0.85616275],
#  [-0.77092643, 0.01339997, 0.79772637],
#  [-0.89110401, -0.08754544, 0.71601312]]

# the difference should be very small. We get 3e-8
# print 'Difference between your scores and correct scores:'
# print np.sum(np.abs(scores - correct_scores))


# reg = 0.1
# loss, _ = two_layer_net(X, model, y, reg)
# correct_loss = 1.38191946092

# should be very small, we get 5e-12
# print 'Difference between your loss and correct loss:'
# print np.sum(np.abs(loss - correct_loss))

# from cs231n.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

# loss, grads = two_layer_net(X, model, y, reg)

# these should all be less than 1e-8 or so
# for param_name in grads:
#   param_grad_num = eval_numerical_gradient(lambda W: two_layer_net(X, model, y, reg)[0], model[param_name], verbose=False)
#   print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

############ vanilla SGD##################
# from cs231n.classifier_trainer import ClassifierTrainer

# model = init_toy_model()
# trainer = ClassifierTrainer()
# # call the trainer to optimize the loss
# # Notice that we're using sample_batches=False, so we're performing Gradient Descent (no sampled batches of data)
# best_model, loss_history, _, _ = trainer.train(X, y, X, y,
#                                              model, two_layer_net,
#                                              reg=0.001,
#                                              learning_rate=1e-1, momentum=0.0, learning_rate_decay=1,
#                                              update='sgd', sample_batches=False,
#                                              num_epochs=100,
#                                              verbose=False)
# print 'Final loss with vanilla SGD: %f' % (loss_history[-1], )

##########momentum update#####################
# model = init_toy_model()
# trainer = ClassifierTrainer()
# # call the trainer to optimize the loss
# # Notice that we're using sample_batches=False, so we're performing Gradient Descent (no sampled batches of data)
# best_model, loss_history, _, _ = trainer.train(X, y, X, y,
#                                              model, two_layer_net,
#                                              reg=0.001,
#                                              learning_rate=1e-1, momentum=0.9, learning_rate_decay=1,
#                                              update='momentum', sample_batches=False,
#                                              num_epochs=100,
#                                              verbose=False)
# correct_loss = 0.494394
# print 'Final loss with momentum SGD: %f. We get: %f' % (loss_history[-1], correct_loss)

##############RMSProp###########################
# model = init_toy_model()
# trainer = ClassifierTrainer()
# # call the trainer to optimize the loss
# # Notice that we're using sample_batches=False, so we're performing Gradient Descent (no sampled batches of data)
# best_model, loss_history, _, _ = trainer.train(X, y, X, y,
#                                              model, two_layer_net,
#                                              reg=0.001,
#                                              learning_rate=1e-1, momentum=0.9, learning_rate_decay=1,
#                                              update='rmsprop', sample_batches=False,
#                                              num_epochs=100,
#                                              verbose=False)
# correct_loss = 0.439368
# print 'Final loss with RMSProp: %f. We get: %f' % (loss_history[-1], correct_loss)

########            ##########
########load the data#########
# from cs231n.data_utils import load_CIFAR10

# def get_CIFAR10_data(num_training=9000, num_validation=1000, num_test=1000):
#     """
#     Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
#     it for the two-layer neural net classifier. These are the same steps as
#     we used for the SVM, but condensed to a single function.  
#     """
#     # Load the raw CIFAR-10 data
#     cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
#     X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
#     # Subsample the data
#     mask = range(num_training, num_training + num_validation)
#     X_val = X_train[mask]
#     y_val = y_train[mask]
#     mask = range(num_training)
#     X_train = X_train[mask]
#     y_train = y_train[mask]
#     mask = range(num_test)
#     X_test = X_test[mask]
#     y_test = y_test[mask]

#     # Normalize the data: subtract the mean image
#     mean_image = np.mean(X_train, axis=0)
#     X_train -= mean_image
#     X_val -= mean_image
#     X_test -= mean_image

#     # Reshape data to rows
#     X_train = X_train.reshape(num_training, -1)
#     X_val = X_val.reshape(num_validation, -1)
#     X_test = X_test.reshape(num_test, -1)

#     return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
# X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
# print 'Train data shape: ', X_train.shape
# print 'Train labels shape: ', y_train.shape
# print 'Validation data shape: ', X_val.shape
# print 'Validation labels shape: ', y_val.shape
# print 'Test data shape: ', X_test.shape
# print 'Test labels shape: ', y_test.shape

###Train a network
# from cs231n.classifiers.neural_net import init_two_layer_model,two_layer_net
# from cs231n.classifier_trainer import ClassifierTrainer

# model = init_two_layer_model(32*32*3, 100, 10) # input size, hidden size, number of classes

# trainer = ClassifierTrainer()
# best_model, loss_history, train_acc, val_acc = trainer.train(X_train, y_train, X_val, y_val,
#                                              model, two_layer_net,
#                                              num_epochs=20, reg=1.0,
#                                              momentum=0.9, learning_rate_decay = 0.95,
#                                              learning_rate=1e-4, verbose=True)


# #best validation accuracy: 0.447000
# ########Debug the training
# # Plot the loss function and train / validation accuracies
# plt.subplot(2, 1, 1)
# plt.plot(loss_history)
# plt.title('Loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

# plt.subplot(2, 1, 2)
# plt.plot(train_acc)
# plt.plot(val_acc)
# plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
# plt.xlabel('Epoch')
# plt.ylabel('Clasification accuracy')
# plt.show()
# from cs231n.vis_utils import visualize_grid

# # Visualize the weights of the network

# def show_net_weights(model):
#     plt.imshow(visualize_grid(model['W1'].T.reshape(-1, 32, 32, 3), padding=3).astype('uint8'))
#     plt.gca().axis('off')
#     plt.show()

# show_net_weights(model)

# # Image processing via convolutions
# from scipy.misc import imread, imresize
# from cs231n.layers import conv_forward_naive

# kitten, puppy = imread('xuesen.jpg'), imread('puppy.jpg')
# # kitten is wide, and puppy is already square
# d = kitten.shape[0] - kitten.shape[1]
# kitten_cropped = kitten[d/2:-d/2, : , :]
# # print kitten_cropped.shape,puppy.shape

# img_size = 200   # Make this smaller if it runs too slow
# x = np.zeros((2, 3, img_size, img_size))
# # two function
# x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
# x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

# # Set up a convolutional weights holding 2 filters, each 3x3
# w = np.zeros((2, 3, 3, 3))

# # The first filter converts the image to grayscale.
# # Set up the red, green, and blue channels of the filter.
# w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
# w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
# w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

# # Second filter detects horizontal edges in the blue channel.
# w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# # Vector of biases. We don't need any bias for the grayscale
# # filter, but for the edge detection filter we want to add 128
# # to each output so that nothing is negative.
# b = np.array([0, 128])

# # Compute the result of convolving each input in x with each filter in w,
# # offsetting by b, and storing the results in out.
# out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})

# def imshow_noax(img, normalize=True):
#     """ Tiny helper to show images as uint8 and remove axis labels """
#     if normalize:
#         img_max, img_min = np.max(img), np.min(img)
#         img = 255.0 * (img - img_min) / (img_max - img_min)
#     plt.imshow(img.astype('uint8'))
#     plt.gca().axis('off')
#     #   get current axis

# # Show the original images and the results of the conv operation
# plt.subplot(2, 3, 1)
# imshow_noax(puppy, normalize=False)
# plt.title('Original image')
# plt.subplot(2, 3, 2)
# imshow_noax(out[0, 0])
# plt.title('Grayscale')
# plt.subplot(2, 3, 3)
# imshow_noax(out[0, 1])
# plt.title('Edges')

# plt.subplot(2, 3, 4)
# imshow_noax(kitten_cropped, normalize=False)
# plt.subplot(2, 3, 5)
# imshow_noax(out[1, 0])
# plt.subplot(2, 3, 6)
# imshow_noax(out[1, 1])
# plt.show()
