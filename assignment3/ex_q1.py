import numpy as np
import matplotlib.pyplot as plt
from time import time
from cs231n.layers import *
from cs231n.fast_layers import *
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# from cs231n.data_utils import load_CIFAR10

# def get_CIFAR10_data(num_training=500, num_validation=1000, num_test=1000, normalize=True):
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
#     if normalize:
#         mean_image = np.mean(X_train, axis=0)
#         X_train -= mean_image
#         X_val -= mean_image
#         X_test -= mean_image
    
#     # Transpose so that channels come first
#     X_train = X_train.transpose(0, 3, 1, 2).copy()
#     X_val = X_val.transpose(0, 3, 1, 2).copy()
#     X_test = X_test.transpose(0, 3, 1, 2).copy()

#     return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
# X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(num_training=500)
# print 'Train data shape: ', X_train.shape

# from cs231n.classifiers.convnet import *
# from cs231n.classifier_trainer import ClassifierTrainer

# model = init_three_layer_convnet(filter_size=5, num_filters=(32, 128))
# trainer = ClassifierTrainer()
# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#           X_train, y_train, X_val, y_val, model, three_layer_convnet, dropout=None,
#           reg=0.05, learning_rate=0.00005, batch_size=50, num_epochs=15,
#           learning_rate_decay=1.0, update='rmsprop', verbose=True)

# plt.subplot(2, 1, 1)
# plt.plot(train_acc_history)
# plt.plot(val_acc_history)
# plt.title('accuracy vs time')
# plt.legend(['train', 'val'], loc=4)
# plt.xlabel('epoch')
# plt.ylabel('classification accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss_history)
# plt.title('loss vs time')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.show()

# Check the dropout forward pass

# x = np.random.randn(100, 100)
# dropout_param_train = {'p': 0.25, 'mode': 'train'}
# dropout_param_test = {'p': 0.25, 'mode': 'test'}

# out_train, _ = dropout_forward(x, dropout_param_train)
# out_test, _ = dropout_forward(x, dropout_param_test)

# # Test dropout training mode; about 25% of the elements should be nonzero
# print np.mean(out_train != 0)

# # Test dropout test mode; all of the elements should be nonzero
# print np.mean(out_test != 0)

# from cs231n.gradient_check import eval_numerical_gradient_array

# # Check the dropout backward pass

# x = np.random.randn(5, 4)
# dout = np.random.randn(*x.shape)
# dropout_param = {'p': 0.8, 'mode': 'train', 'seed': 123}

# dx_num = eval_numerical_gradient_array(lambda x: dropout_forward(x, dropout_param)[0], x, dout)

# _, cache = dropout_forward(x, dropout_param)
# dx = dropout_backward(dout, cache)

# # The error should be around 1e-12
# print 'Testing dropout_backward function:'
# print 'dx error: ', rel_error(dx_num, dx)


#####Data Augmentation
# from cs231n.data_augmentation import *

# X = get_CIFAR10_data(num_training=100, normalize=False)[0]
# num_imgs = 8
# print X.dtype
# X = X[np.random.randint(100, size=num_imgs)]

# X_flip = random_flips(X)
# X_rand_crop = random_crops(X, (28, 28))

# # To give more dramatic visualizations we use large scales for random contrast
# # and tint adjustment.
# X_contrast = random_contrast(X, scale=(0.5, 1.0))
# X_tint = random_tint(X, scale=(-50, 50))

# next_plt = 1
# for i in xrange(num_imgs):
#     titles = ['original', 'flip', 'rand crop', 'contrast', 'tint']
#     for j, XX in enumerate([X, X_flip, X_rand_crop, X_contrast, X_tint]):
#         plt.subplot(num_imgs, 5, next_plt)
#         img = XX[i].transpose(1, 2, 0)
#         if j == 4:
#             # For visualization purposes we rescale the pixel values of the
#             # tinted images
#             low, high = np.min(img), np.max(img)
#             img = 255 * (img - low) / (high - low)
#         plt.imshow(img.astype('uint8'))
#         if i == 0:
#             plt.title(titles[j])
#         plt.gca().axis('off')
#         next_plt += 1
# plt.show()

# input_shape = (3, 28, 28)

# def augment_fn(X):
#     out = random_flips(random_crops(X, input_shape[1:]))
#     out = random_tint(random_contrast(out))
#     return out

# def predict_fn(X):
#     return fixed_crops(X, input_shape[1:], 'center')
    
# model = init_three_layer_convnet(filter_size=5, input_shape=input_shape, num_filters=(32, 128))
# trainer = ClassifierTrainer()

# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#           X_train, y_train, X_val, y_val, model, three_layer_convnet,
#           reg=0.05, learning_rate=0.00005, learning_rate_decay=1.0,
#           batch_size=50, num_epochs=30, update='rmsprop', verbose=True, dropout=0.6,
#           augment_fn=augment_fn, predict_fn=predict_fn)

# Visualize the loss and accuracy for our network trained with dropout and data augmentation.
# You should see less overfitting, and you may also see slightly better performance on the
# validation set.

# plt.subplot(2, 1, 1)
# plt.plot(train_acc_history)
# plt.plot(val_acc_history)
# plt.title('accuracy vs time')
# plt.legend(['train', 'val'], loc=4)
# plt.xlabel('epoch')
# plt.ylabel('classification accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss_history)
# plt.title('loss vs time')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.show()





















