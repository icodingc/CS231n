####Modular neural nets
##########################
# As usual, a bit of setup

import numpy as np
import matplotlib.pyplot as plt
# from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# Test the affine_forward function

# num_inputs = 2
# input_shape = (4, 5, 6)
# output_dim = 3

# input_size = num_inputs * np.prod(input_shape)
# weight_size = output_dim * np.prod(input_shape)

# x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
# w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
# b = np.linspace(-0.3, 0.1, num=output_dim)

# out, _ = affine_forward(x, w, b)
# correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
#                         [ 3.25553199,  3.5141327,   3.77273342]])

# Compare your output with ours. The error should be around 1e-9.
# print 'Testing affine_forward function:'
# print 'difference: ', rel_error(out, correct_out)



# Test the affine_backward function

# x = np.random.randn(10, 2, 3)
# w = np.random.randn(6, 5)
# b = np.random.randn(5)
# dout = np.random.randn(10, 5)

# dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

# _, cache = affine_forward(x, w, b)
# dx, dw, db = affine_backward(dout, cache)

# The error should be less than 1e-10
# print 'Testing affine_backward function:'
# print 'dx error: ', rel_error(dx_num, dx)
# print 'dw error: ', rel_error(dw_num, dw)
# print 'db error: ', rel_error(db_num, db)

##############ReLU layer: forward
#################################
# Test the relu_forward function

# x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

# out, _ = relu_forward(x)
# correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
#                         [ 0.,          0.,          0.04545455,  0.13636364,],
#                         [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

# # Compare your output with ours. The error should be around 1e-8
# print 'Testing relu_forward function:'
# print 'difference: ', rel_error(out, correct_out)

###############ReLU layer: backward
####################################
# x = np.random.randn(10, 10)
# dout = np.random.randn(*x.shape)

# dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

# _, cache = relu_forward(x)
# dx = relu_backward(dout, cache)

# # The error should be around 1e-12
# print 'Testing relu_backward function:'
# print 'dx error: ', rel_error(dx_num, dx)
################Elu layer: forward
# x = np.linspace(-0.5,0.5,num=12).reshape(3,4)
# out,_ = leaky_forward(x)
# print x,"\n"
# print out,"\n"
# print np.exp(x) - 1
# using this to check forward
# expected = x.copy()
# for i in np.ndindex(x.shape):
# 	if x[i] < 0:
# 		expected[i] = np.exp(x[i]) - 1

# print 'error: ' ,rel_error(out,expected)
################Elu layer:backward
# x = np.random.randn(10, 10)
# dout = np.random.randn(*x.shape)

# dx_num = eval_numerical_gradient_array(lambda x: elu_forward(x)[0], x, dout)

# _, cache = elu_forward(x)
# dx = elu_backward(dout, cache)

# # The error should be around 1e-12
# print 'Testing relu_backward function:'
# print 'dx error: ', rel_error(dx_num, dx)

################for leaky_relu Forward
# x = np.linspace(-0.5,0.5,num=12).reshape(3,4)
# alpha = 0.5
# out,_ = leaky_forward(x)
# print x,"\n"
# # print out,"\n"
# expected = x.copy()
# for i in np.ndindex(x.shape):
# 	if x[i] < 0:
# 		expected[i] = alpha * x[i]

# print 'error: ' ,rel_error(out,expected)
###########################Leaky_relu backward
# x = np.random.randn(10, 10)
# dout = np.random.randn(*x.shape)

# dx_num = eval_numerical_gradient_array(lambda x: leaky_forward(x)[0], x, dout)

# _, cache = leaky_forward(x)
# dx = leaky_backward(dout, cache)

# # The error should be around 1e-12
# print 'Testing relu_backward function:'
# print 'dx error: ', rel_error(dx_num, dx)






#######Loss layers:Softmax and SVM
# num_classes, num_inputs = 10, 50
# x = 0.001 * np.random.randn(num_inputs, num_classes)
# y = np.random.randint(num_classes, size=num_inputs)

# dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
# loss, dx = svm_loss(x, y)

# # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
# print 'Testing svm_loss:'
# print 'loss: ', loss
# print 'dx error: ', rel_error(dx_num, dx)

# dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
# loss, dx = softmax_loss(x, y)

# # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
# print '\nTesting softmax_loss:'
# print 'loss: ', loss
# print 'dx error: ', rel_error(dx_num, dx)

#################ConvNets:forward naive
# x_shape = (2, 3, 4, 4)
# w_shape = (3, 3, 4, 4)
# x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
# w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
# b = np.linspace(-0.1, 0.2, num=3)

# conv_param = {'stride': 2, 'pad': 1}
# out, _ = conv_forward_naive(x, w, b, conv_param)
# correct_out = np.array([[[[[-0.08759809, -0.10987781],
#                            [-0.18387192, -0.2109216 ]],
#                           [[ 0.21027089,  0.21661097],
#                            [ 0.22847626,  0.23004637]],
#                           [[ 0.50813986,  0.54309974],
#                            [ 0.64082444,  0.67101435]]],
#                          [[[-0.98053589, -1.03143541],
#                            [-1.19128892, -1.24695841]],
#                           [[ 0.69108355,  0.66880383],
#                            [ 0.59480972,  0.56776003]],
#                           [[ 2.36270298,  2.36904306],
#                            [ 2.38090835,  2.38247847]]]]])

# # Compare your output to ours; difference should be around 1e-8
# print 'Testing conv_forward_naive'
# print 'difference: ', rel_error(out, correct_out)

###################CNN backward naive
# x = np.random.randn(4, 3, 5, 5)
# w = np.random.randn(2, 3, 3, 3)
# b = np.random.randn(2,)
# dout = np.random.randn(4, 2, 5, 5)
# conv_param = {'stride': 1, 'pad': 1}

# dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

# out, cache = conv_forward_naive(x, w, b, conv_param)
# dx, dw, db = conv_backward_naive(dout, cache)

# # Your errors should be around 1e-9'
# print 'Testing conv_backward_naive function'
# print 'dx error: ', rel_error(dx, dx_num)
# print 'dw error: ', rel_error(dw, dw_num)
# print 'db error: ', rel_error(db, db_num)


###########Max pooling layer: forward naive
##############################################
# x_shape = (2, 3, 4, 4)
# x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
# pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

# out, _ = max_pool_forward_naive(x, pool_param)

# correct_out = np.array([[[[-0.26315789, -0.24842105],
#                           [-0.20421053, -0.18947368]],
#                          [[-0.14526316, -0.13052632],
#                           [-0.08631579, -0.07157895]],
#                          [[-0.02736842, -0.01263158],
#                           [ 0.03157895,  0.04631579]]],
#                         [[[ 0.09052632,  0.10526316],
#                           [ 0.14947368,  0.16421053]],
#                          [[ 0.20842105,  0.22315789],
#                           [ 0.26736842,  0.28210526]],
#                          [[ 0.32631579,  0.34105263],
#                           [ 0.38526316,  0.4       ]]]])

# # Compare your output with ours. Difference should be around 1e-8.
# print 'Testing max_pool_forward_naive function:'
# print 'difference: ', rel_error(out, correct_out)
########################Max pooling layer: backward naive
##########################################################
# x = np.random.randn(3, 2, 8, 8)
# dout = np.random.randn(3, 2, 4, 4)
# pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)

# out, cache = max_pool_forward_naive(x, pool_param)
# dx = max_pool_backward_naive(dout, cache)

# # Your error should be around 1e-12
# print 'Testing max_pool_backward_naive function:'
# print 'dx error: ', rel_error(dx, dx_num)


# from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
# from time import time

# x = np.random.randn(100, 3, 31, 31)
# w = np.random.randn(25, 3, 3, 3)
# b = np.random.randn(25,)
# dout = np.random.randn(100, 25, 16, 16)
# conv_param = {'stride': 2, 'pad': 1}

# t0 = time()
# out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
# t1 = time()
# out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
# t2 = time()

# print 'Testing conv_forward_fast:'
# print 'Naive: %fs' % (t1 - t0)
# print 'Fast: %fs' % (t2 - t1)
# print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
# print 'Difference: ', rel_error(out_naive, out_fast)

# t0 = time()
# dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
# t1 = time()
# dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
# t2 = time()

# print '\nTesting conv_backward_fast:'
# print 'Naive: %fs' % (t1 - t0)
# print 'Fast: %fs' % (t2 - t1)
# print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
# print 'dx difference: ', rel_error(dx_naive, dx_fast)
# print 'dw difference: ', rel_error(dw_naive, dw_fast)
# print 'db difference: ', rel_error(db_naive, db_fast)
# Testing conv_forward_fast:
# Naive: 7.489650s
# Fast: 0.021881s
# Speedup: 342.288496x
# Difference:  2.22012639767e-11

# Testing conv_backward_fast:
# Naive: 10.225302s
# Fast: 0.015197s
# Speedup: 672.848324x
# dx difference:  2.27243098929e-11
# dw difference:  6.24449324563e-13
# db difference:  4.23575427937e-13




# from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast
# from time import time

# x = np.random.randn(100, 3, 32, 32)
# dout = np.random.randn(100, 3, 16, 16)
# pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# t0 = time()
# out_naive, cache_naive = max_pool_forward_naive(x, pool_param)
# t1 = time()
# out_fast, cache_fast = max_pool_forward_fast(x, pool_param)
# t2 = time()

# print 'Testing pool_forward_fast:'
# print 'Naive: %fs' % (t1 - t0)
# print 'fast: %fs' % (t2 - t1)
# print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
# print 'difference: ', rel_error(out_naive, out_fast)

# t0 = time()
# dx_naive = max_pool_backward_naive(dout, cache_naive)
# t1 = time()
# dx_fast = max_pool_backward_fast(dout, cache_fast)
# t2 = time()

# print '\nTesting pool_backward_fast:'
# print 'Naive: %fs' % (t1 - t0)     
# print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
# print 'dx difference: ', rel_error(dx_naive, dx_fast)
# Testing pool_forward_fast:
# Naive: 0.546278s
# fast: 0.003624s
# speedup: 150.740526x
# difference:  0.0

# Testing pool_backward_fast:
# Naive: 1.622273s
# speedup: 124.819878x
# dx difference:  0.0





####################Sandwich layers
######################################
# from cs231n.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward

# x = np.random.randn(2, 3, 16, 16) #  N, C, H, W = X.shape
# w = np.random.randn(3, 3, 3, 3)
# b = np.random.randn(3,)
# dout = np.random.randn(2, 3, 8, 8)
# conv_param = {'stride': 1, 'pad': 1}
# pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)
# dx, dw, db = conv_relu_pool_backward(dout, cache)

# dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b, dout)

# print 'Testing conv_relu_pool_forward:'
# print 'dx error: ', rel_error(dx_num, dx)
# print 'dw error: ', rel_error(dw_num, dw)
# print 'db error: ', rel_error(db_num, db)


# from cs231n.layer_utils import affine_relu_forward, affine_relu_backward

# x = np.random.randn(2, 3, 4)
# w = np.random.randn(12, 10)
# b = np.random.randn(10)
# dout = np.random.randn(2, 10)

# out, cache = affine_relu_forward(x, w, b)
# dx, dw, db = affine_relu_backward(dout, cache)

# dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

# print 'Testing affine_relu_forward:'
# print 'dx error: ', rel_error(dx_num, dx)
# print 'dw error: ', rel_error(dw_num, dw)
# print 'db error: ', rel_error(db_num, db)


#plot elu & leaky_relu
plt.style.use('ggplot')

x = np.linspace(-5,5,100)
y = relu_forward(x)[0]
# #'-' '--' '-.' 'o'  '.'
plt.plot(x,y,'')
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.show()