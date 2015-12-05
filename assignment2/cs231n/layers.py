# -*- coding:utf-8 -*-
#xuesen 
import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = x.reshape((x.shape[0],-1)).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x.reshape((x.shape[0],-1)).T.dot(dout)
  db = np.sum(dout,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db

############################Action function Begin###############################
def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0, x)
  cache = x
  return out, cache

def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).
  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout
  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = np.where(x > 0, dout, 0)
  return dx

#alpah>0
#care about deep copy!!
#跟以上比较一个是由函数生成，一个是直接赋值
#以前的问题都出在这里
#elu_forward有两个输出，考虑out的时候要取[0]
#https://github.com/muupan/chainer-elu/blob/master/elu.py
def elu_forward(x,alpha=1.0):
  # out = np.where(x>0,x,alpha*(np.exp(x) - 1))
  out = x.copy()
  neg_indices = (x < 0)
  out[neg_indices] = (alpha * (np.exp(out[neg_indices]) - 1))
  cache = x,alpha
  return out,cache

def elu_backward(dout,cache):
  x,alpha= cache
  dx = dout.copy()
  neg_indices = (x<0)
  dx[neg_indices] *= alpha*np.exp(x[neg_indices])
  # dx = np.where(x>=0,dout,dout*(elu_forward(x)[0] + alpha) )
  return dx


def leaky_forward(x,alpha=0.5):
  out = np.maximum(x,alpha*x)
  cache = x,alpha
  return out,cache


def leaky_backward(dout,cache):
  x,alpha = cache
  dx = np.where(x > 0, dout, alpha*dout)
  return dx

################################# End###################################

def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We keep each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p )/p   
    out = x * mask
    #when out *= mask  error..
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


# how to padding
# how to use index
def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  stride = conv_param['stride']
  pad    = conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride

  # Add padding around each 2D image
  # tricks####################################
  padded = np.lib.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  out = np.zeros((N,F,H_out,W_out))
  for i in xrange(N): # ith example
    for j in xrange(F): # jth filter

      # Convolve this filter over windows
      for k in xrange(H_out):
        hs = k * stride
        for l in xrange(W_out):
          ws = l * stride

          # Window we want to apply the respective jth filter over (C, HH, WW)
          #tricks
          window = padded[i, :, hs:hs+HH, ws:ws+WW]

          # Convolve
          out[i, j, k, l] = np.sum(window*w[j]) + b[j]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  x,w,b,conv_param = cache
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  stride = conv_param['stride']
  pad    = conv_param['pad']

  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  padded = np.lib.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  db = np.zeros_like(b)
  dw = np.zeros_like(w)
  dx = np.zeros_like(x)

  # care about packed
  padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

  for j in xrange(F):db[j]=np.sum(dout[:,j,:,:])

  #according to line 156
  for i in xrange(N):
      for j in xrange(F):
          for h in xrange(H_out):
              hs = h*stride
              for l in xrange(W_out):
                  ws = l*stride 
                  dw[j] += dout[i,j,h,l]*padded[i,:,hs:hs+HH,ws:ws+WW]
                  padded_dx[i,:,hs:hs+HH,ws:ws+WW] += dout[i,j,h,l]*w[j]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dx = padded_dx[:,:,pad:pad+H,pad:pad+W]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  N,C,H,W = x.shape
  pool_height = pool_param['pool_height']
  pool_width  = pool_param['pool_width']
  stride      = pool_param['stride']
  H_out = (H - pool_height)/stride + 1
  W_out = (W - pool_width)/stride + 1

  out = np.zeros((N,C,H_out,W_out))
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  for i in xrange(N):
      for j in xrange(C):
          for h in xrange(H_out):
              hs = h*stride
              for l in xrange(W_out):
                  ws = l*stride
                  out[i,j,h,l]=np.max(x[i,j,hs:hs+pool_height,ws:ws+pool_width])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x,pool_param = cache
  N,C,H,W = x.shape
  pool_height = pool_param['pool_height']
  pool_width  = pool_param['pool_width']
  stride      = pool_param['stride']
  H_out = (H - pool_height)/stride + 1
  W_out = (W - pool_width)/stride + 1
  dx = np.zeros_like(x)
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  for i in xrange(N):
      for j in xrange(C):
          for h in xrange(H_out):
              hs = h*stride
              for l in xrange(W_out):
                  ws = l*stride
                  window = x[i,j,hs:hs+pool_height,ws:ws+pool_width]
                  m = np.max(window)
                  # Gradient of max is indicator  ##tricks
                  dx[i,j,hs:hs+pool_height,ws:ws+pool_width]+=(window == m)*dout[i,j,h,l]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N

  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N

  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

