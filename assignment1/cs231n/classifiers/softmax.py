import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  D,N = X.shape
  C = W.shape[0]
  for i in xrange(N):
    scores = W.dot(X[:,i])  # C * 1
    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    log_c = np.max(scores)
    scores -= log_c
    cs = scores[y[i]]
    sum_i = np.sum(np.exp(scores))
    loss += np.log(sum_i) - (cs)
    
    for j in xrange(C):
        p = np.exp(scores[j])/sum_i
        dW[j,:] += (p -(j==y[i]))*X[:,i]   
  loss /= N
  dW   /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW   += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  D,N = X.shape
  C = W.shape[0]
  scores = W.dot(X)
  scores -= np.max(scores,axis=0)
  #care for y's shape  1-dimensional array
  cs = scores[y,range(N)]

  loss = np.log(np.sum(np.exp(scores))) - np.sum(cs)
  #loss = np.sum(-cs + np.log(np.sum(np.exp(scores),axis=0)))

  p = np.exp(scores)/np.sum(np.exp(scores), axis=0)
  p[y,range(N)] -= 1
  dW = p.dot(X.T) 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= N
  dW   /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW   += reg * W
  return loss, dW


# W = np.array(range(24)).reshape((4,6))
# X = np.array(range(42)).reshape((6,7))
# y = np.array([0,1,0,1,1,0,0])

# print softmax_loss_naive(W,X,y,0)
# print softmax_loss_vectorized(W,X,y,0)


