import numpy as np
import matplotlib.pyplot as plt

# Helpers
relu = lambda x: x * (x > 0)
relu_grad = lambda x: (x > 0)
def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  z1 = X.dot(W1) + b1     # (N,H)
  a1 = z1
  a1[a1<0] = 0
  #a1 = relu(z1)
  z2 = z1.dot(W2) + b2    # (N,C)
  scores = z2
  ####do not use np.exp()[not softmax]
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################
  temp = np.exp(scores)
  correct_score = scores[range(N),y]
  #################################is error########
  # ----log(e1+e2+e3+e4) != log(e1+e2)+log(e3+e4)---------------key------------
  # loss = np.log(np.sum(temp)) - np.sum(correct_score)
  #################################################
  # loss = np.sum(np.log(np.sum(temp,axis=1)) - correct_score)
  loss = np.sum(np.log(np.sum(temp,axis=1))) - np.sum(correct_score)
  loss /= N
  loss += 0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  # compute the gradients
  grads = {}
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
  #two fault:1 backprog use model['W'] not grads['W']
  #          2 relu_grad(z1) not ..
  #          3 don't forget divide by N and regularization
  temp = temp / np.sum(temp,axis=1).reshape(N,-1)
  temp[range(N),y]-=1          
  grads['W2'] = a1.T.dot(temp)
  grads['b2'] = np.sum(temp,axis=0)

  grads['W1'] = X.T.dot((temp.dot(model['W2'].T))*(relu_grad(z1)) )

  grads['b1'] = np.sum((temp.dot(model['W2'].T))*(relu_grad(z1)),axis=0) 
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  grads['W2'] /= N
  grads['W2'] += reg*model['W2']
  grads['b2'] /= N

  grads['W1'] /= N
  grads['W1'] += reg*model['W1']
  grads['b1'] /= N
  return loss, grads

