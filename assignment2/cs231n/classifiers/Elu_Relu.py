import numpy as np

from cs231n.layers import *

def relu(X,model,y=None,reg=0.0):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	a1,cache1 = affine_forward(X,W1,b1)
	a2,cache2 = relu_forward(a1)
	scores,cache3 = affine_forward(a2,W2,b2)

	if y is None:
		return scores
	# softmax   svm
	data_loss,dscores=softmax_loss(scores,y)
	da2,dW2,db2 = affine_backward(dscores,cache3)
	da1 = relu_backward(da2,cache2)
	dX,dW1,db1 = affine_backward(da1,cache1)

	dW1 += reg*W1
	dW2 += reg*W2

	reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  	loss = data_loss + reg_loss
  	grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

  	return loss,grads
def leaky(X,model,y=None,reg=0.0):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	a1,cache1 = affine_forward(X,W1,b1)
	a2,cache2 = leaky_forward(a1)
	scores,cache3 = affine_forward(a2,W2,b2)

	if y is None:
		return scores
	# softmax   svm
	data_loss,dscores=softmax_loss(scores,y)
	da2,dW2,db2 = affine_backward(dscores,cache3)
	da1 = leaky_backward(da2,cache2)
	dX,dW1,db1 = affine_backward(da1,cache1)

	dW1 += reg*W1
	dW2 += reg*W2

	reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  	loss = data_loss + reg_loss
  	grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

  	return loss,grads

def elu(X,model,y=None,reg=0.0):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	a1,cache1 = affine_forward(X,W1,b1)
	a2,cache2 = elu_forward(a1)
	scores,cache3 = affine_forward(a2,W2,b2)

	if y is None:
		return scores
	# softmax   svm
	data_loss,dscores=softmax_loss(scores,y)
	da2,dW2,db2 = affine_backward(dscores,cache3)
	da1 = elu_backward(da2,cache2)
	dX,dW1,db1 = affine_backward(da1,cache1)

	dW1 += reg*W1
	dW2 += reg*W2

	reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  	loss = data_loss + reg_loss
  	grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

  	return loss,grads

def init_elu_relu_model(input_size, hidden_size, output_size):
  # initialize a model * 0.00001          np.sqrt(2.0/input_size)
  model = {}
  model['W1'] = 0.00001* np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model
