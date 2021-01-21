import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  X -= np.reshape(np.max(X, axis=1), (num_train, 1)) # regularization
  for i in xrange(num_train):
    score = X[i].dot(W)
    loss += -score[y[i]] + np.log(np.sum(np.exp(score)))
    for j in xrange(num_classes):
      dW[:, j] += 1 / np.sum(np.exp(score)) * np.exp(score[j]) * X[i]
    dW[:, y[i]] += -X[i]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  X -= np.reshape(np.max(X, axis=1), (num_train, 1)) # regularization
  scores = X.dot(W) # (N, C)
  iterator = xrange(num_train)
  sum_exp = np.sum(np.exp(scores), axis=1) # (N dim array)
  loss = -np.sum(scores[iterator, y[iterator]]) + np.sum(np.log(sum_exp))
  trans_matirx = np.zeros_like(scores)
  trans_matirx[iterator, y[iterator]] = 1
  dW = -np.dot(X.T, trans_matirx) + np.dot(X.T / sum_exp, np.exp(scores))
  
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

