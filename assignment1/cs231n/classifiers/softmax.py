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
  scores = X.dot(W)
  logc = np.max(scores, axis=1)
  logc = np.reshape(np.repeat(logc, num_classes), scores.shape)
  expscores = np.exp(scores+logc)
  for i in range(num_train):
    expmom = sum(expscores[i])
    expson = expscores[i, y[i]]
    loss += -np.log(expson / expmom)
    for j in range(num_classes):
      if j != y[i]:
        dW[:, j] += (expscores[i][j] / expmom) * X[i]
      else :
        dW[:, y[i]] += ((expscores[i][y[i]] / expmom) - 1) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
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
  scores = X.dot(W)
  logc = np.max(scores, axis=1)
  logc = np.reshape(np.repeat(logc, num_classes), scores.shape)
  scores += logc
  expscores = np.exp(scores)
  expmom = np.sum(expscores, axis=1)
  expson = expscores[np.arange(num_train), y]
  loss += np.sum(-(np.log(expson / expmom)))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  expmom = np.reshape(np.repeat(expmom, num_classes), expscores.shape)
  M = expscores / expmom
  M[np.arange(num_train), y] -= 1
  # !!! M[:, y] -= 1 wrong why ?
  dW = X.T.dot(M)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

