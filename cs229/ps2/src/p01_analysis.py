"""
Analysis for p01
"""

import matplotlib.pyplot as plt
import numpy as np
import util


def calc_svm_grad(X, Y, theta, reg):
    """Compute the gradient of the loss with respect to theta."""
    m, _ = X.shape

    margins = np.maximum(0, 1 - Y * X.dot(theta))
    loss = margins.mean() + reg * theta[1:].dot(theta[1:])

    margins[margins > 0] = 1
    # don't do regularization on bias
    theta_mask = np.ones_like(theta)
    theta_mask[0] = 0
    grad = -(1. / m) * np.dot(margins * Y, X) + reg * np.multiply(theta_mask, theta)

    return loss, grad


def svm_hinge_loss(X, Y, learning_rate=10, reg=0.):
    """Train svm using sgd."""
    _, n = X.shape
    theta = np.zeros(n)

    i = 0
    while True:
        i += 1
        prev_theta = theta
        loss, grad = calc_svm_grad(X, Y, theta, reg)

        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print('theta:', theta, 'grad:', grad, 'loss:', loss)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break

        if i > 500000:
            break
    return theta


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, _ = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y, learning_rate=10, scale_lr=False, using_reg=False, iter_limit=100000):
    """Train a logistic regression model."""
    _, n = X.shape
    theta = np.zeros(n)

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        if scale_lr is True:
            learning_rate *= 1. / i
        if using_reg is True:
            theta_mask = np.ones_like(theta)
            theta_mask[0] = 0
            grad += 0.001 * np.multiply(theta_mask, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
        if i >= iter_limit:
            print('Stoped at %d iteration' % i)
            break
    return theta


def plot(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)


if __name__ == "__main__":
    # dataset a is not linearly separable, while dataset b is.

    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)

    # print("===========training on A===========")
    # theta_a = logistic_regression(Xa, Ya)

    # print("===========training on B===========")
    # theta_b_n0 = logistic_regression(Xb, Yb, iter_limit=100000)
    # theta_b_n1 = logistic_regression(Xb, Yb, iter_limit=100001)
    # print("factor of theta:", theta_b_n1 / theta_b_n0)
    # # this shows that (approximately) only theta's magnitude is changing

    # plot(Xa, Ya, theta_a, 'output/ds1_a.png')
    # plot(Xb, Yb, theta_b_n0, 'output/ds1_b_iter0.png')
    # plot(Xb, Yb, theta_b_n1, 'output/ds1_b_iter1.png')

    # print("===========training on B with smaller lr===========")
    # theta_b_small = logistic_regression(Xb, Yb, learning_rate=0.001, iter_limit=1000000)
    # not working

    # print("===========training on B with decreasing lr===========")
    # theta_b_decreasing = logistic_regression(
    #     Xb, Yb, scale_lr=True, iter_limit=1000000)
    # plot(Xb, Yb, theta_b_decreasing, 'output/ds1_b_lr_dec.png')
    # can terminate quickly

    # print("===========training on B with linear scaling===========")
    # Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)
    # # Xb = Xb * 0.1
    # Xb = Xb * 10
    # Xb = util.add_intercept_fn(Xb)
    # theta_b_scale_inputs = logistic_regression(Xb, Yb, iter_limit=1000000)
    # not working

    # print("===========training on B with regularization===========")
    theta_b_reg = logistic_regression(Xb, Yb, using_reg=True, iter_limit=1000000)
    plot(Xb, Yb, theta_b_reg, 'output/ds1_b_reg.png')
    # works well

    # print("===========training on B with Gaussian noise===========")
    # Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)
    # # note that the scale is important
    # Xb += np.random.normal(scale=0.1, size=Xb.shape)
    # Xb = util.add_intercept_fn(Xb)
    # theta_b_noise = logistic_regression(Xb, Yb, iter_limit=1000000)
    # plot(Xb, Yb, theta_b_noise, 'output/ds1_b_noise.png')
    # generally works

    # print("===========training on B with svm===========")
    # note: need to tune regularization extremely small to make it work
    # theta_a_svm = svm_hinge_loss(Xa, Ya)
    # plot(Xb, Yb, theta_a_svm, 'output/ds1_a_svm.png')

    # theta_b_svm = svm_hinge_loss(Xb, Yb)
    # plot(Xb, Yb, theta_b_svm, 'output/ds1_b_svm.png')
    # svm can solve this problem
