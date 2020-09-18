import argparse
import numpy as np
import matplotlib.pyplot as plt

import p01b_logreg
import p01e_gda
import util


def plot_all(x, y, theta1, theta2, save_path, correction=1.0):
    """Plot dataset and fitted GLM parameters.

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
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta1[0] / theta1[2] * correction + theta1[1] / theta1[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)

    x2 = -(theta2[0] / theta2[2] * correction + theta2[1] / theta2[2] * x1)
    plt.plot(x1, x2, c='blue', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.show()
    plt.savefig(save_path)


def main(train_path, eval_path, fig_path):
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    lr = p01b_logreg.LogisticRegression()
    lr.fit(x_train, y_train)

    # Train a GDA classifier
    gda = p01e_gda.GDA()
    gda.fit(x_train[:, 1:], y_train)

    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    plot_all(x_val, y_val, lr.theta, gda.theta, fig_path)


def main_h(train_path, eval_path, fig_path):
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # it's okay without / (1. + median)
    median = np.median(x_train, axis=0)
    x_train = np.log((1. + x_train) / (1. + median))
    x_train = util.add_intercept(x_train)

    # Train a logistic regression classifier
    lr = p01b_logreg.LogisticRegression()
    lr.fit(x_train, y_train)

    # Train a GDA classifier
    gda = p01e_gda.GDA()
    gda.fit(x_train[:, 1:], y_train)

    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    x_val = np.log((1. + x_val) / (1. + median))
    x_val = util.add_intercept(x_val)
    plot_all(x_val, y_val, lr.theta, gda.theta, fig_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('quest', nargs='?', type=str, default='f',
                    help='choose f: dataset1 or g: dataset2 or h: transform ds1')
    args = parser.parse_args()

    # parameters initialization
    if args.quest == 'f':
        train_path='../data/ds1_train.csv'
        eval_path='../data/ds1_valid.csv'
        fig_path = 'output/p01f.png'
        main(train_path, eval_path, fig_path)

    if args.quest == 'g':
        train_path='../data/ds2_train.csv'
        eval_path='../data/ds2_valid.csv'
        fig_path = 'output/p01g.png'
        main(train_path, eval_path, fig_path)
    
    if args.quest == 'h':
        train_path='../data/ds1_train.csv'
        eval_path='../data/ds1_valid.csv'
        fig_path = 'output/p01h.png'
        main_h(train_path, eval_path, fig_path)
