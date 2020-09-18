import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    best_lwr = None
    best_tau = None
    best_mse = 0.0
    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau=tau)
        lwr.fit(x_train, y_train)
        y_pred = lwr.predict(x_val)
        mse = ((y_val - y_pred) ** 2).mean(axis=0)
        # plot
        plt.figure()
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_val, y_pred, 'ro', linewidth=2)
        plt.title('mse: ' + str(mse)[:5])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('output/p05b_' + str(tau) + '.png')
        # choose best model
        if mse < best_mse or best_lwr is None:
            best_lwr = lwr
            best_tau = tau
            best_mse = mse
    print('p05c best tau:', best_tau)
    print('p05c best mse:', best_mse)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = best_lwr.predict(x_test)
    mse = ((y_test - y_pred) ** 2).mean(axis=0)
    print('p05c test mse: ', mse)

    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)

    # Plot data
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_test, y_pred, 'ro', linewidth=2)
    plt.title('test mse: ' + str(mse)[:5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b_test.png')
    # *** END CODE HERE ***
