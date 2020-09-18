import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    print('p5b mse: ', ((y_val - y_pred) ** 2).mean(axis=0))

    # Plot validation predictions on top of training set
    plt.figure()

    # No need to save predictions
    # Plot data
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_val, y_pred, 'ro', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, _ = x.shape
        x_cache = self.x
        y_cache = self.y
        tau = self.tau
        x_cache_t = x_cache.transpose()
        y_pred = np.zeros(m)
        for i in range(m):
            aux = x_cache - x[i]
            aux = np.linalg.norm(aux, ord=2, axis=1)
            aux = np.exp(- aux / (2 * (tau ** 2)))
            w_mat = np.diag(aux)
            thelta = np.linalg.inv(x_cache_t.dot(w_mat).dot(x_cache)).dot(x_cache_t).dot(w_mat).dot(y_cache)
            y_pred[i] = thelta.dot(x[i])
        return y_pred
        # *** END CODE HERE ***
