import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    util.plot(x_val, y_val, model.theta, 'output/p01e.png')

    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, model.predict(x_val))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        m, n = x.shape
        phi = 1. / m * np.sum(y)
        mu_0 = x[y == 0].mean(axis=0)
        mu_1 = x[y == 1].mean(axis=0)
        sigma = 1. / m * np.dot((x - mu_0).transpose(), x - mu_1)

        # Write theta in terms of the parameters
        sigma_inv =  np.linalg.inv(sigma)
        sigma_inv_mu_0 = sigma_inv.dot(mu_0)
        sigma_inv_mu_1 = sigma_inv.dot(mu_1)
        theta = sigma_inv_mu_1 - sigma_inv_mu_0
        theta_0 = 1. / 2 * (sigma_inv_mu_0.dot(mu_0) - sigma_inv_mu_1.dot(mu_1))
        theta_0 += -np.log(1. / phi - 1)
        theta_new = np.zeros(n + 1, dtype=theta.dtype)
        theta_new[0] = theta_0
        theta_new[1:] = theta
        self.theta = theta_new
        return theta_new
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        theta = self.theta
        return 1. / (1 + np.exp(-(x.dot(theta[1:]) + theta[0])))
        # *** END CODE HERE
