import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    x_train_ex = util.add_intercept(x_train)
    model = PoissonRegression(step_size=lr, max_iter=1000)
    model.fit(x_train_ex, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_val, _ = util.load_dataset(eval_path, add_intercept=True)
    np.savetxt(pred_path, model.predict(x_val))
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        iterations = 0
        while True:
            neta = x.dot(self.theta)
            canonical_response = np.exp(neta)
            delta = 1. / m * (y - canonical_response).dot(x)
            updates = self.step_size * delta
            if np.linalg.norm(updates, ord=1) < self.eps or iterations >= self.max_iter:
                # print(iterations)
                self.theta += updates
                return
            self.theta += updates
            iterations += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        neta = x.dot(self.theta)
        canonical_response = np.exp(neta)
        return canonical_response
        # *** END CODE HERE ***
