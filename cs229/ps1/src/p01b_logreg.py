import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_val, y_val, model.theta, 'output/p01b.png')

    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, model.predict(x_val))
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

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
            hypo = 1. / (1 + np.exp(-x.dot(self.theta)))
            delta = (1. / m) * (hypo - y).dot(x)
            hessian = (1. / m) *  x.transpose().dot(np.diag(hypo * (1. - hypo))).dot(x)
            updates = - np.linalg.inv(hessian).dot(delta)
            if np.linalg.norm(updates, ord=1) < self.eps or iterations >= self.max_iter:
                self.theta += updates
                return
            self.theta += updates
            iterations += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1. / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
