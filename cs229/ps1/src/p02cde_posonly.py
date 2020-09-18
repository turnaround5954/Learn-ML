import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    lr_c = LogisticRegression()
    lr_c.fit(x_train, t_train)

    x_val, t_val = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    # print(np.mean(t_val == (lr_c.predict(x_val) > 0.5)))
    y_pred_c = lr_c.predict(x_val)
    np.savetxt(pred_path_c, y_pred_c)

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    _, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    lr_d = LogisticRegression()
    lr_d.fit(x_train, y_train)

    _, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    # print(np.mean(t_val == (lr_c.predict(x_val) > 0.5)))
    y_pred_d = lr_d.predict(x_val)
    np.savetxt(pred_path_d, y_pred_d)
    
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    alpha = y_pred_d[y_val == 1].mean()
    # p(t=1| x) = p(y=1| x) / alpha
    y_pred_e = y_pred_d / alpha
    np.savetxt(pred_path_e, y_pred_e)

    # plot results of c, d
    util.plot(x_val, t_val, lr_c.theta, 'output/p02c.png')
    util.plot(x_val, t_val, lr_d.theta, 'output/p02d.png')
    
    # calculate correction:
    # a1 * x1 + a2 * x2 + a0 = beta
    # x2 + a1 / a2 * x1 + a0 / a2 = beta / a2
    # x2 = beta / a2 - (a1 / a2 * x1 + a0 / a2)
    # x2 = - ((a0 - beta) / a2 + a1 / a2)
    # correction = (a0 - beta) / a0

    # beta = theta.dot(x)
    # 0.5 * alpha = 1. / (1. + np.exp(beta)) 
    # beta = np.log(2. / alpha - 1.)
    # correction = 1. - np.log(2. / alpha - 1.)
    correction = 1. - np.log(2. / alpha - 1.)
    
    # plot result of e
    util.plot(x_val, t_val, lr_c.theta, 'output/p02e.png', correction=correction)
    # *** END CODER HERE
