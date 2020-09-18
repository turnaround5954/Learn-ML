import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    m = x.shape[0]
    random_indices = np.arange(m)
    np.random.shuffle(random_indices)
    k_groups = np.array_split(random_indices, K)
    mu = [np.mean(x[group, :], axis=0) for group in k_groups]
    sigma = [np.cov(x[group, :], rowvar=False) for group in k_groups]

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K) / K

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((m, K)) / K

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        m, n = x.shape

        if prev_ll is None:
            update_w_without_norm(x, w, phi, mu, sigma)
            log_mat = - n / 2.0 * np.log(2 * np.pi) + np.log(w)
            # prev_ll = np.sum(w_old * (log_mat - np.log(w_old)))
            prev_ll = np.sum(1. / K * (log_mat - np.log(1. / K)))
        else:
            prev_ll = ll
        
        w_sum = 1. / w.sum(axis=1)
        w = np.einsum('k,kj->kj', w_sum, w)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        w_sum_m = w.sum(axis=0)
        phi = w_sum_m / m
        mu = w.transpose().dot(x)
        mu = np.einsum('k,ki->ki', w_sum_m, mu)
        mu = list(mu)
        b = [(x - mu[j]) for j in range(K)]
        sigma = [b[j].transpose().dot(np.diag(w[:, j])).dot(b[j]) / w_sum_m[j] for j in range(K)]

        # (3) Compute the log-likelihood of the data to check for convergence.
        it += 1
        prev_w = np.copy(w)
        update_w_without_norm(x, w, phi, mu, sigma)

        log_mat = - x.shape[1] / 2.0 * np.log(2 * np.pi) + np.log(w)
        ll = np.sum(prev_w * (log_mat - np.log(prev_w)))
        
        if it % 10 == 0:
            print('log-likelihood: ', ll)

        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        m, n = x.shape
        m_tilde, n_tilde = x_tilde.shape
        z_idx = z.astype(np.int32)

        if prev_ll is None:
            # calculate unsupervised part
            update_w_without_norm(x, w, phi, mu, sigma)
            log_mat = - n / 2.0 * np.log(2 * np.pi) + np.log(w)
            prev_ll = np.sum(1. / K * (log_mat - np.log(1. / K)))
            
            # calulate supervised part
            w_sup = np.zeros((m_tilde, K))
            update_w_without_norm(x_tilde, w_sup, phi, mu, sigma)
            log_mat = - n_tilde / 2.0 * np.log(2 * np.pi) + np.log(w_sup)
            prev_ll += alpha * np.sum(log_mat[(np.arange(m_tilde), z_idx)])
        else:
            prev_ll = ll

        # update w
        w_sum = 1. / w.sum(axis=1)
        w = np.einsum('k,kj->kj', w_sum, w)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        # z_tag, z_count = np.unique(z, return_counts=True)
        # phi = (w.sum(axis=0) + z_count[z_tag[np.arange(K)]]) / (m + m_tilde)
        z_mat = np.zeros((m_tilde, K))
        z_mat[(np.arange(m_tilde), z_idx)] = 1.0
        z_count = z_mat.sum(axis=0)
        w_sum_m = w.sum(axis=0)
        w_sum_m = w_sum_m + z_count
        phi = w_sum_m / (m + m_tilde)

        mu = w.transpose().dot(x) + z_mat.transpose().dot(x_tilde)
        mu = np.einsum('k,ki->ki', w_sum_m, mu)
        mu = list(mu)
    
        b = [(x - mu[j]) for j in range(K)]
        b_t = [(x_tilde - mu[j]) for j in range(K)]
        sigma = [(b[j].T.dot(np.diag(w[:, j])).dot(b[j]) + b_t[j].T.dot(np.diag(z_mat[:, j])).dot(b_t[j])) / w_sum_m[j] for j in range(K)]

        # (3) Compute the log-likelihood of the data to check for convergence.
        it += 1
        prev_w = np.copy(w)

        # calculate unsupervised part
        update_w_without_norm(x, w, phi, mu, sigma)

        log_mat = - x.shape[1] / 2.0 * np.log(2 * np.pi) + np.log(w)
        ll = np.sum(prev_w * (log_mat - np.log(prev_w)))
        
        # calulate supervised part
        w_sup = np.zeros((m_tilde, K))
        update_w_without_norm(x_tilde, w_sup, phi, mu, sigma)
        log_mat = - n_tilde / 2.0 * np.log(2 * np.pi) + np.log(w_sup)
        ll += alpha * np.sum(log_mat[(np.arange(m_tilde), z_idx)])

        if it % 10 == 0:
            print('log-likelihood: ', ll)

        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
def update_w_without_norm(x, w, phi, mu, sigma):
    for j in range(K):
        phi_j = phi[j]
        mu_j = mu[j]
        sqrt_det_sigma_j = np.sqrt(np.linalg.det(sigma[j]))
        inv_sigma_j = np.linalg.inv(sigma[j])
        b_j = x - mu_j
        exp_term_j = - 0.5 * np.einsum('ki,ki->k', b_j.dot(inv_sigma_j), b_j)
        w[:, j] = 1./ sqrt_det_sigma_j * np.exp(exp_term_j) * phi_j

# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
