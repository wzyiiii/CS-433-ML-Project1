import numpy as np


def _sigmoid(t):
    """Compute sigmoid. Numerically stable sigmoid."""
    t = np.asarray(t)
    out = np.empty_like(t, dtype=float)

    positive = t >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-t[positive]))

    exp_t = np.exp(t[~positive])
    out[~positive] = exp_t / (1.0 + exp_t)
    return out


def _mean_squared_error(y, tx, w):
    """Compute MSE. MSE with the 0.5 factor used in the course."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def _logistic_loss(y, tx, w):
    """Compute logistic loss. Average binary cross-entropy / logistic loss."""
    scores = tx @ w
    # log(1 + exp(scores)) - y * scores, evaluated stably.
    loss = np.logaddexp(0.0, scores) - y * scores
    return np.mean(loss)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using batch gradient descent."""
    y = np.asarray(y).reshape(-1)
    tx = np.asarray(tx)
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    n = y.shape[0]

    for _ in range(max_iters):
        e = y - tx @ w
        gradient = -(tx.T @ e) / n
        w -= gamma * gradient

    loss = _mean_squared_error(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    A random training example is sampled at every iteration.
    """
    y = np.asarray(y).reshape(-1)
    tx = np.asarray(tx)
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    n = y.shape[0]

    for _ in range(max_iters):
        i = np.random.randint(n)
        xi = tx[i]
        yi = y[i]

        error = yi - xi @ w
        gradient = -xi * error
        w -= gamma * gradient

    loss = _mean_squared_error(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least-squares regression using the normal equations."""
    y = np.asarray(y).reshape(-1)
    tx = np.asarray(tx)

    a = tx.T @ tx
    b = tx.T @ y

    # Solve the normal equations without np.linalg.lstsq.
    # pinv also handles singular / rank-deficient feature matrices.
    w = np.linalg.pinv(a) @ b

    loss = _mean_squared_error(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    The regularization convention follows the project:
        (X^T X + 2 N lambda I) w = X^T y

    The returned loss is the unregularized MSE, as required by the
    project specification.
    """
    y = np.asarray(y).reshape(-1)
    tx = np.asarray(tx)

    n, d = tx.shape
    a = tx.T @ tx + 2.0 * n * lambda_ * np.eye(d)
    b = tx.T @ y

    w = np.linalg.solve(a, b)

    loss = _mean_squared_error(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent for y in {0, 1}."""
    y = np.asarray(y).reshape(-1)
    tx = np.asarray(tx)
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    n = y.shape[0]

    for _ in range(max_iters):
        scores = tx @ w
        probabilities = _sigmoid(scores)

        gradient = (tx.T @ (probabilities - y)) / n
        w -= gamma * gradient

    loss = _logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    The optimization objective contains lambda_ * ||w||^2.
    The loss returned here intentionally excludes the regularization
    penalty, as required by the project specification.
    """
    y = np.asarray(y).reshape(-1)
    tx = np.asarray(tx)
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    n = y.shape[0]

    for _ in range(max_iters):
        scores = tx @ w
        probabilities = _sigmoid(scores)

        gradient = (tx.T @ (probabilities - y)) / n
        gradient += 2.0 * lambda_ * w

        w -= gamma * gradient

    loss = _logistic_loss(y, tx, w)
    return w, loss
