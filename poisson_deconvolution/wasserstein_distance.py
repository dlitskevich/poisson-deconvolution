import numpy as np
import scipy.optimize as optim


def getRowSumMatrix(M: int, N: int) -> np.ndarray:
    """
    Returns a matrix with row sum constraints for the flattened problem.

    Parameters:
        M (int): Number of rows.
        N (int): Number of columns.

    Returns:
        np.ndarray: Row sum matrix.
    """
    result = np.zeros((M, M * N), dtype=int)
    for i in range(M):
        result[i, i * N : (i + 1) * N] = 1
    return result


def getColSumMatrix(M: int, N: int) -> np.ndarray:
    """
    Returns a matrix with column sum constraints for the flattened problem.

    Parameters:
        M (int): Number of rows.
        N (int): Number of columns.

    Returns:
        np.ndarray: Column sum matrix.
    """
    result = np.zeros((N, M * N), dtype=int)
    for i in range(N):
        result[i, i::N] = 1
    return result


def getConstraintMatrix(M: int, N: int) -> np.ndarray:
    """
    Returns the constraint matrix for the flattened problem.

    Parameters:
        M (int): Number of rows.
        N (int): Number of columns.

    Returns:
        np.ndarray: Constraint matrix.
    """
    return np.vstack((getRowSumMatrix(M, N), getColSumMatrix(M, N)))


def costOT(mu: np.ndarray, nu: np.ndarray, cost: np.ndarray) -> float:
    """
    Computes the optimal transport cost using linear programming.

    Parameters:
        mu (np.ndarray): Source distribution.
        nu (np.ndarray): Target distribution.
        cost (np.ndarray): Cost matrix.

    Returns:
        float: Optimal transport cost.
    """
    M = mu.size
    N = nu.size
    A = getConstraintMatrix(M, N)
    b = np.concatenate((mu, nu))
    result = optim.linprog(
        cost.ravel(),
        A_ub=None,
        b_ub=None,
        A_eq=A,
        b_eq=b,
        bounds=[(0, None) for i in range(M * N)],
        method="highs-ds",
        callback=None,
        options=None,
        x0=None,
    )

    return cost.ravel().dot(result["x"])


def wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the 2-Wasserstein distance between two distributions.

    Parameters:
        x (np.ndarray): Source distribution.
        y (np.ndarray): Target distribution.

    Returns:
        float: 2-Wasserstein distance.
    """
    cost_matrix = np.array([[np.sum(np.abs(a - b) ** 2) for a in y] for b in x])
    M = x.shape[0]
    N = y.shape[0]
    mu = np.ones(M) / M
    nu = np.ones(N) / N

    return costOT(mu, nu, cost_matrix) ** (1 / 2)


def assignment_ot_cost(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the optimal transport cost using the Hungarian algorithm.

    Parameters:
        x (np.ndarray): Source distribution.
        y (np.ndarray): Target distribution.

    Returns:
        float: Optimal transport cost.
    """
    cost_matrix = np.array([[np.sum(np.abs(a - b) ** 2) for a in y] for b in x])
    rows_ids, col_ids = optim.linear_sum_assignment(cost_matrix)

    return cost_matrix[rows_ids, col_ids].sum() ** (1 / 2) / x.size
