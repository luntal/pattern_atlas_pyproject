import numpy as np

def compute_rbf_weights(control_points, latent_vectors, sigma=0.7, lambd=1e-4):
    """
    Computes RBF weights with Ridge Regularization (lambda) to handle
    many/crowded control points without numerical instability.
    """
    control_points = np.asarray(control_points)
    N = len(control_points)

    # Distance matrix
    dists = np.linalg.norm(control_points[:, None, :] - control_points[None, :, :], axis=2)

    # Gaussian RBF
    Phi = np.exp(-(dists ** 2) / (2 * sigma ** 2))

    # Regularization: Add a tiny bit to the diagonal (Ridge Regression)
    # This keeps the matrix invertible even if points are very close.
    Phi += lambd * np.eye(N)

    # Solve Phi @ W = latent_vectors
    weights = np.linalg.solve(Phi, latent_vectors)
    return weights



def rbf_interpolate_exact(control_points, weights, cursor_pos, sigma=1.0):
    """
    Interpolates latent vector at cursor_pos using precomputed RBF weights.
    """
    dists = np.linalg.norm(control_points - cursor_pos, axis=1)
    rbf_vals = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    return rbf_vals @ weights