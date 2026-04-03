import torch
import torch.nn.functional as F
import math

def slerp(v0, v1, t):
    """
    Performs spherical interpolation (slerp) between two vectors v0 and v1.

    v0, v1: tensors of shape (1, D)
    t: float between 0 and 1
    """
    # Normalize to avoid numerical issues
    v0_norm = F.normalize(v0, dim=-1)
    v1_norm = F.normalize(v1, dim=-1)

    # Dot product between normalized vectors
    dot = (v0_norm * v1_norm).sum(dim=-1).clamp(-1.0, 1.0)  # keep safe range

    # Angle between them
    theta = torch.acos(dot)  # [1]

    # If the angle is very small, fall back to linear interpolation
    if torch.all(theta < 1e-6):
        return (1 - t) * v0 + t * v1

    # Compute interpolation weights
    sin_theta = torch.sin(theta)
    w1 = torch.sin((1 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta

    # Final interpolation
    return w1.unsqueeze(-1) * v0 + w2.unsqueeze(-1) * v1


def interpolate_latent_vectors_spherical(latent_vectors, steps_between=5):
    """
    Spherical interpolation between a list of latent vectors (SLERP).

    Args:
        latent_vectors: List of tensors with shape (1, D)
        steps_between: int, number of steps between each pair

    Returns:
        List of tensors including original + interpolated vectors
    """
    result = []
    result.append(latent_vectors[0])

    for i in range(len(latent_vectors) - 1):
        start_vector = latent_vectors[i]
        end_vector = latent_vectors[i + 1]

        # SLERP between start and end
        for step in range(1, steps_between + 1):
            t = step / (steps_between + 1)
            interp = slerp(start_vector, end_vector, t)
            result.append(interp)

        # Add end except for last pair
        if i < len(latent_vectors) - 2:
            result.append(end_vector)

    result.append(latent_vectors[-1])

    return result


# linear interpolation
def interpolate_latent_vectors(latent_vectors, steps_between=5):
    """
    Interpolate between a list of latent vectors.

    Args:
        latent_vectors: List of tensors with shape (1, 256)
        steps_between: Number of interpolation steps between each pair of vectors

    Returns:
        List of tensors containing the original vectors and the interpolated ones
    """
    result = []

    # Add the first vector to the result list
    result.append(latent_vectors[0])

    # Interpolate between consecutive pairs of vectors
    for i in range(len(latent_vectors) - 1):
        start_vector = latent_vectors[i]
        end_vector = latent_vectors[i + 1]

        # Linear interpolation
        for step in range(1, steps_between + 1):
            # Calculate interpolation ratio
            ratio = step / (steps_between + 1)

            # Create interpolated vector
            interpolated = start_vector * (1 - ratio) + end_vector * ratio
            result.append(interpolated)

        # Add the end vector (except for the very last one which we'll add outside the loop)
        if i < len(latent_vectors) - 2:
            result.append(end_vector)

    # Add the final vector
    result.append(latent_vectors[-1])

    return result