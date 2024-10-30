# Import public modules
import torch
import random
import numpy as np


def imgtrans(x):
    x = np.transpose(x, (1, 2, 0))
    return x


def set_random_seed(random_seed):
    """Set random seed(s) for reproducibility."""
    # Set random seeds for any modules that potentially use randomness
    random.seed(random_seed)
    np.random.seed(random_seed + 1)
    torch.random.manual_seed(random_seed + 2)


def expaddlog(a, b, eps=1e-15):
    """
    Perform multiplication of two tensors a and b using the exp-log trick:
    a*b = sign(a)*sign(b)*exp( log(|a|) + log(|b|) )

    Args:
        a (torch.tensor): First input tensor.
        b (torch.tensor): Second input tensor.
        eps (float): Tiny value used for numerical stability in the logarithm.
            (Default: 1e-15)

    Return:
        (torch.tensor): Result of a*b as torch tensor.

    Remark: The shapes of a and b must be such that they can be multiplied/added.

    """
    # Determine the signs of all entries of both a and b that should be tensors
    # of the same shape as a and b, respectively.
    sign_a = torch.sign(a)
    sign_b = torch.sign(b)

    # Determine the absolute values of all entries of both a and b that should
    # be tensors of the same shape as the a and b, respectively.
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)

    # Return a*b = sign(a)*sign(b)*exp( log(|a|) + log(|b|) ) where a tiny
    # epsilon is used within the logarithms for numerical stability.
    return sign_a * sign_b * torch.exp(torch.log(abs_a + eps) + torch.log(abs_b + eps))


def expsublog(a, b, eps=1e-15):
    """
    Perform division of two tensors a and b using the exp-log trick:
    a/b = sign(a)*sign(b)*exp( log(|a|) - log(|b|) )

    Args:
        a (torch.tensor): First input tensor.
        b (torch.tensor): Second input tensor.
        eps (float): Tiny value used for numerical stability in the logarithm.
            (Default: 1e-15)

    Return:
        (torch.tensor): Result of a/b as torch tensor.

    Remark: The shapes of a and b must be such that they can be divided/subtracted.

    """
    # Determine the signs of all entries of both a and b that should be tensors
    # of the same shape as a and b, respectively.
    sign_a = torch.sign(a)
    sign_b = torch.sign(b)

    # Determine the absolute values of all entries of both a and b that should
    # be tensors of the same shape as the a and b, respectively.
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)

    # Return a*b = sign(a)*sign(b)*exp( log(|a|) + log(|b|) ) where a tiny
    # epsilon is used within the logarithms for numerical stability.
    return sign_a * sign_b * torch.exp(torch.log(abs_a + eps) - torch.log(abs_b + eps))
