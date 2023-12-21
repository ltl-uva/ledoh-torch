import math
import torch
from torch import Tensor

def forward_sliced_dispersion(X: Tensor, p: Tensor, q: Tensor) -> Tensor:
    """
    calculates forward pass for sliced dispersion
    :param X: Tensor of shape (N, d)
    :param p, q: vectors defining great circle. p is orthogonal to q

    :return: squared distance between projected and dispered angles
    """
    N = X.size(0)
    device = X.device

    Xp = X @ p
    Xq = X @ q

    thetas = torch.arctan2(Xq, Xp)

    ix = torch.argsort(thetas)
    invix = torch.empty_like(ix)
    invix[ix] = torch.arange(N, device=device)

    phis = -math.pi - math.pi / N + (2 * math.pi *
                                              torch.arange(1, N + 1,
                                                           device=device)
                                              ) / N


    phis = phis[invix]
    thetas_star = torch.mean(thetas)+phis

    return 0.5*torch.mean(torch.pow(thetas-thetas_star, 2))


def sliced_dispersion_gradient(Xp: Tensor,
                               Xq: Tensor,
                               p: Tensor, q: Tensor,
                               theta_minus_thetastr: Tensor,
                               grad_output: Tensor) -> Tensor:
    """
    :param Xp: X @ p
    :param Xq: X @ q
    :param p,q: vectors defining great circle. p is orthogonal to q 
    :param theta_minus_thetastr: define the subtraction between projected angles theta and optimal dispersed theta_star
    :param grad_output: output of the forward function
    :return: gradient of the sliced dispersion
    """

    grad = (
            (Xp * q.unsqueeze(-1) - Xq * p.unsqueeze(-1)) / (Xp ** 2 + Xq ** 2)
    ) * theta_minus_thetastr

    return grad_output * grad.T