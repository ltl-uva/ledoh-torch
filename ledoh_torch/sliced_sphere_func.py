import torch
from .sliced_dispersion import SlicedSphereDispersion


class SlicedSphereDispersionFunction(torch.autograd.Function):
    """
    calculates forward and backward pass for sliced sphere despersion regularized
    """
    @staticmethod
    def forward(ctx, X, p, q, reduction, return_hidden_states):
        dist, hs = SlicedSphereDispersion.forward(X, p, q, reduction=reduction, return_hidden_states=return_hidden_states)

        ctx.save_for_backward(hs["xp"], hs["xq"], p, q, hs["theta_diff"])

        return dist

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        Xp, Xq, p, q, theta_minus_thetastr = ctx.saved_tensors

        grad = SlicedSphereDispersion.backward(Xp, Xq, p, q, theta_minus_thetastr, grad_output)

        return grad, None, None, None, None

compute_sliced_sphere_dispersion = SlicedSphereDispersionFunction.apply