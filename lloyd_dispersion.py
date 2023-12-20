import torch.nn.functional as F
import torch

def lloyd_dispersion(X, n_samples=100):
    d = X.size(-1)
    device = X.device
    samples = F.normalize(torch.randn(n_samples,
                                          1,
                                          d,
                                          device=device
                                          ),
                              dim=-1)
    #compute the distance to the nearest point on the sphere
    dist2 = torch.acos(samples @ X.T) ** 2
    # choose the closest center for each sample and sum results
    return (torch.min(dist2, dim=-1)[0]).mean()