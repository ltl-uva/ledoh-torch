import ot
from ledoh_torch import SphereDispersion


class SphericalSlicedWassersteinDispersion(SphereDispersion):
    def __init__(self, n_projections=100):
        super().__init__()
        self.n_projections = n_projections


    def forward(self, X):
        return ot.sliced_wasserstein_sphere_unif(X, n_projections=self.n_projections)
