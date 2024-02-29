import torch
import torch.nn.functional as F

from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import SphereExact, Sphere
from geoopt.optim import RiemannianSGD
from ledoh_torch import AxisAlignedSlicedSphereDispersion

torch.manual_seed(42)
X_init = F.normalize(10 + torch.rand(10000, 128), dim=-1)
# print(torch.norm(X_init, dim=1).shape)
# print(
#     torch.isclose(torch.norm(X_init, dim=1), torch.tensor([1.]))
#     .all()
# )

manifold = SphereExact()
X = ManifoldParameter(X_init, manifold=manifold)
opt = RiemannianSGD([X], lr=0.01, stabilize=1)

last_grad_norm = None

for i in range(80):
    # print((torch.norm(X, dim=1) == 1))
    # print((torch.norm(X, dim=1) == 1).any())
        
    opt.zero_grad()
    loss = AxisAlignedSlicedSphereDispersion.forward(X, i=None, j=None)
    
    if torch.isnan(loss):
        print(f"Loss is NaN: i={i}")
        print(f"Grad norm in prev iteration: {last_grad_norm}")
        break
    
   
    loss.backward()
    last_grad_norm = X.grad.norm().detach()
    opt.step()
