# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: torchcfm2
#     language: python
#     name: torchcfm-vanilla
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import os
from typing import Optional, Callable

import matplotlib.pyplot as plt
import torch
import torchdiffeq
import torchsde
import tifffile
from torchdyn.core import NeuralODE
from torchvision.transforms import v2
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel

savedir = "models/cond_mnist"
os.makedirs(savedir, exist_ok=True)

# %%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
n_epochs = 100

# %%
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TIFFDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str,
                 num_classes: int = 2,
                 allowed_indices: list = [0],
                 transform: Optional[Callable] = None):
        """
        Args:
            file_path (str): Path to the TIFF file
            transform (callable, optional): Optional transform to be applied on images
        """
        self.file_path = file_path
        self.transform = transform
        self.num_classes = num_classes
        self.allowed_indices = allowed_indices

        # Read all pages from TIFF file
        self.tifhandle = tifffile.TiffFile(file_path)
        self.pages = self.tifhandle.pages
        self.num_pages = len(self.pages)


    def __len__(self) -> int:
        return len(self.allowed_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx (int): Index of the page to retrieve

        Returns:
            torch.Tensor: Transformed image tensor
        """
        # Read the specific page
        idx_ = self.allowed_indices[idx]
        assert idx_ < self.num_pages
        page = self.pages[idx_]
        image = page.asarray()

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # TODO: document!
        return image, idx % self.num_classes


# %%
transforms = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[-14.9965996,], std=[10.962,]),
])

trainset = TIFFDataset("wigner_onequarter_resaved_v2_1to4.tiff",
                       #we explicitely overfit on image at index 2
                       allowed_indices=[2],
                       transform = transforms)
assert len(trainset) == 1, f"trainset contains different number of items than expected"

batch_size = 1

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, drop_last=False
)

# %%
#################################
#    Class Conditional CFM
#################################

sigma = 0.0
model = UNetModel(
    dim=(1, 256, 256),
    num_channels=64,
    channel_mult = (8, 8, 8, 8, 8, 8),
    num_res_blocks=8,
    num_heads=32,
    num_classes=2, class_cond=True
).to(device)

num_params = count_trainable_params(model)
print(f"training Unet with {num_params} parameters")

optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)
# Users can try target FM by changing the above line by
# FM = TargetConditionalFlowMatcher(sigma=sigma)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

# %%
for epoch in range(n_epochs):
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x1 = data[0].to(device)
        assert x1.shape == (batch_size,1,256,256), f"found data in x1 not matching expected shape {x1.shape}"

        y = data[1].to(device)
        assert y.shape == (batch_size,), f"found data in y not matching expected shape {y.shape}"
        x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = model(t, xt, y)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")

# %%
USE_TORCH_DIFFEQ = True
repeat_by = 9
generated_class_list = torch.zeros((repeat_by,)).int().to(device)
num_generated = generated_class_list.shape[0]
num_steps = 50
print(f"generated {generated_class_list.shape[0]} dummy conditions for inference")
with torch.no_grad():
    traj = torchdiffeq.odeint(
        lambda t, x: model.forward(t, x, generated_class_list),
        torch.randn(num_generated, 1, 256, 256, device=device),
        torch.linspace(0, 1, num_steps, device=device),
        atol=1e-4,
        rtol=1e-4,
        method="dopri5",
    )

grid = make_grid(torch.cat(
    [trainset[0][0].unsqueeze(0).to(device),
     traj[-1, :num_generated].view([-1, 1, 256, 256]).clip(-1, 1)
     ], axis=0),
    value_range=(-1, 1),
    padding=4,
    nrow=5
)
img = ToPILImage()(grid)
plt.imshow(img)
plt.suptitle(f"Unet with {num_params} parameters after {n_epochs} epochs ({num_steps} steps during inference)")
plt.show()

output_fname = "wigner_onequarter_resaved_v2_1to4_idx2.png"
plt.savefig(output_fname)
print(f"Stored generated images to {output_fname}")
