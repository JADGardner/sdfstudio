# %%
# for every ray there is an associated camera, I want to collect all the rays
# for each camera to run through reni and get the background colour and then
# put the colour back into the same shape as the original directions tensor
# however cameras can appear a random number of times.
import torch


def so2_invariant_representation(Z, D):
    """Generates an SO2 invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x npix x 3)

    Returns:
        torch.Tensor: SO2 invariant representation (B x npix x 2 x ndims + ndims^2 + 2)
    """
    Z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
    D_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
    # Invariant representation of Z, gram matrix G=Z*Z' is size B x ndims x ndims
    G = torch.bmm(Z_xz, torch.transpose(Z_xz, 1, 2))
    # Flatten G and replicate for all pixels, giving size B x npix x ndims^2
    Z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
    # innerprod is size B x npix x ndims
    innerprod = torch.bmm(D_xz, torch.transpose(Z_xz, 1, 2))
    D_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)
    # Copy Z_y for every pixel to be size B x npix x ndims
    Z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)
    # Just the y component of D (B x npix x 1)
    D_y = D[:, :, 1].unsqueeze(2)
    # Conditioning via concatenation
    model_input = torch.cat((innerprod, Z_xz_invar, D_xz_norm, Z_y, D_y), 2)
    # model_input is size B x npix x 2 x ndims + ndims^2 + 2
    return model_input


def so2_invariant_representation_4D(Z, D):
    """Generates an SO2 invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x N x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x N x 3)

    Returns:
        torch.Tensor: SO2 invariant representation (B x npix x 2 x ndims + ndims^2 + 2)
    """
    Z = latent_codes[camera_idxs, :]
    D = directions
    Z = Z.reshape(-1, Z.shape[-2], Z.shape[-1])  # (K x ndims x 3)
    D = D.reshape(-1, D.shape[-1])  # (K x 3)
    Z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
    D_xz = torch.stack((D[:, 0], D[:, 2]), -1)
    # Invariant representation of Z, gram matrix G=Z*Z' is size npix x ndims x ndims
    G = torch.bmm(Z_xz, torch.transpose(Z_xz, 1, 2))
    Z_xz_invar = G.flatten(start_dim=1).unsqueeze(0)  # (1 x K x ndims^2)
    # innerprod is size B x npix x ndims unsqueeze to add batch dim
    innerprod = torch.einsum("ij,ikj->ik", D_xz, Z_xz).unsqueeze(0)  # (1 x K x ndims)
    D_xz_norm = torch.norm(D_xz, dim=1).unsqueeze(0).unsqueeze(2)  # (1 x K x 1)
    Z_y = Z[:, :, 1].unsqueeze(0)  # (1 x K x ndims)
    # Just the y component of D
    D_y = D[:, 1].unsqueeze(0).unsqueeze(2)  # (1 x K x 1)
    # Conditioning via concatenation
    model_input = torch.cat((innerprod, Z_xz_invar, D_xz_norm, Z_y, D_y), 2)
    # model_input is size B x npix x 2 x ndims + ndims^2 + 2
    return model_input


def reni_forward(Z, D):
    """
    psuedo function for reni
    Z: [N, K, 3]
    D: [N, D, 3]
    """
    assert Z.shape[0] == D.shape[0]  # batch is the same size
    assert Z.shape[2] == D.shape[2] == 3  # 3D
    return torch.randn((Z.shape[0], D.shape[1], 3))


number_of_cameras = 10
latent_code_dims = 9

latent_codes = torch.randn((number_of_cameras, latent_code_dims, 3))

batch_size = 2  # number of rays
cameras_per_batch = 6

camera_idxs = torch.randint(0, number_of_cameras, (batch_size, cameras_per_batch))
# unique camera indices can appear in a random number of batches
directions = torch.randn((batch_size, cameras_per_batch, 3))
# %%
unique_idxs, inverse_idxs = torch.unique(camera_idxs, return_inverse=True)
# %%
latent_codes_unique_idxs = latent_codes[unique_idxs, :, :]  # Z for reni forward
# %%
# so now I want a tensor of shape [unique_idxs, directions_for_that_idx, 3]
# but directions_for_that_camera will be jagged along dim=1 as depends on how many times
# that camera randomly appeared in a batch

# for example:
a, b, c = unique_idxs[0], unique_idxs[1], unique_idxs[2]
if directions[camera_idxs == a].shape == directions[camera_idxs == b].shape == directions[camera_idxs == c].shape:
    print("all the same, by chance")
else:
    print("oh no we will have a jagged tensor")  # need to pad??

# in the ideal case where tensor not jagged then pass to reni_forward
ideal_num_directions = 2
directions_for_unique_idxs = torch.randn((unique_idxs.shape[0], ideal_num_directions, 3))
background_colour = reni_forward(latent_codes_unique_idxs, directions)  # [unique_idxs, ideal_num_directions, 3]

# now need to put the colours back into the shape [batch_size, cameras_per_batch, 3]

# %%
Z_in = latent_codes[camera_idxs, :]
D_in = directions
# %%
Z = latent_codes[camera_idxs, :]
D = directions
Z = Z.reshape(-1, Z.shape[-2], Z.shape[-1])  # (K x ndims x 3)
D = D.reshape(-1, D.shape[-1])  # (K x 3)
Z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
D_xz = torch.stack((D[:, 0], D[:, 2]), -1)
# Invariant representation of Z, gram matrix G=Z*Z' is size npix x ndims x ndims
G = torch.bmm(Z_xz, torch.transpose(Z_xz, 1, 2))
Z_xz_invar = G.flatten(start_dim=1).unsqueeze(0)  # (1 x K x ndims^2)
# innerprod is size B x npix x ndims unsqueeze to add batch dim
innerprod = torch.einsum("ij,ikj->ik", D_xz, Z_xz).unsqueeze(0)  # (1 x K x ndims)
D_xz_norm = torch.norm(D_xz, dim=1).unsqueeze(0).unsqueeze(2)  # (1 x K x 1)
Z_y = Z[:, :, 1].unsqueeze(0)  # (1 x K x ndims)
# Just the y component of D
D_y = D[:, 1].unsqueeze(0).unsqueeze(2)  # (1 x K x 1)
# Conditioning via concatenation
model_input = torch.cat((innerprod, Z_xz_invar, D_xz_norm, Z_y, D_y), 2)
# model_input is size B x npix x 2 x ndims + ndims^2 + 2
print(model_input.shape)

# %%
Z = torch.randn((5, 9, 3))
D = torch.randn((5, 10, 3))
Z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
D_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
# Invariant representation of Z, gram matrix G=Z*Z' is size B x ndims x ndims
G = torch.bmm(Z_xz, torch.transpose(Z_xz, 1, 2))
# Flatten G and replicate for all pixels, giving size B x npix x ndims^2
Z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
# innerprod is size B x npix x ndims
innerprod = torch.bmm(D_xz, torch.transpose(Z_xz, 1, 2))
D_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)  # B x npix x 1
# Copy Z_y for every pixel to be size B x npix x ndims
Z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)
# Just the y component of D (B x npix x 1)
D_y = D[:, :, 1].unsqueeze(2)
# Conditioning via concatenation
model_input = torch.cat((innerprod, Z_xz_invar, D_xz_norm, Z_y, D_y), 2)
# model_input is size [B x npix x (2 * ndims + ndims^2 + 2)]

# %%
import torch

x = torch.randn((2, 3, 4), requires_grad=True)
print(x.requires_grad)
torch.is_grad_enabled()
with torch.no_grad():
    print(torch.is_grad_enabled())
import matplotlib.pyplot as plt
import numpy as np

# %%
import torch
from PIL import Image

semantic = Image.open("data/NeRF-OSR/Data/stjacob/final/train/cityscapes_mask/10-08_19_00_IMG_8285.png")
img = np.array(semantic)
plt.imshow(img)

# is_sky = semantic != 2  # sky id is 2
# fg_masks.append(torch.from_numpy(is_sky).unsqueeze(-1))
# %%
# plot image


# %%
is_sky = semantic != 2
# %%

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.io import load_from_json


def find_files(directory, exts):
    """Find all files in a directory that have a certain file extension.

    Parameters
    ----------
    directory : str
        The directory to search for files.
    exts : list of str
        A list of file extensions to search for. Each file extension should be in the form '*.ext'.

    Returns
    -------
    list of str
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    if os.path.isdir(directory):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    return []


semantic_dir = "/workspaces/sdfstudio/data/NeRF-OSR/Data/stjacob/final/train/cityscapes_mask"
segmentation_filenames = find_files(f"{semantic_dir}", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])
panoptic_classes = load_from_json(
    Path("/workspaces/sdfstudio/data/NeRF-OSR/Data/stjacob/final/train/cityscapes_classes.json")
)
classes = panoptic_classes["classes"]
colors = torch.tensor(panoptic_classes["colours"], dtype=torch.long) / 255.0

# %%
segmentation_image = Image.open(segmentation_filenames[245])
segmentation_image = torch.tensor(np.array(segmentation_image), dtype=torch.int64) / 255.0

# Define the specific classes you want to mask
sky_class = colors[classes.index("sky")]
car_class = colors[classes.index("car")]

# Create an empty mask with the same shape as your image
mask = torch.ones_like(segmentation_image[:, :, 0], dtype=torch.long)

# Fill the mask with 1s for the pixels that belong to the sky or car classes
mask = torch.where(
    torch.all(torch.eq(segmentation_image, colors[classes.index("vegetation")]), dim=2)
    | torch.all(torch.eq(segmentation_image, colors[classes.index("person")]), dim=2)
    | torch.all(torch.eq(segmentation_image, colors[classes.index("rider")]), dim=2)
    | torch.all(torch.eq(segmentation_image, colors[classes.index("car")]), dim=2)
    | torch.all(torch.eq(segmentation_image, colors[classes.index("truck")]), dim=2)
    | torch.all(torch.eq(segmentation_image, colors[classes.index("bus")]), dim=2)
    | torch.all(torch.eq(segmentation_image, colors[classes.index("train")]), dim=2)
    | torch.all(torch.eq(segmentation_image, colors[classes.index("motorcycle")]), dim=2)
    | torch.all(torch.eq(segmentation_image, colors[classes.index("bicycle")]), dim=2),
    torch.zeros_like(mask),
    mask,
)

# get index in dictionary classes of sky
# %%
plt.imshow(segmentation_image)
# %%
plt.imshow(mask)
# %%
mask.dtype
# %%
mask
# %%
mask.bool()
# %%
import torch

D = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23, 24],
        [25, 26, 27],
    ]
)
camera_indices = torch.tensor([1, 1, 0, 0, 2, 2, 3, 3, 3])

unique_camera_indices, camera_index_mapping = torch.unique(camera_indices, return_inverse=True)
num_unique_cameras = unique_camera_indices.shape[0]
num_rays = D.shape[0]

# %%
rays_per_camera = num_rays // num_unique_cameras
sorted_indices = torch.argsort(camera_index_mapping)
D = D[sorted_indices]
K = D.reshape(num_unique_cameras, rays_per_camera, 3)

# %%
rays_per_camera = num_rays // num_unique_cameras
sorted_indices = torch.argsort(camera_index_mapping)
D = D[sorted_indices]
K = torch.split(D, rays_per_camera, dim=0)
# %%
import torch

# load /workspaces/sdfstudio/h.pt
h = torch.load("/workspaces/sdfstudio/h.pt")
h
# %%
if not torch.isnan(h).any():
    print("nan")
# %%
capture_sessions = {
    "sessions": {
        "session1": {"image_ids": [1, 2, 3, 4, 5], "envmaps": None, "segmap": None, "fg_mask": None},
        "session2": {"image_ids": [1, 2, 3, 4, 5], "envmaps": None, "segmap": None, "fg_mask": None},
    }
}

for session in capture_sessions["sessions"]:
    print(session)
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display
from PIL import Image
from torchvision.transforms import InterpolationMode, Resize, ToTensor
from torchvision.utils import make_grid

from nerfstudio.fields.reni_field import get_directions, get_reni_field, get_sineweight
from nerfstudio.model_components.losses import RENITestLoss
from nerfstudio.utils.colormaps import sRGB
from nerfstudio.utils.io import load_from_json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

W = 256
H = W // 2
resize = Resize(
    (H, W),
    InterpolationMode.NEAREST,
)

envmap_gt = Image.open(
    "/workspaces/sdfstudio/data/NeRF-OSR/Data/stjacob/final/ENV_MAP_CC/01-08_7_30/20210801_085329.jpg"
)
envmap_gt = ToTensor()(envmap_gt)
envmap_gt = resize(envmap_gt)  # shape (3, H, W)
envmap_gt = envmap_gt.permute(1, 2, 0)  # shape (H, W, 3)

panoptic_classes = load_from_json(
    Path("/workspaces/sdfstudio/data/NeRF-OSR/Data/stjacob/final/validation/cityscapes_classes.json")
)
classes = panoptic_classes["classes"]
colors = torch.tensor(panoptic_classes["colours"], dtype=torch.float32) / 255.0

segmentation_image = Image.open(
    "/workspaces/sdfstudio/data/NeRF-OSR/Data/stjacob/final/ENV_MAP_CC/01-08_7_30/cityscapes_mask/20210801_085329.png"
)
segmentation_image = ToTensor()(segmentation_image)
segmentation_image = resize(segmentation_image)  # shape (3, H, W)
segmentation_image = segmentation_image.permute(1, 2, 0)  # shape (H, W, 3)

fg_mask = torch.zeros_like(segmentation_image[:, :, 0])
fg_mask = torch.where(
    torch.all(torch.eq(segmentation_image, colors[classes.index("sky")]), dim=2),
    torch.ones_like(fg_mask),
    fg_mask,
)
# %%
def plot_on_epoch_end(imgs, model_output, mask, H, W):
    imgs = imgs.view(-1, H, W, 3)
    model_output = model_output.view(-1, H, W, 3)
    mask = mask.view(-1, H, W, 3)
    imgs = imgs.permute(0, 3, 1, 2)  # (B, C, H, W)
    model_output = model_output.permute(0, 3, 1, 2)  # (B, C, H, W)
    mask = mask.permute(0, 3, 1, 2)  # (B, C, H, W)
    masked_imgs = imgs * mask

    imgs = torch.concat([imgs, masked_imgs, model_output], dim=0)
    img_grid = make_grid(imgs, nrow=3, pad_value=2)
    img_grid = img_grid.permute(1, 2, 0).cpu().detach().numpy()
    img_grid = (img_grid * 255).astype(np.uint8)
    plt.imshow(img_grid)
    plt.axis("off")
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.show()


reni = get_reni_field(
    "checkpoints/reni_pretrained_weights/latent_dim_36_net_5_256_vad_cbc_tanh_hdr/version_0/checkpoints/fit_decoder_epoch=1589.ckpt",
    num_latent_codes=2,
)

reni.to(device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 500
opt = torch.optim.Adam(reni.parameters(), lr=1e-1)
criterion = RENITestLoss(alpha=1e-9, beta=1e-1)

# # directions to sample the reni field
directions = get_directions(W)  # [1, H*W, 3]
# sineweight compensation for the irregular sampling of the equirectangular image
sineweight = get_sineweight(W)  # [1, H*W, 3]

for epoch in range(epochs):
    idx = 0
    envmap = envmap_gt.to(device)  # [H, W]
    envmap = envmap.unsqueeze(0)  # [1, H, W, 3]
    envmap = envmap.reshape(1, -1, 3)  # [1, H*W, 3]
    mask = fg_mask.to(device)  # [H, W]
    mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
    mask = mask.repeat(1, 1, 1, 3)  # [1, H, W, 3]
    mask = mask.reshape(1, -1, 3)  # [1, H*W, 3]

    D = directions.type_as(envmap)
    S = sineweight.type_as(envmap)
    S = S * mask

    Z = reni.get_Z()[idx, :, :].unsqueeze(0)  # [1, ndims, 3]

    model_output = reni(Z, D)  # [1, H*W, 3]
    model_output = reni.unnormalise(model_output)  # [1, H*W, 3]
    model_output = sRGB(model_output)  # [1, H*W, 3]

    opt.zero_grad()
    loss, _, _, _ = criterion(model_output, envmap, S, Z)
    loss.backward()
    opt.step()
    plt.imshow(model_output[0].cpu().detach().numpy())
    plt.axis("off")
    if not (epoch) % 20:
        plot_on_epoch_end(envmap, model_output, mask, H, W)

# %%
import time

from rich.progress import track

loss = 0

for i in track(range(20), description=f"Processing... {loss:.2f}"):
    loss = 0.5 * i  # Replace with actual loss value
    time.sleep(1)  # Simulate work being done
# %%
from time import sleep

from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Column

text_column = TextColumn("{task.description}", table_column=Column(ratio=1))
bar_column = BarColumn(bar_width=None, table_column=Column(ratio=2))
progress = Progress(text_column, bar_column, expand=True)

with progress:
    for n in progress.track(range(100)):
        progress.print(n)
        sleep(0.1)

import time

# %%
from rich.console import Console
from rich.progress import Progress

# Initialize console and progress bar
console = Console()
progress = Progress(console=console)

# Start progress bar
task = progress.add_task("[green]Training...", total=100)

# Loop through iterations and update progress bar
for i in range(100):
    loss = 0.5 * i  # Replace with actual loss value
    progress.update(task, advance=1, description=f"Loss: {loss:.4f}")
    time.sleep(0.1)  # Simulate training time

# Finish progress bar
progress.stop()

import time

# %%
from rich.progress import Console, track

CONSOLE = Console(width=120)

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
)

# Loop through iterations and update progress bar
for i in track(range(100), description="[green]Training..."):
    loss = 0.5 * i  # Replace with actual loss value
    CONSOLE.print(f"Loss: {loss:.4f}")
    time.sleep(0.1)  # Simulate training time

# %%\
import torch

directions = torch.randn(1, 256, 3)
sampler = lambda positions: directions.repeat(positions.shape[0], 1, 1)
# line above but specify positions is a tensor
sampler = lambda positions: directions.repeat(len(positions), 1, 1)
# %%
sampler([1, 2, 3]).shape
# %%
Z = torch.randn(1, 36, 3)
D = torch.randn(1, 256, 3)
z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
d_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
# Invariant representation of Z, gram matrix G=Z*Z' is size B x ndims x ndims
G = torch.bmm(z_xz, torch.transpose(z_xz, 1, 2))
# Flatten G and replicate for all pixels, giving size B x npix x ndims^2
z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
# innerprod is size B x npix x ndims
innerprod = torch.bmm(d_xz, torch.transpose(z_xz, 1, 2))
d_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)
# Copy Z_y for every pixel to be size B x npix x ndims
z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)
# Just the y component of D (B x npix x 1)
d_y = D[:, :, 1].unsqueeze(2)

model_input = torch.cat((d_xz_norm, d_y, innerprod), 2)  # [B, npix, 2 + ndims]
conditioning_input = torch.cat((z_xz_invar, z_y), 2)  # [B, npix, ndims^2 + ndims]


model_input_concat = torch.cat((innerprod, z_xz_invar, d_xz_norm, z_y, d_y), 2)

import matplotlib.pyplot as plt

# %%
import torch
from PIL import Image
from torchvision.transforms import ToTensor

img = "/workspaces/sdfstudio/outputs/data-NeRF-OSR-Data/RENI-NeuS/2023-02-27_121840/wandb/latest-run/files/media/images/Eval Images/visibility_100_988b2a7899666cd8a0c1.png"

img = Image.open(img)
img = ToTensor()(img)

# H, W = img.shape[1], img.shape[2]
# img = img.reshape(-1, 3)
# img = img.permute(1, 0)
# img = img.reshape(H, W, 3)
img = img.permute(1, 2, 0)

# now reshape back but using row-major order
# %%
import matplotlib.pyplot as plt
import icosphere
import torch
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
# %%
from nerfstudio.model_components.illumination_samplers import IcosahedronSampler

# %%
random_rotations = torch.eye(3).repeat(10, 1, 1)  # [n, 3, 3]
random_rotations[:, :3, :3] = torch.from_numpy(Rotation.random(10).as_matrix())
# %%
vertices, _ = icosphere.icosphere(3)
# %%
# plot vertices using plotly as points
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode="markers")])
fig.show()
# %%
sampler = IcosahedronSampler(3, True, True)
vertices = sampler.generate_direction_samples(5)

# %%
vertices, _ = icosphere.icosphere(3)  # [K, 3] and random rotations [N, 3, 3]
vertices = torch.from_numpy(vertices).float()
# unsqueeze to [1, K, 3] and repeat to [N, K, 3] and then apply rotations
vertices = vertices.unsqueeze(0).repeat(10, 1, 1)
vertices = torch.bmm(vertices, random_rotations)

# remove lower
# %%
vertices, _ = icosphere.icosphere(3)  # [K, 3]
R = torch.from_numpy(Rotation.random(1).as_matrix())[0].float()
vertices = torch.from_numpy(vertices).float()
vertices = vertices @ R
# %%
# remove lower hemisphere of vertices
vertices = vertices[vertices[:, 2] > 0]
# %%
fig = go.Figure(data=[go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode="markers")])
fig.show()
import plotly.graph_objects as go

# %%
from nerfstudio.model_components.illumination_samplers import IcosahedronSampler

sampler = IcosahedronSampler(4, True, True)
vertices = sampler.generate_direction_samples()
# double size of tensor with more vertgices from sampler
vertices = torch.cat((vertices, sampler.generate_direction_samples(), sampler.generate_direction_samples()), 0)
fig = go.Figure(data=[go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode="markers")])
fig.show()
# %%
import torch

# random N, 3 tensor
t = torch.randn(10, 3)

# random mask tensor of true and false values shape N, 1
mask = torch.rand(10, 1) > 0.5

# apply mask to tensor

t = t[mask.squeeze(), :]
# %%
t.shape
# %%
