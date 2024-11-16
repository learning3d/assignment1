import torch
import pytorch3d
from starter.utils import get_mesh_renderer, load_cow_mesh, get_device
import os
import matplotlib.pyplot as plt

device = get_device()

# Rendering your first mesh
renderer = get_mesh_renderer()

vertices, faces = load_cow_mesh(path="data/cow.obj")

vertices = vertices.unsqueeze(0)  # Add a batch dimension
faces = faces.unsqueeze(0)  # Add a batch dimension

texture_rgb = torch.ones_like(vertices)  # N x 3
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
textures = pytorch3d.renderer.TexturesVertex(texture_rgb)  # important

meshes = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=textures,  # Correctly pass the textures object
)

cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=torch.eye(3).unsqueeze(0),
    T=torch.tensor([[0, 0, 3]]),
    fov=60,
)


lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]])

meshes = meshes.to(device)
cameras = cameras.to(device)
lights = lights.to(device)

rend = renderer(meshes, cameras=cameras, lights=lights)
image = rend[0, ..., :3].cpu().numpy()

output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.imsave(f"{output_dir}/cow_render.jpg", image)
