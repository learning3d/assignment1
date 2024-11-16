import torch
import pytorch3d
from starter.utils import get_mesh_renderer, load_cow_mesh


#Rendering your first mesh

mesh_render = get_mesh_renderer()

vertices, faces = load_cow_mesh(path="data/cow.obj")

texture_rgb = torch.ones_like(vertices) # N X 3
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
texture_rgb = texture_rgb.unsqueeze(0)  
textures = pytorch3d.renderer.TexturesVertex(texture_rgb) # important

meshes = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=textures,
)

cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=torch.eye(3).unsqueeze(0),
    T=torch.tensor([[0, 0, 3]]),
    fov=60,
)

print(cameras.get_camera_center())