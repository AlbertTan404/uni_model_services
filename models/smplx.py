from typing import Dict
import os
from collections import defaultdict
import tqdm
import numpy as np
import torch
import imageio
import smplx
from smplx import SMPLX
import pyrender
import trimesh

from models.base import WrappedBASE


class WrappedSMPLX(WrappedBASE):
    I2V_COMMAND = 'ffmpeg -framerate 24 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p {}/0.mp4'

    layer_args = {
        'create_global_orient': False, 
        'create_body_pose': False, 
        'create_left_hand_pose': False, 
        'create_right_hand_pose': False, 
        'create_jaw_pose': False, 
        'create_leye_pose': False, 
        'create_reye_pose': False, 
        'create_betas': False, 
        'create_expression': False, 
        'create_transl': False}

    def __init__(self, name_or_path, outputs_dir, device, **kwargs) -> None:
        super().__init__(name_or_path, outputs_dir, device, **kwargs)

        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        self.smplx_model: SMPLX = smplx.create(
            name_or_path,
            model_type='smplx',
            gender='NEUTRAL', 
            use_pca=False, 
            use_face_contour=True, 
            **self.layer_args
        ).to(self.device).eval()

        self.renderer = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)

    def get_preprocessed_frame(self, frame: Dict[str, np.array]) -> Dict[str, torch.Tensor]:
        res = {}
        for k, v in frame.items():
            res[k] = torch.from_numpy(v).float().view(1, -1).to(self.device)
        return res

    def batch_motion(self, motion):
        res = defaultdict(torch.Tensor)
        keys = motion[0].keys()

        for key in keys:
            res[key] = torch.cat([torch.from_numpy(frame[key]).float().view(1, -1) for frame in motion], dim=0).to(self.device)
        return res 

    def call(self, **kwargs):
        # seg_path = get_random_seg_path(data_root_dir=data_root_dir)
        seg_path = kwargs['data_path']

        current_save_dir = self.outputs_dir / '_'.join(str(seg_path).split('/')[-3:])
        current_save_dir.mkdir(parents=True)

        with (seg_path/'smplx.npy').open('rb') as f:
            motion = np.load(f, allow_pickle=True)

        batch = self.batch_motion(motion)

        # fix the orientation
        batch['global_orient'] = torch.zeros_like(batch['global_orient'])

        # subtract orientation by the first frame
        # begin_orientation = batch['global_orient'][0].clone()
        # batch['global_orient'] -= begin_orientation
        # batch['global_orient'] = (batch['global_orient'] + torch.pi) % (2 * torch.pi) - torch.pi

        smplx_output = self.smplx_model(**batch)
        vertices = smplx_output.vertices.detach().squeeze().cpu().numpy()

        for i in tqdm.trange(smplx_output.vertices.shape[0]):
            scene = pyrender.Scene()
            scene.bg_color = [0,0,0,1]

            tri_mesh = trimesh.Trimesh(vertices[i], self.smplx_model.faces)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            scene.add(mesh)

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            camera_translation = mesh.centroid + np.array([0, 0, mesh.scale])
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_translation
            scene.add(camera, pose=camera_pose)

            direc_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)  
            scene.add(direc_light, pose=camera_pose)  
            color, depth = self.renderer.render(scene)
            imageio.imwrite(str(current_save_dir/f'{i}.png'), color)

        os.system(self.I2V_COMMAND.format(str(current_save_dir), str(current_save_dir)))
        return str(current_save_dir)
