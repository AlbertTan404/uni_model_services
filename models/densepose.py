import os
import numpy as np
import cv2
from pathlib import Path
import torch
import imageio

from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image

from densepose.structures import DensePoseDataRelative
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.base import MatrixVisualizer


from models.base import WrappedBASE
from models.apply_net import create_argument_parser, DumpAction


class WrappedDENSEPOSE(WrappedBASE):
    I2V_COMMAND = 'ffmpeg -framerate 24 -i {}/%d.png -c:v libx264 -pix_fmt yuv420p {}/0.mp4'

    def __init__(self, name_or_path: str, outputs_dir: Path, device: str, **kwargs) -> None:
        super().__init__(name_or_path, outputs_dir, device, **kwargs)

        parser = create_argument_parser()
        args = parser.parse_args(args=[
            "dump",
            "../detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "../../pretrained_models/densepose_rcnn_R_50_FPN_s1x.pkl",
            "input.jpg",
            "--output",
            "dump.pkl"
        ])
        cfg = DumpAction.setup_config(args.cfg, args.model, args, opts=[])

        self.predictor = DefaultPredictor(cfg)
        self.context = DumpAction.create_context(args, cfg)

        self.img_size = 512
        val_scale = 255.0 / DensePoseDataRelative.N_PART_LABELS
        self.mask_visualizer = MatrixVisualizer(
            inplace=True, cmap=cv2.COLORMAP_VIRIDIS, val_scale=val_scale, alpha=1.0
        )

    def generate_densepose_to_session(self, **data):
        session_name = data['session_name']

        densepose_session_dir = densepose_outputs_dir / session_name
        densepose_session_dir.mkdir()

        for p in smplx_session_dir.glob('*.png'):
            img = read_image(str(p))
            idx = p.name.split('.')[0]  # idx.png

            outputs = self.predictor(img)["instances"].cpu()

            extractor = DensePoseResultExtractor()
            data = extractor(outputs)
            densepose_result, boxes_xywh = data

            matrix_scaled_8u = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            matrix_vis = cv2.applyColorMap(matrix_scaled_8u, cv2.COLORMAP_VIRIDIS)

            result = densepose_result[0]
            iuv_array = torch.cat(
                (result.labels[None].type(torch.float32), result.uv * 255.0)
            ).cpu().type(torch.uint8).cpu().numpy()

            bbox_xywh = boxes_xywh.cpu().numpy()[0]

            matrix = iuv_array[0, :, :]
            segm = iuv_array[0, :, :]

            mask = np.zeros(matrix.shape, dtype=np.uint8)
            mask[segm >= 0] = 1
            self.mask_visualizer.visualize(matrix_vis, mask, matrix, bbox_xywh)
            imageio.imwrite(str(densepose_session_dir/f'{idx}.png'), matrix_vis)

        os.system(self.I2V_COMMAND.format(str(densepose_session_dir), str(densepose_session_dir)))
        return str(densepose_session_dir / '0.mp4')
