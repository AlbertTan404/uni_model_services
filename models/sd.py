import torch
import diffusers

from models.base import WrappedBASE


class WrappedSD(WrappedBASE):
    def __init__(self, name_or_path, outputs_dir, device, **kwargs) -> None:
        super().__init__(name_or_path, outputs_dir, device)

        pipe_cls = self.get_pipe_cls(name_or_path)

        self.pipe = pipe_cls.from_pretrained(pretrained_model_name_or_path=name_or_path, torch_dtype=torch.float16, **kwargs).to(device)
    
    @staticmethod
    def get_pipe_cls(name_or_path: str):
        if 'stable-diffusion-xl' in name_or_path:
            return diffusers.StableDiffusionXLPipeline
        else:
            return diffusers.StableDiffusionPipeline

    def switch_scheduler(self, scheduler_name: str):
        scheduler_cls = getattr(diffusers, scheduler_name)
        self.pipe.scheduler = scheduler_cls.from_config(self.pipe.scheduler.config)
        return f'scheduler switched to {scheduler_cls}'

    def call(self, **kwargs):
        image = self.pipe(**kwargs).images[0]

        save_path = str(self.outputs_dir / ('_'.join(kwargs['prompt'].split())[:30] + '.png'))
        image.save(save_path)
        return save_path
