import torch
import diffusers

from models.base import WrappedBASE


class WrappedSD(WrappedBASE):
    def __init__(self, name_or_path, outputs_dir, device, **kwargs) -> None:
        super().__init__(name_or_path, outputs_dir, device)

        pipe_cls = self.get_pipe_cls(name_or_path)

        if pipe_cls == diffusers.StableDiffusionControlNetPipeline:
            # add controlnet
            kwargs['controlnet'] = diffusers.ControlNetModel.from_pretrained(name_or_path)
            # switch name_or_path to base model
            name_or_path = kwargs.pop('base_model')

        self.pipe = pipe_cls.from_pretrained(pretrained_model_name_or_path=name_or_path, torch_dtype=torch.float16, **kwargs).to(device)
    
    @staticmethod
    def get_pipe_cls(name_or_path: str):
        name_lower = name_or_path.lower()

        if 'controlnet' in name_lower:
            return diffusers.StableDiffusionControlNetPipeline

        elif 'stable-diffusion-xl' in name_lower:
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
