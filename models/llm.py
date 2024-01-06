from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.base import WrappedBASE


class WrappedLLM(WrappedBASE):
    def __init__(self, name_or_path: str, outputs_dir: Path, device: str, **kwargs) -> None:
        super().__init__(name_or_path, outputs_dir, device, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, device_map=device, torch_dtype=torch.float16, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
 
    def call(self, **kwargs):
        prompt = kwargs.pop('prompt')
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        res = self.model.generate(
            **inputs,
            **kwargs
        )
        return self.tokenizer.decode(res[0][len(inputs['input_ids'][0]): ], skip_special_tokens=True)

    def qa(self, **kwargs):
        prompt_template = '###System: You are a helpful, respectful and honest assistant. ###User:{}. ###Assistant:'
        kwargs['prompt'] = prompt_template.format(kwargs['prompt'])
        return self(**kwargs)  # __call__
