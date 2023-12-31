"""
- Filename:  `model's name or abbreviation in lowercase`
- Classname: `Wrapped` + `model's name or abbreviation in UPPERCASE`
"""

import torch
from pathlib import Path


class WrappedBASE:
    def __init__(self, name_or_path: str, outputs_dir: Path, device: str, **kwargs) -> None:
        self.name_or_path = name_or_path
        self.outputs_dir = outputs_dir
        self.device = device

    def __call__(self, **kwargs):
         with torch.no_grad():
            return self.call(**kwargs)
    
    def call(self, **kwargs):
       raise NotImplementedError
