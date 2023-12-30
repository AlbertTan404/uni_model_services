from pathlib import Path
from models.base import WrappedBASE


class WrappedTEST(WrappedBASE):
    def __init__(self, name_or_path: str, outputs_dir: Path, device: str, **kwargs) -> None:
        super().__init__(name_or_path, outputs_dir, device, **kwargs)

        print(name_or_path, outputs_dir, device, kwargs)
    
    def call(self, **data):
        return f'calling call({data})'
    
    def foo(self, **data):
        return f'calling foo({data})'
