"""
@author: Albert Wenhui Tan
@email: albert_twh@qq.com
@date: 2023-12-28
"""

import argparse
import importlib
from typing import Dict
from pathlib import Path
from flask import Flask, request


def process_kv(extra_kv: Dict[str, str]) -> Dict:
    for k, v in extra_kv.items():
        if '.' not in k:
            continue

        # fp
        try:
            val = float(v)
        except ValueError:
            pass
        else:
            extra_kv[k] = val
            continue
        
        # like torch.float16
        try:
            vals = v.split('.')
            if len(vals) != 2:
                continue
            module = importlib.import_module(vals[0])
            val = getattr(module, vals[1])
        except:
            pass
        else:
            extra_kv[k] = v
            continue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=['sd', 'llm', 'densepose', 'smplx', 'test'],
        # required=True,
        default='sd',
    )

    parser.add_argument(
        '--host',
        type=str,
        default='localhost'
    )

    parser.add_argument(
        '--port',
        type=int,
        # required=True,
        default=6691,
    )

    parser.add_argument(
        '--model_name_or_path',
        type=str,
        # required=True,
        default='~/data/pretrained_models/stable-diffusion-2-1'
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0
    )

    parser.add_argument(
        '--extra_kv',
        type=str,
        nargs='+',
        default=None,
        help="usage: --extra_kv k1 v1 k2 v2 k3 v3",
    )

    parser.add_argument(
        '--outputs_root_dir',
        type=str,
        default='~/data/outputs'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # model name
    model = args.model

    # process outputs directory
    outputs_dir = (Path(args.outputs_root_dir) / model).expanduser()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    # process model name or pretrained path
    model_name_or_path = args.model_name_or_path
    if model_name_or_path[:2] == '~/':
        model_name_or_path = str(Path(model_name_or_path).expanduser())

    # process extra key-values for initializing model
    if kv_list := args.extra_kv:
        assert len(kv_list) % 2 == 0
        import itertools
        extra_kv = {k: v for k, v in itertools.pairwise(kv_list)}
    else:
        extra_kv = dict()

    # initialize model
    model_cls = getattr(importlib.import_module(name='models.' + model), 'Wrapped' + model.upper())
    print(f'launching {model_cls}')
    model = model_cls(
        name_or_path=model_name_or_path, outputs_dir=outputs_dir, device=f'cuda:{args.device}', **extra_kv
    )

    app = Flask(__name__)

    @app.route('/call', methods=['POST'])
    def call():
        try:
            data = request.json
            return model(**data)
        except Exception as e:
            return {"error": str(e)}, 400

    @app.route('/other_func', methods=['POST'])
    def other_func():
        try:
            data = request.json
            func_name = data.pop('func_name')
            func = getattr(model, func_name)
            return func(**data)
        except Exception as e:
            return {"error": str(e)}, 400

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
