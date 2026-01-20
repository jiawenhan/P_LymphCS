import os
from typing import Iterable

import numpy as np
import torch
from P_LymphCS.core import create_model
from P_LymphCS.core import create_standard_image_transformer


def extract(samples, model, transformer, device=None, fp=None):
    results = []
    # Inference
    if not isinstance(samples, (list, tuple)):
        samples = [samples]
    model.eval()

    with torch.no_grad():
        for sample, path in samples:
            if fp is not None:
                fp.write(f"{os.path.basename(path)},")
            sample = sample.to(device)
            outputs = model(sample.unsqueeze(0))
            results.append(outputs)
    return results

def print_feature_hook_LpCTransVss(module, inp, outp, fp, post_process=None):
    features = outp.mean(dim=1).cpu().numpy()
    if post_process is not None:
        features = post_process(features)
    print(','.join(map(lambda x: f"{x:.6f}", np.reshape(features, -1))), file=fp)


def reg_hook_on_module(name, model, hook):
    handles = []
    find_ = 0
    for n, m in model.named_modules():
        if name == n:
            handle = m.register_forward_hook(hook)
            handles.append(handle)
            find_ += 1
    return handles


def init_from_model(
    model_name: str,
    num_classes: int,
    input_size: int or tuple,
    model_path: str
):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    transform_config = {'phase': 'valid', 'input_size': input_size}
    transformer = create_standard_image_transformer(**transform_config)
    model_config = {
        'model_name': model_name,
        'num_classes': num_classes
    }
    model = create_model(**model_config)
    model_path = os.path.join(model_path, 'P_LymphCS.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)['model_state_dict']
    for key in list(state_dict.keys()):
        if key.startswith('module.'):
            new_key = key[7:]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
    model.eval()
    return model, transformer, device


def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()

