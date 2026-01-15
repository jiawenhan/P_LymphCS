
from P_LymphCS import models

__all__ = ['create_model']


def create_model(model_name, **kwargs):
    supported_models = [k for k in models.__dict__
                        if not k.startswith('_') and type(models.__dict__[k]).__name__ != 'module']
    if model_name in supported_models:
        return models.__dict__[model_name](**kwargs)
    raise ValueError(f'{model_name} not supported!')
