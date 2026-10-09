"""Dataset registry; all default paths are relative to the artifact root."""
from importlib import import_module

DATASET_CONFIG = {
    'fm': {'data_dir': 'data/fashionmnist', 'target_layer': 'fc2'},
    'cifar10': {'data_dir': 'data/cifar10', 'target_layer': 'fc2'},
    'gtsrb': {'data_dir': 'data/gtsrb', 'target_layer': 'fc2'},
}
MODEL_NAMES = {'fm': 'ArachneModel', 'cifar10': 'ArachneCIFAR10Model',
               'gtsrb': 'ArachneGTSRBModel'}


def get_model_and_loader(dataset):
    if dataset not in MODEL_NAMES:
        raise ValueError(f'Unknown dataset: {dataset}')
    model = import_module(f'src.models.{dataset}')
    data = import_module(f'src.data_loading.{dataset}')
    return getattr(model, MODEL_NAMES[dataset]), data.get_loaders
