import os
import torch
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from P_LymphCS.custom.comp import extract, print_feature_hook_LpCTransVss, reg_hook_on_module, init_from_model

class ImageDataset(Dataset):
    def __init__(self, image_paths, transformer):
        self.image_paths = image_paths
        self.transformer = transformer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transformer(image)
        return image, image_path


def feature_extract(model_name, data, model_path):
    if model_name == 'LpCTransVss':
        feature_name = 'norm'
        transform = transforms.Compose([
            transforms.Resize((512, 512)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model, transformer, device = init_from_model(model_name=model_name, num_classes=5, input_size=512,
                                                 model_path=model_path)
    model.to(device)
    model.eval()
    save_path = r'./Output/Patch_Features'
    os.makedirs(save_path, exist_ok=True)

    for entry in os.listdir(data):
        full_path = os.path.join(data, entry)
        if os.path.isdir(full_path):
            test_samples = [os.path.join(full_path, p) for p in os.listdir(full_path) if
                            p.endswith('.png') or p.endswith('.jpg')]
            csv_filename = f"{entry}.csv"
            csv_path = os.path.join(save_path, csv_filename)

            with open(csv_path, 'w') as outfile:
                try:
                    hook = lambda module, inp, outp: print_feature_hook_LpCTransVss(module, inp, outp, outfile)
                    hook_handles = reg_hook_on_module(feature_name, model, hook)
                    dataset = ImageDataset(test_samples, transform)
                    dataloader = DataLoader(dataset, batch_size=1, num_workers=16)
                    for batch in dataloader:
                        images, paths = batch
                        results = extract(list(zip(images, paths)), model, transformer, device, fp=outfile)

                finally:
                    for handle in hook_handles:
                        handle.remove()
                outfile.close()
