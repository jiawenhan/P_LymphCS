import os
import math
import numpy as np
import torch
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from P_LymphCS.custom.utils import GradCAM, show_cam_on_image, center_crop_img
from P_LymphCS.models.vits.LpCTransVss import P_LymphCS


def reshape_transform(x):
    if isinstance(x, tuple):
        x = x[0]
    # 4D/3D
    if len(x.shape) == 4:
        return x.permute(0, 2, 3, 1)
    elif len(x.shape) == 3:
        batch_size = x.shape[0]
        channels = x.shape[-1]
        height, width = 16, 16 
        return x.reshape(batch_size, height, width, channels)
    else:
        raise ValueError(f"Error：{x.shape}")


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))
        result = result.permute(0, 3, 1, 2)

        return result


def model_gradcam(model_path, data, target_category):
    img_size = 512
    assert img_size % 32 == 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = P_LymphCS()
    weights_path = os.path.join(model_path, 'P_LymphCS.pth')
    save_dir = r'./Output/GradCam'

    weights_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights_dict['model_state_dict'], strict=True)

    model.to(device)
    target_layers = [model.blocks[3][1]]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True,
                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))

    dir_total = data
    img_dirs = [os.path.join(dir_total, dir_path) for dir_path in os.listdir(dir_total)]
    for img_dir in img_dirs:
        img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
        for img_path in img_paths:
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            img = center_crop_img(img, img_size)

            # [C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)

            original_filename = os.path.basename(img_path)
            save_path = os.path.join(save_dir, original_filename)
            print(save_path)
            print('*')

            plt.imsave(save_path, visualization)






