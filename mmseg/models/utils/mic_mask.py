# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
from PIL import Image
import torch

from mmseg.ops import resize
from torchvision import transforms
from datetime import datetime


class BlockMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask < self.mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    mask_ratio, mask_block_size = 0.8, 4
    mic_mask = BlockMaskGenerator(mask_ratio, mask_block_size)
    path = r'D:\1-workplace\SePiCo\SePiCo-main\data\dark_zurich\rgb_anon\train\night\GOPR0351\GOPR0351_frame_000071_rgb_anon.png'
    img = Image.open(path)
    # img.convert('RGB')

    # Pil2Tensor = transforms.PILToTensor()
    # img_tensor = Pil2Tensor(img).to(device)

    img_tensor = transforms.ToTensor()(img).to(device)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    mask_image = mic_mask.mask_image(img_tensor)[0]

    img_RGB = transforms.ToPILImage()(mask_image)
    current_time = datetime.now().strftime("%d%H%M%S")
    img_RGB.save(f'mask_image{mask_ratio}-{mask_block_size}-{current_time}.png')
