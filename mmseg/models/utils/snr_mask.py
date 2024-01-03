import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from datetime import datetime


class SNRMaskGenerator:
    def __init__(self, mask_ratio=0., mask_block_size=0, mode=''):
        # 信噪比阈值; 信噪比越大越好, 过滤极小值
        self.mask_ratio = mask_ratio
        # 根据信噪比图，按块生成掩码
        self.mask_block_size = mask_block_size
        self.mode = mode

    @torch.no_grad()
    def generate_mask(self, imgs):
        imgs_denoise = []
        for img in imgs.cpu():
            img_nf = img.permute(1, 2, 0).numpy() * 255.0
            img_nf = cv2.blur(img_nf, (4, 4))
            img_nf = img_nf * 1.0 / 255.0
            img = torch.Tensor(img_nf).float().permute(2, 0, 1)
            img = torch.unsqueeze(img, 0)
            imgs_denoise.append(img)
        imgs_denoise = torch.cat(imgs_denoise, 0).to(imgs.device)

        topil = transforms.ToPILImage()
        current_time = datetime.now().strftime("%d-%H-%M-%S")

        grayscale = transforms.Grayscale(1)
        imgs_gray = grayscale(imgs)
        imgs_dn_gray = grayscale(imgs_denoise)
        noise = torch.abs(imgs_gray - imgs_dn_gray)
        mask = torch.div(imgs_dn_gray, noise + 0.0001)
        # img_dn_pil = topil(imgs_denoise[0])
        # img_dn_pil.save(f'img/img_dn{current_time}.png')
        # topil(imgs_gray[0]).save(f'img/img_gray{current_time}.png')
        # topil(imgs_dn_gray[0]).save(f'img/imgs_dn_gray{current_time}.png')
        # topil(mask[0]).save(f'img/mask{current_time}.png')

        # dark = imgs
        # dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        # light = imgs_denoise  # lj denoise
        # light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114  # 灰度化
        # noise = torch.abs(dark - light)
        #
        # mask = torch.div(light, noise + 0.0001)
        # 将snr值映射到[0,1]，归一化
        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)
        # batch_size, 1, h, w; [0,1]
        mask = torch.clamp(mask, min=0, max=1.0)

        # max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        # min = torch.min(mask.view(batch_size, -1), dim=1)[0]
        # print(f'max:{max}, min:{min}')
        # topil(mask[0]).save(f'img/mask_nor{current_time}.png')

        # 直接用信噪比图作为mask
        if self.mode == 'snr_mask':
            # mask = mask.float()
            snr_mask = (mask > self.mask_ratio).float()
            # snr_mask[snr_mask <= self.mask_ratio] = 0.0
            topil(snr_mask[0]).save(f'img/snrmask{current_time}.png')
            return snr_mask
        # snr按块生成mask
        elif self.mode == 'snr_block_mask':
            mask_unfold = F.unfold(mask, kernel_size=self.mask_block_size, dilation=1, stride=self.mask_block_size,
                                   padding=0)
            mask_unfold = mask_unfold.permute(0, 2, 1)
            mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
            mask_unfold = mask_unfold.repeat(1, self.mask_block_size**2, 1)
            # mask_unfold[mask_unfold <= self.mask_ratio] = 0.0
            mask_unfold = (mask_unfold > self.mask_ratio).float()
            snr_block_mask = F.fold(mask_unfold, output_size=(height, width), kernel_size=self.mask_block_size,
                                    dilation=1, stride=self.mask_block_size, padding=0)
            topil(snr_block_mask[0]).save(f'img/snrBmask{current_time}.png')
            return snr_block_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # snr_mask or snr_block_mask
    # mode = 'snr_mask'
    mode = 'snr_block_mask'
    mask_ratio, mask_block_size = 0.05, 16
    snr_mask = SNRMaskGenerator(mask_ratio, mask_block_size, mode)
    path = r'D:\1-workplace\SePiCo\SePiCo-main\data\dark_zurich\rgb_anon\train\night\GOPR0351\GOPR0351_frame_000071_rgb_anon.png'
    img = Image.open(path)
    # img.convert('RGB')

    # Pil2Tensor = transforms.PILToTensor()
    # img_tensor = Pil2Tensor(img).to(device)

    img_tensor = transforms.ToTensor()(img).to(device)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    mask_image = snr_mask.mask_image(img_tensor)[0]

    img_RGB = transforms.ToPILImage()(mask_image)
    current_time = datetime.now().strftime("%H-%M-%S")
    if mode == 'snr_block_mask':
        img_RGB.save(f'img/snr_block_mask_image-{mask_ratio}-{mask_block_size}-{current_time}.png')
    elif mode == 'snr_mask':
        img_RGB.save(f'img/snr_mask_image-thr{mask_ratio};{current_time}.png')
