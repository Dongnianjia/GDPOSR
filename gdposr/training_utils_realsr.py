import os
import cv2
import random
import argparse
import glob
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F

from my_utils.mask import create_complexity_matrix, binarize_complexity_matrix, extract_and_dilate_edges


def parse_args_realsr_training(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--groupsize", default=6, type=int)
    parser.add_argument("--time_min", default=150, type=int)
    parser.add_argument("--time_max", default=350, type=int)
    parser.add_argument("--updatestep", default=4000, type=int)
    parser.add_argument("--patchsize", default=125, type=int)
    parser.add_argument("--beta_dpo", default=0.25, type=float)
    parser.add_argument("--klloss", default=1.0, type=float)
    parser.add_argument("--grpoloss", default=1.0, type=float)
    parser.add_argument("--positive_prompt", type=str, default='')
    parser.add_argument("--negative_prompt", type=str, default='')
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--lambda_klloss", default=0.0, type=float)
    parser.add_argument("--min_dm_step_ratio", default=0.02, type=float)
    parser.add_argument("--max_dm_step_ratio", default=0.98, type=float)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)
    parser.add_argument("--cfg_csd", default=7.5, type=float)
    parser.add_argument("--snr_gamma_vsd", default=None)
    parser.add_argument("--lora_rank_unet_vsd", default=8, type=int)
    parser.add_argument("--pretrained_model_name_or_path_vsd", default='', type=str)
    parser.add_argument("--basemodel_path", default='', type=str)
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.2, type=float)
    parser.add_argument("--lambda_lpips", default=2, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--dataset_folder", default='', type=str)
    parser.add_argument("--testdataset_folder", default='', type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--null_text_ratio", default=1., type=float)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100)
    parser.add_argument("--viz_freq", type=int, default=100)
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo")
    parser.add_argument('--tiled_size', type=int, default=768)
    parser.add_argument('--tiled_overlap', type=int, default=256)
    parser.add_argument("--pretrained_model_name_or_path", default='', type=str)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--cliptextmodule", type=str, default=None)
    parser.add_argument("--upsampler", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_unet2", default=0, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--time_step", default=999, type=int)
    parser.add_argument("--time_step_noise", default=250, type=int)
    parser.add_argument("--pretrained_path", default=None, type=str)
    parser.add_argument("--pretrained_unet_path", default=None, type=str)
    parser.add_argument("--pretrained_vae_path", default=None, type=str)
    parser.add_argument("--stage2", default=None, type=str)
    parser.add_argument("--stage3", default=None, type=str)
    parser.add_argument("--output_dir", default='experience/OSSR_vaeEcLora_ntr1_vsd_ntr0_nostage_clip_test')
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--use_online_deg", action="store_true")
    parser.add_argument("--deg_file_path", default="params_pasd.yml", type=str)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--use_vae_encode_lora", action="store_true")
    parser.add_argument("--use_vae_decode_lora", action="store_true")
    parser.add_argument("--use_lr_999noise", action="store_true")
    parser.add_argument("--use_lr_concat_lr_999noise", action="store_true")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def build_transform(image_prep):
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([transforms.Resize((256, 256), interpolation=Image.LANCZOS)])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([transforms.Resize((512, 512), interpolation=Image.LANCZOS)])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    else:
        raise ValueError(f"Unsupported image_prep: {image_prep}")
    return T


# =========================
# Mask-DiFuser dual masking
# 严格对应原文 5 种 degradation，去掉 to_grayscale 和 ir_simulation
# 原因：灰度化会破坏 xA/xB 的 RGB 互补性，
#       导致 L_col 无法正确约束颜色一致性
# =========================

def random_choice(image):
    """5 种 degradation，与 Mask-DiFuser 原文完全一致"""

    def random_odd_number(low=5, high=35):
        number = random.randint(low, high)
        return number if number % 2 != 0 else number + 1

    def add_blur(image_):
        ksize = random_odd_number(3, 7)
        return cv2.GaussianBlur(image_, (ksize, ksize), 0)

    def add_noise(image_):
        std = random_odd_number(low=5, high=35)
        gauss = np.random.normal(0, std, image_.shape).astype('float32')
        return np.clip(image_.astype('float32') + gauss, 0, 255).astype(np.uint8)

    def add_snow(image_):
        noise = np.zeros_like(image_)
        for _ in range(1000):
            x = np.random.randint(0, image_.shape[1])
            y = np.random.randint(0, image_.shape[0])
            noise[y:y + 2, x:x + 2] = 255
        noise = cv2.GaussianBlur(noise, (5, 5), 0)
        return cv2.addWeighted(image_, 0.7, noise, 0.3, 0).astype(np.uint8)

    def degrade_frequency_domain(image_):
        if image_.ndim == 3:
            out = []
            for c in range(image_.shape[2]):
                channel = image_[:, :, c].astype(float)
                f = np.fft.fft2(channel)
                fshift = np.fft.fftshift(f)
                mag = cv2.GaussianBlur(np.abs(fshift), (0, 0), 0.5)
                phase = np.angle(fshift)
                for _ in range(5):
                    i1 = np.random.randint(0, phase.shape[0], 2)
                    i2 = np.random.randint(0, phase.shape[1], 2)
                    phase[i1[0], i2[0]], phase[i1[1], i2[1]] = \
                        phase[i1[1], i2[1]], phase[i1[0], i2[0]]
                recon = np.abs(np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * phase))))
                out.append(np.clip(recon, 0, 255).astype('uint8'))
            return np.stack(out, axis=2)
        else:
            f = np.fft.fft2(image_.astype(float))
            fshift = np.fft.fftshift(f)
            mag = cv2.GaussianBlur(np.abs(fshift), (0, 0), 0.5)
            phase = np.angle(fshift)
            for _ in range(5):
                i1 = np.random.randint(0, phase.shape[0], 2)
                i2 = np.random.randint(0, phase.shape[1], 2)
                phase[i1[0], i1[1]], phase[i2[0], i2[1]] = \
                    phase[i2[0], i2[1]], phase[i1[0], i1[1]]
            recon = np.abs(np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * phase))))
            return np.clip(recon, 0, 255).astype('uint8')

    def gamma_transform(image_):
        gamma = random.uniform(0.3, 3.0)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image_, table)

    # ← 严格只有这 5 种，与 Mask-DiFuser 原文对应
    operations = [
        add_blur,
        add_noise,
        add_snow,
        degrade_frequency_domain,
        gamma_transform,
    ]
    return random.choice(operations)(image)


def random_mask(pil_image, num_row_col=(8, 8), ratio=1.0, blur_ratio=0.1):
    """Mask-DiFuser dual masking scheme（Eq.2）"""
    image = np.array(pil_image)
    rows, cols = image.shape[:2]
    patch_h = rows // num_row_col[0]
    patch_w = cols // num_row_col[1]

    all_patches = [(i, j) for i in range(num_row_col[0])
                   for j in range(num_row_col[1])]
    selected = random.sample(all_patches, k=int(len(all_patches) * ratio))

    first  = random.sample(selected, k=int(len(selected) * 0.5))
    second = list(set(selected) - set(first))

    fp  = random.sample(first,  k=int(len(first)  * 0.9))   # first pixel-fill
    fd  = list(set(first)  - set(fp))                        # first deg
    sp  = random.sample(second, k=int(len(second) * 0.9))   # second pixel-fill
    sd  = list(set(second) - set(sp))                        # second deg

    # cross-blur: small fraction of the OTHER half gets degraded
    xd1 = random.sample(second, k=int(len(sp) * blur_ratio))
    xd2 = random.sample(first,  k=int(len(fp) * blur_ratio))

    pixel_val   = np.random.randint(0, 256)
    deg_img     = random_choice(image)

    img1 = image.copy()
    img2 = image.copy()

    def fill(img, pixel_ids, deg_ids, pixel):
        for i, j in pixel_ids:
            img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = pixel
        for i, j in deg_ids:
            img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = \
                deg_img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
        return img

    img1 = fill(img1, fp, fd + xd1, pixel_val)
    img2 = fill(img2, sp, sd + xd2, pixel_val)

    return Image.fromarray(img1.astype(np.uint8)), \
           Image.fromarray(img2.astype(np.uint8))


class PairedSROnlineDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep,
                 deg_file_path=None, image_size=512, args=None):
        super().__init__()
        self.split = split
        self.args  = args

        if split == 'train':
            gt_folder = os.path.join(dataset_folder, "train")
            self.gt_list = sorted(
                glob.glob(os.path.join(gt_folder, '*.png')) +
                glob.glob(os.path.join(gt_folder, '*.jpg')) +
                glob.glob(os.path.join(gt_folder, '*.jpeg'))
            )
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'test':
            dataset_folder = args.testdataset_folder
            self.lr_list = sorted(glob.glob(
                os.path.join(dataset_folder, "test_SR_bicubic", '*.png')))
            self.gt_list = sorted(glob.glob(
                os.path.join(dataset_folder, "test_HR", '*.png')))
            self.T = build_transform(image_prep)
            assert len(self.lr_list) == len(self.gt_list)
        else:
            raise ValueError(f"Unsupported split: {split}")

    def __len__(self):
        return len(self.gt_list)

    @staticmethod
    def _to_tensor(pil_img):
        t = F.to_tensor(pil_img)
        return F.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __getitem__(self, idx):
        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            # Mask-DiFuser dual masking
            xA_pil, xB_pil = random_mask(gt_img, num_row_col=(8, 8),
                                          ratio=1.0, blur_ratio=0.1)
            hr_t  = self._to_tensor(gt_img)
            xA_t  = self._to_tensor(xA_pil)
            xB_t  = self._to_tensor(xB_pil)
            lr_cat = torch.cat([xA_t, xB_t], dim=0)   # 6ch for compat

            # complexity map（保持原逻辑，GDPO stage2 可能用到）
            gt_np  = np.array(gt_img)
            gray   = cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY)
            cmap   = create_complexity_matrix(gray / 255.0, patch_size=10)
            bmap, fid_ratio, det_ratio = binarize_complexity_matrix(cmap, threshold=50)
            edges  = extract_and_dilate_edges(gray, 100, 200, 3, 8)

            return {
                "HR":     hr_t,
                "LR_A":  xA_t,
                "LR_B":  xB_t,
                "LR":    lr_cat,
                "negative_prompt": self.args.negative_prompt,
                "fedilty_ratio":   torch.tensor(fid_ratio),
                "detail_ratio":    torch.tensor(det_ratio),
                "complexity_matrix":     torch.tensor(cmap).unsqueeze(0),
                "binary_matrix":         torch.tensor(bmap).unsqueeze(0),
                "downsampled_edges_mask":torch.tensor(edges),
                "base_name": os.path.basename(self.gt_list[idx]),
            }

        else:  # test
            input_img  = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_t  = F.normalize(F.to_tensor(self.T(input_img)),  [0.5]*3, [0.5]*3)
            out_t  = F.normalize(F.to_tensor(self.T(output_img)), [0.5]*3, [0.5]*3)
            return {
                "HR": out_t,
                "LR": img_t,
                "negative_prompt": self.args.negative_prompt,
                "base_name": os.path.basename(self.lr_list[idx]),
            }
