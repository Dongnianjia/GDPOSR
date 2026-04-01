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
# from datasets.realesrgan import RealESRGAN_degradation


def parse_args_realsr_training(input_args=None):
    """
    Parses command-line arguments used for configuring a paired session (pix2pix-Turbo).
    """
    parser = argparse.ArgumentParser()

    # args for grpo training
    parser.add_argument("--groupsize", default=6, type=int)
    parser.add_argument("--time_min", default=150, type=int)
    parser.add_argument("--time_max", default=350, type=int)
    parser.add_argument("--updatestep", default=4000, type=int)
    parser.add_argument("--patchsize", default=125, type=int)
    parser.add_argument("--beta_dpo", default=0.25, type=float)
    parser.add_argument("--klloss", default=1.0, type=float)
    parser.add_argument("--grpoloss", default=1.0, type=float)

    # args for the vsd training
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

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.2, type=float)
    parser.add_argument("--lambda_lpips", default=2, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", default='', type=str)
    parser.add_argument("--testdataset_folder", default='', type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--null_text_ratio", default=1., type=float)

    # validation eval args
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")
    parser.add_argument('--tiled_size', type=int, default=768)
    parser.add_argument('--tiled_overlap', type=int, default=256)

    # details about the model architecture
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

    # training details
    parser.add_argument("--output_dir", default='experience/OSSR_vaeEcLora_ntr1_vsd_ntr0_nostage_clip_test')
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate for model parameters.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`.',
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true")

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--use_online_deg", action="store_true")
    parser.add_argument("--deg_file_path", default="params_pasd.yml", type=str)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')

    # vae lora
    parser.add_argument("--use_vae_encode_lora", action="store_true")
    parser.add_argument("--use_vae_decode_lora", action="store_true")

    # use_lr_999noise
    parser.add_argument("--use_lr_999noise", action="store_true")
    parser.add_argument("--use_lr_concat_lr_999noise", action="store_true")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.
    """
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
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    else:
        raise ValueError(f"Unsupported image_prep: {image_prep}")
    return T


# =========================
# Mask-DiFuser dual masking
# =========================

def random_choice(image):
    def random_odd_number(low=5, high=35):
        number = random.randint(low, high)
        return number if number % 2 != 0 else number + 1

    def add_blur(image_):
        ksize = random_odd_number(3, 7)
        return cv2.GaussianBlur(image_, (ksize, ksize), 0)

    def add_noise(image_, noise_type='gaussian'):
        if noise_type == 'gaussian':
            mean = 0
            std = random_odd_number(low=5, high=35)
            gauss = np.random.normal(mean, std, image_.shape).astype('float32')
            noisy_image = image_ + gauss
            noisy_image = np.clip(noisy_image, 0, 255)
            return noisy_image.astype(np.uint8)
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    def add_rain_snow(image_, effect='snow'):
        noise = np.zeros_like(image_)
        if effect == 'rain':
            num_drops = 1000
            for _ in range(num_drops):
                x = np.random.randint(0, image_.shape[1])
                y = np.random.randint(0, image_.shape[0])
                noise[y:y + 5, x:x + 2] = 255
            noise = cv2.blur(noise, (5, 5))
            image_out = cv2.addWeighted(image_, 0.8, noise, 0.2, 0)
        elif effect == 'snow':
            num_flakes = 1000
            for _ in range(num_flakes):
                x = np.random.randint(0, image_.shape[1])
                y = np.random.randint(0, image_.shape[0])
                noise[y:y + 2, x:x + 2] = 255
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
            image_out = cv2.addWeighted(image_, 0.7, noise, 0.3, 0)
        else:
            raise ValueError(f"Unsupported effect: {effect}")
        return image_out.astype(np.uint8)

    def degrade_frequency_domain(image_, np_random=5):
        if image_.ndim == 3:
            out = []
            for c in range(image_.shape[2]):
                channel = image_[:, :, c]
                f = np.fft.fft2(channel)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = np.abs(fshift)
                phase_spectrum = np.angle(fshift)

                magnitude_spectrum = cv2.GaussianBlur(magnitude_spectrum, (0, 0), 0.5)
                for _ in range(np_random):
                    idx1 = np.random.randint(0, phase_spectrum.shape[0], 2)
                    idx2 = np.random.randint(0, phase_spectrum.shape[1], 2)
                    phase_spectrum[idx1[0], idx2[0]], phase_spectrum[idx1[1], idx2[1]] = \
                        phase_spectrum[idx1[1], idx2[1]], phase_spectrum[idx1[0], idx2[0]]

                fshift_new = magnitude_spectrum * np.exp(1j * phase_spectrum)
                f_ishift = np.fft.ifftshift(fshift_new)
                image_back = np.fft.ifft2(f_ishift)
                image_back = np.abs(image_back)
                out.append(np.clip(image_back, 0, 255).astype('uint8'))
            return np.stack(out, axis=2)
        else:
            f = np.fft.fft2(image_)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            phase_spectrum = np.angle(fshift)
            magnitude_spectrum = cv2.GaussianBlur(magnitude_spectrum, (0, 0), 0.5)
            for _ in range(np_random):
                idx1 = np.random.randint(0, phase_spectrum.shape[0], 2)
                idx2 = np.random.randint(0, phase_spectrum.shape[1], 2)
                phase_spectrum[idx1[0], idx1[1]], phase_spectrum[idx2[0], idx2[1]] = \
                    phase_spectrum[idx2[0], idx2[1]], phase_spectrum[idx1[0], idx1[1]]
            fshift_new = magnitude_spectrum * np.exp(1j * phase_spectrum)
            f_ishift = np.fft.ifftshift(fshift_new)
            image_back = np.fft.ifft2(f_ishift)
            image_back = np.abs(image_back)
            return np.clip(image_back, 0, 255).astype('uint8')

    def gamma_transform(image_, gamma_range=(0.3, 3.0)):
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image_, table)

    def to_grayscale(image_):
        """模拟 IR 模态：转灰度后复制到三通道"""
        gray = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def ir_simulation(image_):
        """更完整的 IR 模拟：灰度化 + 对比度增强"""
        gray = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)
        # CLAHE 对比度增强，模拟热成像的高对比特性
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    operations = [
        lambda img: add_blur(img),
        lambda img: add_noise(img, noise_type='gaussian'),
        lambda img: add_rain_snow(img, effect='snow'),
        lambda img: degrade_frequency_domain(img, np_random=5),
        lambda img: gamma_transform(img, gamma_range=(0.3, 3.0)),
        lambda img: to_grayscale(img),      # ← 新增
        lambda img: ir_simulation(img),     # ← 新增
    ]

    operation = random.choice(operations)
    return operation(image)


def random_mask(pil_image, num_row_col=(8, 8), ratio=1.0, blur_ratio=0.1):
    """
    Directly adapted from Mask-DiFuser get_data.py logic.
    Input: PIL RGB image
    Output: two complementary masked PIL RGB images
    """
    image = np.array(pil_image)
    rows, cols, channels = image.shape

    patch_size_row = rows // num_row_col[0]
    patch_size_col = cols // num_row_col[1]

    patches_indices = [(i, j) for i in range(num_row_col[0]) for j in range(num_row_col[1])]
    selected_indices = random.sample(patches_indices, k=int(len(patches_indices) * ratio))

    first_mask_indices = random.sample(selected_indices, k=int(len(selected_indices) * 0.5))
    first_pixel_ids = random.sample(first_mask_indices, k=int(len(first_mask_indices) * 0.9))
    first_deg_ids = list(set(first_mask_indices) - set(first_pixel_ids))

    second_mask_indices = list(set(selected_indices) - set(first_mask_indices))
    second_pixel_ids = random.sample(second_mask_indices, k=int(len(second_mask_indices) * 0.9))
    second_deg_ids = list(set(second_mask_indices) - set(second_pixel_ids))

    deg_ids1 = random.sample(second_mask_indices, k=int(len(second_pixel_ids) * blur_ratio))
    deg_ids2 = random.sample(first_mask_indices, k=int(len(first_pixel_ids) * blur_ratio))

    image_1 = image.copy()
    image_2 = image.copy()
    deg_choice_img = random_choice(image)
    pixel_level = np.random.randint(0, 256)

    def process_blocks(target_image, pixel_ids, p_ids, deg_ids, pixel):
        for idx in pixel_ids:
            i, j = idx
            target_image[
                i * patch_size_row:(i + 1) * patch_size_row,
                j * patch_size_col:(j + 1) * patch_size_col,
                :
            ] = pixel

        for idx in p_ids:
            i, j = idx
            target_image[
                i * patch_size_row:(i + 1) * patch_size_row,
                j * patch_size_col:(j + 1) * patch_size_col,
                :
            ] = deg_choice_img[
                i * patch_size_row:(i + 1) * patch_size_row,
                j * patch_size_col:(j + 1) * patch_size_col,
                :
            ]

        for idx in deg_ids:
            i, j = idx
            target_image[
                i * patch_size_row:(i + 1) * patch_size_row,
                j * patch_size_col:(j + 1) * patch_size_col,
                :
            ] = deg_choice_img[
                i * patch_size_row:(i + 1) * patch_size_row,
                j * patch_size_col:(j + 1) * patch_size_col,
                :
            ]
        return target_image

    image_1 = process_blocks(image_1, first_pixel_ids, first_deg_ids, deg_ids1, pixel=pixel_level)
    image_2 = process_blocks(image_2, second_pixel_ids, second_deg_ids, deg_ids2, pixel=pixel_level)

    pil_image_1 = Image.fromarray(image_1.astype(np.uint8))
    pil_image_2 = Image.fromarray(image_2.astype(np.uint8))
    return pil_image_1, pil_image_2


class PairedSROnlineDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, deg_file_path=None, image_size=512, args=None):
        super().__init__()
        self.split = split
        self.args = args

        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        self.clip_normalize = transforms.Normalize(mean=clip_mean, std=clip_std)

        if split == 'train':
            self.gt_folder = os.path.join(dataset_folder, "train")
            self.gt_list = []
            self.gt_list += glob.glob(os.path.join(self.gt_folder, '*.png'))
            self.gt_list += glob.glob(os.path.join(self.gt_folder, '*.jpg'))
            self.gt_list += glob.glob(os.path.join(self.gt_folder, '*.jpeg'))
            self.gt_list = sorted(self.gt_list)

            self.T = build_transform(image_prep)
            self.split = split

            # self.degradation = RealESRGAN_degradation(deg_file_path, device='cpu')

            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ])

        elif split == 'test':
            dataset_folder = args.testdataset_folder
            self.input_folder = os.path.join(dataset_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(dataset_folder, "test_HR")

            self.lr_list = []
            self.gt_list = []
            self.lr_list += glob.glob(os.path.join(self.input_folder, '*.png'))
            self.gt_list += glob.glob(os.path.join(self.output_folder, '*.png'))

            self.lr_list = sorted(self.lr_list)
            self.gt_list = sorted(self.gt_list)

            self.T = build_transform(image_prep)
            self.split = split
            assert len(self.lr_list) == len(self.gt_list)

        else:
            raise ValueError(f"Unsupported split: {split}")

    def __len__(self):
        return len(self.gt_list)

    def _to_normalized_tensor(self, pil_img):
        tensor = F.to_tensor(pil_img)
        tensor = F.normalize(tensor, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        return tensor

    def __getitem__(self, idx):
        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            # Mask-DiFuser dual masking
            deg1_img, deg2_img = random_mask(gt_img, num_row_col=(8, 8), ratio=1.0, blur_ratio=0.1)

            # to tensor, scaled to [-1, 1]
            hr_t = self._to_normalized_tensor(gt_img)
            lr_a_t = self._to_normalized_tensor(deg1_img)
            lr_b_t = self._to_normalized_tensor(deg2_img)

            # route B: directly prepare 6-channel input
            lr_cat_t = torch.cat([lr_a_t, lr_b_t], dim=0)

            # keep your original ratio estimation logic, but now compute from GT
            gt_np = np.array(gt_img)
            gray_gt_img_org = cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY)
            gray_gt_img = gray_gt_img_org / 255.0

            complexity_matrix = create_complexity_matrix(gray_gt_img, patch_size=10)
            binary_matrix, fedilty_zero_ratio, detail_one_ratio = binarize_complexity_matrix(
                complexity_matrix, threshold=50
            )
            downsampled_edges_mask = extract_and_dilate_edges(
                gray_gt_img_org, threshold1=100, threshold2=200, dilation_size=3, downscale_factor=8
            )

            complexity_matrix = torch.tensor(complexity_matrix).unsqueeze(0)
            binary_matrix = torch.tensor(binary_matrix).unsqueeze(0)
            fedilty_zero_ratio = torch.tensor(fedilty_zero_ratio)
            detail_one_ratio = torch.tensor(detail_one_ratio)
            downsampled_edges_mask = torch.tensor(downsampled_edges_mask)

            return {
                "HR": hr_t,                        # 3ch GT
                "LR_A": lr_a_t,                   # 3ch masked input A
                "LR_B": lr_b_t,                   # 3ch masked input B
                "LR": lr_cat_t,                   # 6ch concat input for route B
                "negative_prompt": self.args.negative_prompt,
                "fedilty_ratio": fedilty_zero_ratio,
                "detail_ratio": detail_one_ratio,
                "complexity_matrix": complexity_matrix,
                "binary_matrix": binary_matrix,
                "downsampled_edges_mask": downsampled_edges_mask,
                "base_name": os.path.basename(self.gt_list[idx]),
            }

        elif self.split == 'test':
            # 这里先保留原 SR 风格测试逻辑，避免你现有评测脚本一起炸
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            input_img_noresize = Image.open(self.gt_list[idx].replace('test_HR/', 'test_LR/')).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')

            img_t = self.T(input_img)
            img_t = F.to_tensor(img_t)

            img_t_noresize = self.T(input_img_noresize)
            img_t_noresize = F.to_tensor(img_t_noresize)

            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            img_t_noresize = F.normalize(img_t_noresize, mean=[0.5], std=[0.5])

            output_t = self.T(output_img)
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            return {
                "HR": output_t,
                "LR": img_t,
                "negative_prompt": self.args.negative_prompt,
                "base_name": os.path.basename(self.lr_list[idx]),
            }