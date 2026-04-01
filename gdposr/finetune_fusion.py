"""
finetune_fusion.py  —  Stage-1.5 IR-VIS 有监督微调
路径: /private/home/wuhao/dnj/GDPO-main/GDPOSR/train/finetune_fusion.py

伪GT = α*VIS + (1-α)*IR_gray3ch
  IR 直接用灰度三通道（不做 colormap），颜色中性，不引入额外色偏
  α=0.6：60% VIS 颜色 + 40% IR 亮度/热目标，模型被迫从 IR 提取热目标

启动命令:
  CUDA_VISIBLE_DEVICES=5,6,7 torchrun \
    --nproc_per_node=3 --master_port=29502 \
    train/finetune_fusion.py \
    --pretrained_model_name_or_path /path/to/sd1.5 \
    --pretrained_path experience/stage1_run3/checkpoints/model_030000.pkl \
    --learning_rate 1e-5 \
    --max_train_steps 3000 \
    --checkpointing_steps 500 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --mixed_precision fp16 \
    --use_vae_encode_lora --use_vae_decode_lora \
    --lora_rank_unet 8 --lora_rank_vae 4 \
    --time_step 999 --time_step_noise 250 \
    --report_to wandb \
    --tracker_project_name fusion_finetune \
    --output_dir experience/stage1_ft_run1 \
    --seed 123
"""

import os
import sys
import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import transformers
import diffusers
import wandb
import cv2

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler, DDIMScheduler
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

sys.path.append("/private/home/wuhao/dnj/GDPO-main/GDPOSR")
from modelfile.NAOSD import NAOSD
from my_utils.training_utils_realsr import parse_args_realsr_training


# ══════════════════════════════════════════════
# 伪GT：α*VIS + (1-α)*IR_gray3ch
# ══════════════════════════════════════════════

def make_pseudo_gt(ir_np, vis_np, alpha=0.6):
    """
    IR 转灰度后复制到三通道（颜色中性，不引入色偏）
    伪GT = α*VIS + (1-α)*IR_gray3ch
    """
    gray = cv2.cvtColor(ir_np, cv2.COLOR_RGB2GRAY)
    ir_gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)   # (H,W,3) 灰度三通道
    pseudo = alpha * vis_np.astype(np.float32) + \
             (1 - alpha) * ir_gray3.astype(np.float32)
    return np.clip(pseudo, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════

class IRVISDataset(Dataset):
    def __init__(self, ir_dir, vis_dir, image_size=512, alpha=0.6):
        super().__init__()
        self.alpha      = alpha
        self.image_size = image_size

        ir_list  = sorted(glob.glob(os.path.join(ir_dir,  "*.png")))
        vis_list = sorted(glob.glob(os.path.join(vis_dir, "*.png")))
        if not ir_list:
            ir_list  = sorted(glob.glob(os.path.join(ir_dir,  "*.jpg")))
            vis_list = sorted(glob.glob(os.path.join(vis_dir, "*.jpg")))

        assert len(ir_list) == len(vis_list) and len(ir_list) > 0, \
            f"IR/VIS 数量不一致或为空: {ir_dir}"

        self.ir_list  = ir_list
        self.vis_list = vis_list
        print(f"[IRVISDataset] {len(ir_list)} pairs, alpha={alpha}")

    def _to_tensor(self, pil_img):
        t = TF.to_tensor(pil_img)
        return TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.ir_list)

    def __getitem__(self, idx):
        ir_img  = Image.open(self.ir_list[idx]).convert("RGB")
        vis_img = Image.open(self.vis_list[idx]).convert("RGB")

        ir_img  = ir_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        vis_img = vis_img.resize((self.image_size, self.image_size), Image.LANCZOS)

        # 同步随机翻转，保持 IR/VIS 空间对齐
        if torch.rand(1).item() > 0.5:
            ir_img  = TF.hflip(ir_img)
            vis_img = TF.hflip(vis_img)

        ir_np  = np.array(ir_img)
        vis_np = np.array(vis_img)

        pseudo_gt = Image.fromarray(make_pseudo_gt(ir_np, vis_np, self.alpha))

        ir_t  = self._to_tensor(ir_img)
        vis_t = self._to_tensor(vis_img)
        gt_t  = self._to_tensor(pseudo_gt)

        return {
            "HR":   gt_t,
            "LR_A": ir_t,
            "LR_B": vis_t,
            "LR":   torch.cat([ir_t, vis_t], dim=0),
        }


# ══════════════════════════════════════════════
# 损失函数（与 run3 train_NAOSD.py 完全一致）
# ══════════════════════════════════════════════

def loss_diffusion(pred, target):
    return F.mse_loss(pred.float(), target.float())

def loss_pixel(pred, gt):
    return F.mse_loss(pred.float(), gt.float())

def loss_ssim(x_pred, x_gt):
    x_pred, x_gt = x_pred.float(), x_gt.float()
    C1, C2 = 0.01**2, 0.03**2
    mu_p = F.avg_pool2d(x_pred, 11, 1, 5)
    mu_g = F.avg_pool2d(x_gt,   11, 1, 5)
    mu_p2, mu_g2, mu_pg = mu_p*mu_p, mu_g*mu_g, mu_p*mu_g
    s_p2 = F.avg_pool2d(x_pred*x_pred, 11, 1, 5) - mu_p2
    s_g2 = F.avg_pool2d(x_gt  *x_gt,   11, 1, 5) - mu_g2
    s_pg = F.avg_pool2d(x_pred*x_gt,   11, 1, 5) - mu_pg
    ssim = ((2*mu_pg+C1)*(2*s_pg+C2)) / ((mu_p2+mu_g2+C1)*(s_p2+s_g2+C2))
    return 1.0 - ssim.mean()

class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.s1 = nn.Sequential(*list(vgg.features)[:4]).eval()
        self.s2 = nn.Sequential(*list(vgg.features)[4:9]).eval()
        self.s3 = nn.Sequential(*list(vgg.features)[9:16]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, pred, gt):
        def norm(x):
            return ((x.float()+1)/2 - self.mean) / self.std
        pred, gt = norm(pred), norm(gt)
        loss = 0.0
        for s in [self.s1, self.s2, self.s3]:
            pred = s(pred); gt = s(gt)
            loss += F.mse_loss(pred, gt)
        return loss

def loss_color(pred, gt):
    return (1.0 - F.cosine_similarity(
        F.normalize(pred.float(), p=2, dim=1),
        F.normalize(gt.float(),   p=2, dim=1), dim=1)).mean()

def predict_x0_from_noise(noise_pred, noisy_latent, alphas_cumprod, t):
    a_t = alphas_cumprod.to(noisy_latent.device)[t].view(-1,1,1,1)
    return ((noisy_latent - (1-a_t).sqrt() * noise_pred.float()) / a_t.sqrt()).clamp(-5, 5)


# ══════════════════════════════════════════════
# 验证推理（DDIM 20步，t_start=timesteps[0]）
# ══════════════════════════════════════════════

def run_val_fusion(net_unwrapped, step, output_dir, device,
                   val_ir_path, val_vis_path):
    if not (val_ir_path and val_vis_path):
        return
    if not (os.path.isfile(val_ir_path) and os.path.isfile(val_vis_path)):
        print(f"[val] 找不到验证图，跳过")
        return

    net_unwrapped.set_eval()
    try:
        with torch.no_grad():
            def load(path):
                img = Image.open(path).convert("RGB").resize((512,512), Image.LANCZOS)
                t = TF.normalize(TF.to_tensor(img), [0.5,0.5,0.5], [0.5,0.5,0.5])
                return t.unsqueeze(0).to(device)

            x_ir  = load(val_ir_path)
            x_vis = load(val_vis_path)

            ddim = DDIMScheduler.from_pretrained(
                net_unwrapped.args.pretrained_model_name_or_path,
                subfolder="scheduler")
            ddim.set_timesteps(20, device=device)

            latent = net_unwrapped.vae.encode(x_vis).latent_dist.sample()
            latent = latent * net_unwrapped.vae.config.scaling_factor
            noise  = torch.randn_like(latent)
            t_start = ddim.timesteps[0]
            latent  = ddim.add_noise(latent, noise,
                                     torch.tensor([t_start], device=device))

            caption = net_unwrapped.encode_prompt([""])
            fa  = net_unwrapped.semantic_encoder(x_ir)
            fb  = net_unwrapped.semantic_encoder(x_vis)
            f_f = net_unwrapped.content_encoder(x_ir, x_vis)

            for ts in ddim.timesteps:
                lh, lw = latent.shape[-2:]
                xa_lat = F.interpolate(x_ir,  (lh,lw), mode="bilinear", align_corners=False)
                xb_lat = F.interpolate(x_vis, (lh,lw), mode="bilinear", align_corners=False)
                unet_input = torch.cat([latent, xa_lat, xb_lat], dim=1)
                net_unwrapped.timesteps = torch.tensor([ts], device=device).long()
                noise_pred = net_unwrapped._unet_forward(unet_input, caption, fa, fb, f_f)
                latent = ddim.step(noise_pred, ts, latent).prev_sample

            output = net_unwrapped.vae.decode(
                latent / net_unwrapped.vae.config.scaling_factor
            ).sample.clamp(-1, 1)

            grid = torch.cat([x_ir.cpu(), x_vis.cpu(), output.cpu()], dim=-1)
            save_path = os.path.join(output_dir, "eval", f"fusion_step_{step:06d}.png")
            save_image((grid+1)/2, save_path)
            print(f"[val] saved → {save_path}")

    except Exception as e:
        print(f"[val] failed at step {step}: {e}")

    net_unwrapped.timesteps = torch.tensor(
        [net_unwrapped.args.time_step], device=device).long()
    net_unwrapped.set_train()


# ══════════════════════════════════════════════
# main
# ══════════════════════════════════════════════

def main(args):
    if args.report_to == "wandb":
        wandb.login()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # 验证图取 MSRS 第一张
    ir_files  = sorted(glob.glob("/private/home/wuhao/dnj/data/MSRS_detection/ir/*.png"))
    vis_files = sorted(glob.glob("/private/home/wuhao/dnj/data/MSRS_detection/vi/*.png"))
    VAL_IR_PATH  = ir_files[0]  if ir_files  else None
    VAL_VIS_PATH = vis_files[0] if vis_files else None

    # ── 模型 ────────────────────────────────────
    net = NAOSD(args=args)
    net.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.use_vae_encode_lora and args.use_vae_decode_lora:
        net.vae.set_adapter(["default_encoder", "default_decoder"])
    elif args.use_vae_encode_lora:
        net.vae.set_adapter(["default_encoder"])
    elif args.use_vae_decode_lora:
        net.vae.set_adapter(["default_decoder"])
    net.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    vgg_loss = VGGPerceptual().cuda()

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    layers_to_opt = [p for p in net.parameters() if p.requires_grad]
    print(f"[optimizer] trainable params: {sum(p.numel() for p in layers_to_opt):,}")

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=100 * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    dataset_train = IRVISDataset(
        ir_dir=args.ir_dir, vis_dir=args.vis_dir,
        image_size=512, alpha=args.pseudo_gt_alpha,
    )
    dl_train = DataLoader(
        dataset_train, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.dataloader_num_workers,
        pin_memory=True, drop_last=True,
    )

    net, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net, optimizer, dl_train, lr_scheduler)
    vgg_loss = accelerator.prepare(vgg_loss)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(range(args.max_train_steps), desc="FT Steps",
                        disable=not accelerator.is_local_main_process)

    alphas_cumprod = noise_scheduler.alphas_cumprod.cuda()

    lambda_diff = 1.0
    lambda_pix  = 0.05
    lambda_ssim = 0.05
    lambda_per  = 0.1
    lambda_col  = 1.0

    global_step = 0
    nan_count   = 0
    train_iter  = iter(dl_train)

    while global_step < args.max_train_steps:

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            batch = next(train_iter)

        with accelerator.accumulate(net):

            x_tgt      = batch["HR"]
            x_ir       = batch["LR_A"]
            x_vis_b    = batch["LR_B"]
            extra_cond = batch["LR"]
            B          = x_tgt.shape[0]

            t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (B,), device=x_tgt.device).long()

            net_unwrapped = accelerator.unwrap_model(net)
            with torch.no_grad():
                latent = net_unwrapped.vae.encode(x_tgt).latent_dist.sample()
                latent = latent * net_unwrapped.vae.config.scaling_factor

            noise        = torch.randn_like(latent)
            noisy_latent = noise_scheduler.add_noise(latent, noise, t)

            xa = extra_cond[:, :3]
            xb = extra_cond[:, 3:6]
            fa  = net_unwrapped.semantic_encoder(xa)
            fb  = net_unwrapped.semantic_encoder(xb)
            f_f = net_unwrapped.content_encoder(xa, xb)

            lh, lw = noisy_latent.shape[-2:]
            xa_lat = F.interpolate(xa, (lh,lw), mode="bilinear", align_corners=False)
            xb_lat = F.interpolate(xb, (lh,lw), mode="bilinear", align_corners=False)
            unet_input = torch.cat([noisy_latent, xa_lat, xb_lat], dim=1)

            caption_enc = net_unwrapped.encode_prompt([""] * B)

            orig_ts = net_unwrapped.timesteps
            net_unwrapped.timesteps = t
            noise_pred = net_unwrapped._unet_forward(unet_input, caption_enc, fa, fb, f_f)
            net_unwrapped.timesteps = orig_ts

            l_diff = loss_diffusion(noise_pred, noise)

            x0_lat = predict_x0_from_noise(noise_pred, noisy_latent, alphas_cumprod, t)
            with torch.no_grad():
                x0_pred = net_unwrapped.vae.decode(
                    x0_lat / net_unwrapped.vae.config.scaling_factor
                ).sample.clamp(-1, 1)

            l_pix  = loss_pixel(x0_pred, x_tgt)
            l_ssim = loss_ssim(x0_pred, x_tgt)
            l_per  = vgg_loss(x0_pred, x_tgt)
            l_col  = loss_color(x0_pred, x_tgt)

            loss = (lambda_diff * l_diff + lambda_pix  * l_pix  +
                    lambda_ssim * l_ssim + lambda_per  * l_per  +
                    lambda_col  * l_col)

            if not torch.isfinite(loss):
                nan_count += 1
                print(f"[step {global_step}] NaN [{nan_count}/50]")
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                if nan_count >= 50:
                    break
                continue

            nan_count = 0
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                logs = {
                    "loss":   loss.detach().item(),
                    "l_diff": l_diff.detach().item(),
                    "l_col":  l_col.detach().item(),
                    "lr":     lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(
                    loss=f"{loss.item():.3f}", col=f"{l_col.item():.3f}")
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:

                    # 训练预览图：IR | VIS | 伪GT | x̂0预测
                    train_vis = torch.cat([
                        x_ir[:1].detach().cpu().float(),
                        x_vis_b[:1].detach().cpu().float(),
                        x_tgt[:1].detach().cpu().float(),
                        x0_pred[:1].detach().cpu().float(),
                    ], dim=-1)
                    save_image(
                        (train_vis+1)/2,
                        os.path.join(args.output_dir, "eval",
                                     f"train_step_{global_step:06d}.png"))

                    run_val_fusion(
                        net_unwrapped=accelerator.unwrap_model(net),
                        step=global_step,
                        output_dir=args.output_dir,
                        device=accelerator.device,
                        val_ir_path=VAL_IR_PATH,
                        val_vis_path=VAL_VIS_PATH,
                    )

                    outf = os.path.join(args.output_dir, "checkpoints",
                                        f"model_{global_step:06d}.pkl")
                    accelerator.unwrap_model(net).save_model(outf)

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoints", "model_final.pkl")
        accelerator.unwrap_model(net).save_model(save_path)
        print("Saved →", save_path)

    accelerator.end_training()


if __name__ == "__main__":
    import argparse as _ap

    args = parse_args_realsr_training()

    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--ir_dir",  type=str,
                    default="/private/home/wuhao/dnj/data/MSRS_detection/ir")
    _p.add_argument("--vis_dir", type=str,
                    default="/private/home/wuhao/dnj/data/MSRS_detection/vi")
    _p.add_argument("--pseudo_gt_alpha", type=float, default=0.6)
    _extra, _ = _p.parse_known_args()

    args.ir_dir          = _extra.ir_dir
    args.vis_dir         = _extra.vis_dir
    args.pseudo_gt_alpha = _extra.pseudo_gt_alpha

    main(args)