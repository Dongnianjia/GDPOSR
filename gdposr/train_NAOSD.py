"""
train_NAOSD.py  —  Stage-1 训练脚本（Mask-DiFuser 损失版）
路径: /private/home/wuhao/dnj/GDPO-main/GDPOSR/train/train_NAOSD.py

损失函数改为 Mask-DiFuser 论文公式(18)的5项损失：
  L = L_diff + λ1*L_pix + λ2*L_ssim + λ3*L_per + λ4*L_col
  λ1=0.05, λ2=0.05, λ3=0.1, λ4=1.0

其中 L_diff 是最关键的改动：
  原来：固定 t=999，只做 one-step 监督
  现在：随机采样 t∈[0,T]，在所有时间步上监督噪声预测
  这让模型学到完整的扩散生成先验，而不只是 t=999 这一刀

模型结构（NAOSD.py）不需要改动。
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import transformers
import diffusers
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

sys.path.append("/private/home/wuhao/dnj/GDPO-main/GDPOSR")

from modelfile.NAOSD import NAOSD
from my_utils.training_utils_realsr import parse_args_realsr_training, PairedSROnlineDataset

from diffusers import DDIMScheduler


# ══════════════════════════════════════════════
# Mask-DiFuser 损失函数（直接复制，不依赖外部导入）
# ══════════════════════════════════════════════

def loss_diffusion(noise_pred, noise_target):
    """
    L_diff：扩散去噪损失，公式(12)
    预测噪声和真实噪声的 L2 距离
    """
    return F.mse_loss(noise_pred.float(), noise_target.float())


def loss_pixel(x_pred, x_gt):
    """
    L_pix：像素 L2 损失，公式(14)
    在像素空间约束整体数值接近 GT
    """
    return F.mse_loss(x_pred.float(), x_gt.float())


def loss_ssim(x_pred, x_gt):
    """
    L_ssim：结构相似性损失，公式(15)
    约束局部均值、方差、协方差，保持结构
    """
    x_pred = x_pred.float()
    x_gt   = x_gt.float()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_p  = F.avg_pool2d(x_pred, 11, 1, 5)
    mu_g  = F.avg_pool2d(x_gt,   11, 1, 5)

    mu_p2 = mu_p * mu_p
    mu_g2 = mu_g * mu_g
    mu_pg = mu_p * mu_g

    sig_p2  = F.avg_pool2d(x_pred * x_pred, 11, 1, 5) - mu_p2
    sig_g2  = F.avg_pool2d(x_gt   * x_gt,   11, 1, 5) - mu_g2
    sig_pg  = F.avg_pool2d(x_pred * x_gt,   11, 1, 5) - mu_pg

    ssim_map = ((2 * mu_pg + C1) * (2 * sig_pg + C2)) / \
               ((mu_p2 + mu_g2 + C1) * (sig_p2 + sig_g2 + C2))

    return 1.0 - ssim_map.mean()


class VGGPerceptual(nn.Module):
    """
    L_per：VGG 感知损失，公式(16)
    在 VGG 多层特征空间约束感知相似性
    注：lpips 库也可以做同样的事，这里直接用 torchvision VGG 保持与 Mask-DiFuser 一致
    """
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        # 取 relu1_2, relu2_2, relu3_3 三层，与论文 Section III-B.3 一致
        self.slice1 = nn.Sequential(*list(vgg.features)[:4]).eval()
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9]).eval()
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16]).eval()
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet 归一化
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x_pred, x_gt):
        # [-1,1] → [0,1] → ImageNet normalize
        x_pred = (x_pred.float() + 1) / 2
        x_gt   = (x_gt.float()   + 1) / 2
        x_pred = (x_pred - self.mean) / self.std
        x_gt   = (x_gt   - self.mean) / self.std

        loss = 0.0
        for s in [self.slice1, self.slice2, self.slice3]:
            x_pred = s(x_pred)
            x_gt   = s(x_gt)
            loss  += F.mse_loss(x_pred, x_gt)
        return loss


def loss_color(x_pred, x_gt):
    """
    L_col：颜色一致性损失，公式(17)
    最小化预测图和 GT 的 RGB 颜色向量夹角
    权重最大（λ4=1.0），直接抑制颜色偏移和色彩伪影
    """
    # L2 normalize along channel dim
    pred_norm = F.normalize(x_pred.float(), p=2, dim=1)
    gt_norm   = F.normalize(x_gt.float(),   p=2, dim=1)
    # 1 - cosine_similarity → 角度越大，loss 越大
    return (1.0 - F.cosine_similarity(pred_norm, gt_norm, dim=1)).mean()


def predict_x0_from_noise(noise_pred, noisy_latent, alphas_cumprod, t):
    acp = alphas_cumprod.to(noisy_latent.device)
    a_t = acp[t].view(-1, 1, 1, 1)
    x0_pred = (noisy_latent - (1 - a_t).sqrt() * noise_pred.float()) / a_t.sqrt()
    return x0_pred.clamp(-5, 5) 


# ══════════════════════════════════════════════
# 验证推理
# ══════════════════════════════════════════════
def run_val_fusion(net_unwrapped, step, output_dir, device,
                   val_ir_path, val_vis_path):
    if val_ir_path is None or val_vis_path is None:
        return
    if not (os.path.isfile(val_ir_path) and os.path.isfile(val_vis_path)):
        return

    net_unwrapped.set_eval()
    try:
        with torch.no_grad():
            def load_img(path):
                img = Image.open(path).convert("RGB").resize(
                    (512, 512), Image.LANCZOS)
                t = TF.to_tensor(img)
                t = TF.normalize(t, [0.5,0.5,0.5], [0.5,0.5,0.5])
                return t.unsqueeze(0).to(device)

            x_ir  = load_img(val_ir_path).half()
            x_vis = load_img(val_vis_path).half()

            # DDIM 5步推理
            ddim = DDIMScheduler.from_pretrained(
                net_unwrapped.args.pretrained_model_name_or_path,
                subfolder="scheduler"
            )
            ddim.set_timesteps(5, device=device)

            # 从 VIS latent 加噪到最大时间步出发
            latent = net_unwrapped.vae.encode(x_vis).latent_dist.sample()
            latent = latent * net_unwrapped.vae.config.scaling_factor
            noise  = torch.randn_like(latent)
            latent = ddim.add_noise(latent, noise, torch.tensor([999], device=device))

            caption = net_unwrapped.encode_prompt([""])
            fa  = net_unwrapped.semantic_encoder(x_ir)
            fb  = net_unwrapped.semantic_encoder(x_vis)
            f_f = net_unwrapped.content_encoder(x_ir, x_vis)

            for ts in ddim.timesteps:
                lh, lw = latent.shape[-2:]
                xa_lat = F.interpolate(x_ir,  size=(lh,lw),
                    mode="bilinear", align_corners=False)
                xb_lat = F.interpolate(x_vis, size=(lh,lw),
                    mode="bilinear", align_corners=False)
                unet_input = torch.cat([latent, xa_lat, xb_lat], dim=1)

                # 临时覆盖 timesteps
                net_unwrapped.timesteps = torch.tensor(
                    [ts], device=device).long()
                noise_pred = net_unwrapped._unet_forward(
                    unet_input, caption, fa, fb, f_f)

                latent = ddim.step(noise_pred, ts, latent).prev_sample

            output = net_unwrapped.vae.decode(
                latent / net_unwrapped.vae.config.scaling_factor
            ).sample.clamp(-1, 1)

            grid = torch.cat(
                [x_ir.cpu(), x_vis.cpu(), output.cpu()], dim=-1)
            save_path = os.path.join(
                output_dir, "eval", f"fusion_step_{step:06d}.png")
            save_image((grid + 1) / 2, save_path)
            print(f"[val] saved → {save_path}")

    except Exception as e:
        print(f"[val] failed at step {step}: {e}")

    # 恢复原始 timesteps
    net_unwrapped.timesteps = torch.tensor(
        [net_unwrapped.args.time_step],
        device=device
    ).long()
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

    # ── 验证图路径 ──────────────────────────────
    VAL_IR_PATH  = "/private/home/wuhao/dnj/data/M3FD/Ir/00000.png"
    VAL_VIS_PATH = "/private/home/wuhao/dnj/data/M3FD/Vis/00000.png"

    # ── 模型 ────────────────────────────────────
    net = NAOSD(args=args)
    net.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available")

    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── VAE adapter 激活 ────────────────────────
    if args.use_vae_encode_lora and args.use_vae_decode_lora:
        print("==== LoRA at VAE Encoder & Decoder ====")
        net.vae.set_adapter(["default_encoder", "default_decoder"])
    elif args.use_vae_encode_lora:
        net.vae.set_adapter(["default_encoder"])
    elif args.use_vae_decode_lora:
        net.vae.set_adapter(["default_decoder"])
    else:
        if hasattr(net.vae, "peft_config") and net.vae.peft_config:
            net.vae.disable_adapters()

    net.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    # ── 感知损失模块（VGG，frozen）──────────────
    vgg_loss = VGGPerceptual().cuda()

    # ── DDPMScheduler（用于随机时间步训练）───────
    # 关键：用全时间步的 scheduler 而不是 one-step scheduler
    # 每个 step 随机采样 t，让模型在所有噪声级别上受到监督
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    # ── optimizer ───────────────────────────────
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
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ── dataset ─────────────────────────────────
    dataset_train = PairedSROnlineDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_image_prep,
        split="train",
        deg_file_path=args.deg_file_path,
        args=args,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.dataloader_num_workers > 0),
    )

    # ── accelerator prepare ─────────────────────
    net, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net, optimizer, dl_train, lr_scheduler
    )
    vgg_loss = accelerator.prepare(vgg_loss)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # alphas_cumprod 用于从噪声预测反推 x̂0
    alphas_cumprod = noise_scheduler.alphas_cumprod.cuda()

    # ── 损失权重（对应论文 λ1~λ4）──────────────
    lambda_diff = 1.0    # L_diff 权重（最主要的扩散损失）
    lambda_pix  = 0.05   # L_pix
    lambda_ssim = 0.05   # L_ssim
    lambda_per  = 0.1    # L_per
    lambda_col  = 1.0    # L_col（颜色损失，权重最大，直接解决颜色偏移）

    # ── training loop ───────────────────────────
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

            x_tgt      = batch["HR"]    # (B,3,H,W) GT，[-1,1]
            extra_cond = batch["LR"]    # (B,6,H,W) = [x_A | x_B]
            B          = x_tgt.shape[0]

            positive_prompt = [""] * B
            negative_prompt = [""] * B

            # ── 随机采样时间步（Mask-DiFuser 的核心训练方式）──
            # 不再固定 t=999，而是对每个样本随机采样 t∈[0,T-1]
            t = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=x_tgt.device,
            ).long()

            # ── VAE encode GT → latent → 加噪 ───────────────
            net_unwrapped = accelerator.unwrap_model(net)
            with torch.no_grad():
                latent = net_unwrapped.vae.encode(x_tgt).latent_dist.sample()
                latent = latent * net_unwrapped.vae.config.scaling_factor

            noise        = torch.randn_like(latent)
            noisy_latent = noise_scheduler.add_noise(latent, noise, t)

            # ── 条件准备 ─────────────────────────────────────
            xa = extra_cond[:, :3]
            xb = extra_cond[:, 3:6]
            fa  = net_unwrapped.semantic_encoder(xa)
            fb  = net_unwrapped.semantic_encoder(xb)
            f_f = net_unwrapped.content_encoder(xa, xb)

            lh, lw = noisy_latent.shape[-2:]
            xa_lat = F.interpolate(xa, size=(lh, lw), mode="bilinear", align_corners=False)
            xb_lat = F.interpolate(xb, size=(lh, lw), mode="bilinear", align_corners=False)
            unet_input = torch.cat([noisy_latent, xa_lat, xb_lat], dim=1)

            caption_enc = net_unwrapped.encode_prompt(positive_prompt)

            # ── UNet 前向：预测噪声 ──────────────────────────
            # 注意：这里直接调用 _unet_forward，但 timesteps 用随机采样的 t
            # 需要临时覆盖 timesteps
            original_timesteps = net_unwrapped.timesteps
            net_unwrapped.timesteps = t
            noise_pred = net_unwrapped._unet_forward(
                unet_input, caption_enc, fa, fb, f_f)
            net_unwrapped.timesteps = original_timesteps

            # ── L_diff：噪声预测损失 ─────────────────────────
            l_diff = loss_diffusion(noise_pred, noise)

            # ── 从噪声预测反推 x̂0（在 latent 空间）────────────
            x0_latent_pred = predict_x0_from_noise(
                noise_pred, noisy_latent, alphas_cumprod, t)

            # ── VAE decode → 像素空间 ────────────────────────
            with torch.no_grad():
                x0_pred = net_unwrapped.vae.decode(
                    x0_latent_pred / net_unwrapped.vae.config.scaling_factor
                ).sample.clamp(-1, 1)

            # ── 像素空间损失 ─────────────────────────────────
            l_pix  = loss_pixel(x0_pred, x_tgt)
            l_ssim = loss_ssim(x0_pred, x_tgt)
            l_per  = vgg_loss(x0_pred, x_tgt)
            l_col  = loss_color(x0_pred, x_tgt)

            # ── 总损失（公式18）─────────────────────────────
            loss = (lambda_diff * l_diff +
                    lambda_pix  * l_pix  +
                    lambda_ssim * l_ssim +
                    lambda_per  * l_per  +
                    lambda_col  * l_col)

            # ── NaN 防护 ─────────────────────────────────────
            if not torch.isfinite(loss):
                nan_count += 1
                print(f"[step {global_step}] non-finite loss, skipping. [{nan_count}/50]")
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                if nan_count >= 50:
                    print("Too many non-finite losses, aborting.")
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
                    "loss":       loss.detach().item(),
                    "loss_diff":  l_diff.detach().item(),
                    "loss_pix":   l_pix.detach().item(),
                    "loss_ssim":  l_ssim.detach().item(),
                    "loss_per":   l_per.detach().item(),
                    "loss_col":   l_col.detach().item(),
                    "lr":         lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    diff=f"{l_diff.item():.3f}",
                    col=f"{l_col.item():.3f}",
                )
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    # 训练预览图
                    adapter_w = accelerator.unwrap_model(net).cond_adapter.fuse[4][-1].weight.abs().mean().item()
                    print(f"[step {global_step}] cond_adapter fuse[4] weight mean: {adapter_w:.6f}")
                    accelerator.log({"adapter_w": adapter_w}, step=global_step)
                    train_vis = torch.cat([
                        x_tgt[:1].detach().cpu().float(),
                        x0_pred[:1].detach().cpu().float(),
                    ], dim=-1)
                    save_image(
                        (train_vis + 1) / 2,
                        os.path.join(args.output_dir, "eval",
                                    f"train_step_{global_step:06d}.png"),
                    )
                    # IR-VIS 融合验证图
                    run_val_fusion(
                        net_unwrapped=accelerator.unwrap_model(net),
                        step=global_step,
                        output_dir=args.output_dir,
                        device=accelerator.device,
                        val_ir_path=VAL_IR_PATH,
                        val_vis_path=VAL_VIS_PATH,
                    )

                    outf = os.path.join(
                        args.output_dir, "checkpoints",
                        f"model_{global_step:06d}.pkl",
                    )
                    accelerator.unwrap_model(net).save_model(outf)

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoints", "model_final.pkl")
        accelerator.unwrap_model(net).save_model(save_path)
        print("Saved final model →", save_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args_realsr_training()
    main(args)