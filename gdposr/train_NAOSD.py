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
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

sys.path.append("/private/home/wuhao/dnj/GDPO-main/GDPOSR")

from modelfile.NAOSD import NAOSD
from my_utils.training_utils_realsr import parse_args_realsr_training, PairedSROnlineDataset


# ══════════════════════════════════════════════
# Loss functions（与 Mask-DiFuser Eq.14-17 对应）
# ══════════════════════════════════════════════

def loss_pixel(x_pred, x_gt):
    """L_pix: 像素 L2，约束 x̂0 vs x0"""
    return F.mse_loss(x_pred.float(), x_gt.float())


def loss_ssim(x_pred, x_gt):
    """L_ssim: 结构相似性损失"""
    x_pred = x_pred.float()
    x_gt   = x_gt.float()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p  = F.avg_pool2d(x_pred, 11, 1, 5)
    mu_g  = F.avg_pool2d(x_gt,   11, 1, 5)
    mu_p2, mu_g2, mu_pg = mu_p*mu_p, mu_g*mu_g, mu_p*mu_g
    sig_p2 = F.avg_pool2d(x_pred*x_pred, 11, 1, 5) - mu_p2
    sig_g2 = F.avg_pool2d(x_gt  *x_gt,   11, 1, 5) - mu_g2
    sig_pg = F.avg_pool2d(x_pred*x_gt,   11, 1, 5) - mu_pg
    ssim_map = ((2*mu_pg+C1)*(2*sig_pg+C2)) / \
               ((mu_p2+mu_g2+C1)*(sig_p2+sig_g2+C2))
    return 1.0 - ssim_map.mean()


class VGGPerceptual(nn.Module):
    """L_per: VGG 感知损失（relu1_2, relu2_2, relu3_3）"""
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.slice1 = nn.Sequential(*list(vgg.features)[:4]).eval()
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9]).eval()
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x_pred, x_gt):
        # 输入是 [-1,1]，先转 [0,1] 再 ImageNet normalize
        x_pred = ((x_pred.float() + 1) / 2 - self.mean) / self.std
        x_gt   = ((x_gt.float()   + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for s in [self.slice1, self.slice2, self.slice3]:
            x_pred = s(x_pred)
            x_gt   = s(x_gt)
            loss  += F.mse_loss(x_pred, x_gt)
        return loss


def loss_color(x_pred, x_gt):
    """L_col: RGB 颜色向量余弦距离（Eq.17），最强颜色约束"""
    pred_norm = F.normalize(x_pred.float(), p=2, dim=1)
    gt_norm   = F.normalize(x_gt.float(),   p=2, dim=1)
    return (1.0 - F.cosine_similarity(pred_norm, gt_norm, dim=1)).mean()


# ══════════════════════════════════════════════
# 验证推理（DDIM 多步，看真实 IR-VIS 融合效果）
# ══════════════════════════════════════════════

def run_val_fusion(net_unwrapped, step, output_dir, device,
                   val_ir_path, val_vis_path, ddim_steps=20):
    if val_ir_path is None or val_vis_path is None:
        return
    if not (os.path.isfile(val_ir_path) and os.path.isfile(val_vis_path)):
        print(f"[val] 找不到验证图，跳过")
        return

    from diffusers import DDIMScheduler

    net_unwrapped.set_eval()
    try:
        with torch.no_grad():
            def load_img(path):
                img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
                t = TF.to_tensor(img)
                return TF.normalize(t, [0.5]*3, [0.5]*3).unsqueeze(0).to(device)

            x_ir  = load_img(val_ir_path)
            x_vis = load_img(val_vis_path)

            ddim = DDIMScheduler.from_pretrained(
                net_unwrapped.args.pretrained_model_name_or_path,
                subfolder="scheduler")
            ddim.set_timesteps(ddim_steps, device=device)

            # 加噪起点：VIS latent 加噪到第一步时间步
            latent = net_unwrapped.vae.encode(x_vis).latent_dist.sample()
            latent = latent * net_unwrapped.vae.config.scaling_factor
            noise  = torch.randn_like(latent)
            t_start = ddim.timesteps[0]
            latent  = ddim.add_noise(latent, noise,
                                     torch.tensor([t_start], device=device))

            caption = net_unwrapped.encode_prompt([""])
            fa, fb, f_f = net_unwrapped._extract_fusion_cond(x_ir, x_vis)

            for ts in ddim.timesteps:
                lh, lw = latent.shape[-2:]
                xa_lat = F.interpolate(x_ir,  (lh,lw), mode="bilinear", align_corners=False)
                xb_lat = F.interpolate(x_vis, (lh,lw), mode="bilinear", align_corners=False)
                unet_input = torch.cat([latent, xa_lat, xb_lat], dim=1)
                t_tensor = torch.tensor([ts], device=device).long()
                noise_pred = net_unwrapped._unet_forward(
                    unet_input, caption, fa, fb, f_f, t=t_tensor)
                latent = ddim.step(noise_pred, ts, latent).prev_sample

            output = net_unwrapped.vae.decode(
                latent / net_unwrapped.vae.config.scaling_factor
            ).sample.clamp(-1, 1)

            grid = torch.cat([x_ir.cpu(), x_vis.cpu(), output.cpu()], dim=-1)
            save_path = os.path.join(output_dir, "eval",
                                     f"fusion_step_{step:06d}.png")
            save_image((grid + 1) / 2, save_path)
            print(f"[val] saved → {save_path}")

    except Exception as e:
        import traceback
        print(f"[val] failed at step {step}: {e}")
        traceback.print_exc()

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
        net.vae.set_adapter(["default_encoder", "default_decoder"])
    elif args.use_vae_encode_lora:
        net.vae.set_adapter(["default_encoder"])
    elif args.use_vae_decode_lora:
        net.vae.set_adapter(["default_decoder"])
    else:
        if hasattr(net.vae, "peft_config") and net.vae.peft_config:
            net.vae.disable_adapters()
    net.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    # ── 感知损失（VGG frozen）───────────────────
    vgg_loss = VGGPerceptual().cuda()

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
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
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
        net, optimizer, dl_train, lr_scheduler)
    vgg_loss = accelerator.prepare(vgg_loss)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(range(args.max_train_steps), desc="Steps",
                        disable=not accelerator.is_local_main_process)

    # ── 损失权重（Mask-DiFuser Eq.18）────────────
    lambda_diff = 1.0
    lambda_pix  = 0.05
    lambda_ssim = 0.05
    lambda_per  = 0.1
    lambda_col  = 1.0

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

            x_tgt = batch["HR"]     # (B,3,H,W) GT，[-1,1]
            xA    = batch["LR_A"]   # (B,3,H,W) masked A
            xB    = batch["LR_B"]   # (B,3,H,W) masked B
            B     = x_tgt.shape[0]

            # ── Mask-DiFuser 正确训练范式 ─────────────
            # t 随机采样在 forward_train 内部完成
            eps_pred, eps_target, x0_hat, _ = \
                accelerator.unwrap_model(net).forward_train(
                    x0=x_tgt, xA=xA, xB=xB
                )

            # ── 5项损失（Eq.18）──────────────────────
            # L_diff: 随机时间步噪声预测 MSE（真正的 diffusion loss）
            l_diff = F.mse_loss(eps_pred.float(), eps_target.float())

            # L_pix/L_ssim/L_per/L_col: 全部约束 (x̂0, x0)
            # x̂0 由 predict_x0_from_noise 反推，与 GT 在像素空间对齐
            l_pix  = loss_pixel(x0_hat, x_tgt)
            l_ssim = loss_ssim(x0_hat, x_tgt)
            l_per  = vgg_loss(x0_hat, x_tgt)
            l_col  = loss_color(x0_hat, x_tgt)

            loss = (lambda_diff * l_diff +
                    lambda_pix  * l_pix  +
                    lambda_ssim * l_ssim +
                    lambda_per  * l_per  +
                    lambda_col  * l_col)

            # ── NaN 防护 ─────────────────────────────
            if not torch.isfinite(loss):
                nan_count += 1
                print(f"[step {global_step}] non-finite loss ({nan_count}/50), skip")
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
                    "loss":      loss.detach().item(),
                    "loss_diff": l_diff.detach().item(),
                    "loss_pix":  l_pix.detach().item(),
                    "loss_ssim": l_ssim.detach().item(),
                    "loss_per":  l_per.detach().item(),
                    "loss_col":  l_col.detach().item(),
                    "lr":        lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    col=f"{l_col.item():.3f}",
                    per=f"{l_per.item():.3f}",
                )
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:

                    # ── adapter 激活监控 ──────────────
                    try:
                        adapter_w = accelerator.unwrap_model(net) \
                            .cond_adapter.fuse[4][-1].weight.abs().mean().item()
                        print(f"[step {global_step}] adapter_w: {adapter_w:.6f}")
                        accelerator.log({"adapter_w": adapter_w}, step=global_step)
                    except Exception:
                        pass

                    # ── 训练预览图：masked_A | x̂0 | GT ─
                    train_vis = torch.cat([
                        xA[:1].detach().cpu().float(),
                        x0_hat[:1].detach().cpu().float(),
                        x_tgt[:1].detach().cpu().float(),
                    ], dim=-1)
                    save_image(
                        (train_vis.clamp(-1,1) + 1) / 2,
                        os.path.join(args.output_dir, "eval",
                                     f"train_step_{global_step:06d}.png"),
                    )

                    # ── IR-VIS 融合验证（DDIM 20步）────
                    run_val_fusion(
                        net_unwrapped=accelerator.unwrap_model(net),
                        step=global_step,
                        output_dir=args.output_dir,
                        device=accelerator.device,
                        val_ir_path=VAL_IR_PATH,
                        val_vis_path=VAL_VIS_PATH,
                        ddim_steps=20,
                    )

                    # ── checkpoint ────────────────────
                    outf = os.path.join(args.output_dir, "checkpoints",
                                        f"model_{global_step:06d}.pkl")
                    accelerator.unwrap_model(net).save_model(outf)

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoints", "model_final.pkl")
        accelerator.unwrap_model(net).save_model(save_path)
        print("Saved final model →", save_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args_realsr_training()
    main(args)
