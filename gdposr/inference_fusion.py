import os
import sys
import argparse
import glob

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from diffusers import DDIMScheduler

sys.path.append("/private/home/wuhao/dnj/GDPO-main/GDPOSR")
from modelfile.NAOSD import NAOSD

INFER_SIZE = 512


def load_img(path, device):
    img = Image.open(path).convert("RGB").resize(
        (INFER_SIZE, INFER_SIZE), Image.LANCZOS)
    t = TF.to_tensor(img)
    t = TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return t.unsqueeze(0).to(device)


def ddim_inference(model, x_ir, x_vis, device, ddim_scheduler):
    with torch.no_grad():
        latent = model.vae.encode(x_vis).latent_dist.sample()
        latent = latent * model.vae.config.scaling_factor
        noise  = torch.randn_like(latent)

        # 正确起始点：加噪到 DDIM 第一步对应的时间步
        t_start = ddim_scheduler.timesteps[0]
        latent  = ddim_scheduler.add_noise(
            latent, noise, torch.tensor([t_start], device=device))

        caption = model.encode_prompt([""])
        fa  = model.semantic_encoder(x_ir)
        fb  = model.semantic_encoder(x_vis)
        f_f = model.content_encoder(x_ir, x_vis)

        for ts in ddim_scheduler.timesteps:
            lh, lw = latent.shape[-2:]
            xa_lat = F.interpolate(x_ir,  (lh,lw), mode="bilinear", align_corners=False)
            xb_lat = F.interpolate(x_vis, (lh,lw), mode="bilinear", align_corners=False)
            unet_input = torch.cat([latent, xa_lat, xb_lat], dim=1)
            model.timesteps = torch.tensor([ts], device=device).long()
            noise_pred = model._unet_forward(unet_input, caption, fa, fb, f_f)
            latent = ddim_scheduler.step(noise_pred, ts, latent).prev_sample

        output = model.vae.decode(
            latent / model.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

    return output


def main(args):
    device = torch.device("cuda")
    print("===> loading model")

    model = NAOSD(args)
    model.vae.set_adapter(["default_encoder", "default_decoder"])
    model.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
    model.set_eval()
    model = model.to(device).half()

    ddim_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_scheduler.set_timesteps(args.ddim_steps, device=device)

    ir_list  = sorted(glob.glob(os.path.join(args.ir_dir,  "*.png")))
    vis_list = sorted(glob.glob(os.path.join(args.vis_dir, "*.png")))
    if not ir_list:
        ir_list  = sorted(glob.glob(os.path.join(args.ir_dir,  "*.jpg")))
        vis_list = sorted(glob.glob(os.path.join(args.vis_dir, "*.jpg")))

    assert len(ir_list) == len(vis_list) and len(ir_list) > 0
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Total pairs: {len(ir_list)}, DDIM steps: {args.ddim_steps}")

    for i, (ir_path, vis_path) in enumerate(zip(ir_list, vis_list)):
        name = os.path.basename(ir_path)
        orig_w, orig_h = Image.open(ir_path).size

        x_ir  = load_img(ir_path,  device).half()
        x_vis = load_img(vis_path, device).half()

        output = ddim_inference(model, x_ir, x_vis, device, ddim_scheduler)

        output  = (output.clamp(-1, 1) + 1) / 2
        out_pil = TF.to_pil_image(output[0].float().cpu())
        if (orig_w, orig_h) != (INFER_SIZE, INFER_SIZE):
            out_pil = out_pil.resize((orig_w, orig_h), Image.LANCZOS)

        out_pil.save(os.path.join(args.output_path, name))
        if i % 50 == 0:
            print(f"[{i}/{len(ir_list)}] {name}")

    print(f"===> done → {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_dir",      type=str, required=True)
    parser.add_argument("--vis_dir",     type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--ddim_steps",      type=int, default=20)
    parser.add_argument("--time_step",       type=int, default=999)
    parser.add_argument("--time_step_noise", type=int, default=250)
    parser.add_argument("--lora_rank_unet",  type=int, default=8)
    parser.add_argument("--lora_rank_vae",   type=int, default=4)
    args = parser.parse_args()
    main(args)
