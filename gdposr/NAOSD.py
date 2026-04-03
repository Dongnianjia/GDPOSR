import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

from .fusion_modules import ContentEncoder, SemanticEncoder, CrossAttentionSpatial


# ══════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════

def make_1step_sched(pretrained_model_path: str):
    sched = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    sched.set_timesteps(1, device="cuda")
    sched.alphas_cumprod = sched.alphas_cumprod.cuda()
    return sched


def initialize_vae(rank, return_lora_module_names=False,
                   pretrained_model_name_or_path=None):
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    enc_mods, dec_mods, oth_mods = [], [], []
    patterns = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "to_k", "to_q", "to_v", "to_out.0"]
    for n, _ in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pat in patterns:
            if pat in n and "encoder" in n:
                enc_mods.append(n.replace(".weight", "")); break
            elif pat in n and "decoder" in n:
                dec_mods.append(n.replace(".weight", "")); break
            elif "quant_conv" in n and "post_quant_conv" not in n:
                enc_mods.append(n.replace(".weight", "")); break
            elif "post_quant_conv" in n:
                dec_mods.append(n.replace(".weight", "")); break
            elif pat in n:
                oth_mods.append(n.replace(".weight", "")); break

    vae.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian",
                               target_modules=enc_mods),
                    adapter_name="default_encoder")
    vae.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian",
                               target_modules=dec_mods),
                    adapter_name="default_decoder")
    if return_lora_module_names:
        return vae, enc_mods, dec_mods, oth_mods
    return vae


def initialize_unet_sr(rank, return_lora_module_names=False,
                        pretrained_model_name_or_path=None, args=None):
    """conv_in 扩展为 10 通道：前 4ch 复制 SD 权重，后 6ch 置零"""
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet")

    old = unet.conv_in
    new_conv = nn.Conv2d(10, old.out_channels,
                         old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4].copy_(old.weight.data)
        if old.bias is not None:
            new_conv.bias.copy_(old.bias.data)
    unet.conv_in = new_conv
    unet.requires_grad_(False)
    unet.train()

    enc_mods, dec_mods, oth_mods = [], [], []
    patterns = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in",
                "ff.net.2", "ff.net.0.proj"]
    for n, _ in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pat in patterns:
            if pat in n and ("down_blocks" in n or "conv_in" in n):
                enc_mods.append(n.replace(".weight", "")); break
            elif pat in n and "up_blocks" in n:
                dec_mods.append(n.replace(".weight", "")); break
            elif pat in n:
                oth_mods.append(n.replace(".weight", "")); break

    unet.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian",
                                target_modules=enc_mods),
                     adapter_name="default_encoder")
    unet.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian",
                                target_modules=dec_mods),
                     adapter_name="default_decoder")
    unet.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian",
                                target_modules=oth_mods),
                     adapter_name="default_others")

    if return_lora_module_names:
        return unet, enc_mods, dec_mods, oth_mods
    return unet


# ══════════════════════════════════════════════
# FusionConditionAdapter（无 direct_proj）
# ══════════════════════════════════════════════

class FusionConditionAdapter(nn.Module):
    """
    单路注入：attention 路（fa/fb cross-attention + f_f content）
    不含 direct_proj：在训练步数不足时 direct_proj 会引入未训练好的偏置，
    导致 IR 特征丢失、图像偏暗，步数充足后可再考虑加入。
    """
    def __init__(self, channel_list, cond_ch: int = 128, ff_ch: int = 64):
        super().__init__()
        self.channel_list = channel_list

        self.ff_proj = nn.ModuleList()
        self.fa_attn = nn.ModuleList()
        self.fb_attn = nn.ModuleList()
        self.fuse    = nn.ModuleList()

        for ch in channel_list:
            g = min(32, ch)
            self.ff_proj.append(nn.Sequential(
                nn.Conv2d(ff_ch, ch, 1),
                nn.GroupNorm(g, ch),
                nn.SiLU(),
            ))
            self.fa_attn.append(CrossAttentionSpatial(ch, cond_ch))
            self.fb_attn.append(CrossAttentionSpatial(ch, cond_ch))

            fuse_block = nn.Sequential(
                nn.Conv2d(ch * 4, ch, 3, 1, 1),
                nn.GroupNorm(g, ch),
                nn.SiLU(),
                nn.Conv2d(ch, ch, 1),   # zero-init → delta≈0 at start
            )
            nn.init.zeros_(fuse_block[-1].weight)
            nn.init.zeros_(fuse_block[-1].bias)
            self.fuse.append(fuse_block)

    def forward(self, h, idx, fa, fb, f_f):
        H, W = h.shape[-2:]
        ff    = F.interpolate(f_f, (H, W), mode="bilinear", align_corners=False)
        ff    = self.ff_proj[idx](ff)
        za    = self.fa_attn[idx](h, fa)
        zb    = self.fb_attn[idx](h, fb)
        delta = self.fuse[idx](torch.cat([h, ff, za, zb], dim=1))
        return h + delta


# ══════════════════════════════════════════════
# NAOSD
# ══════════════════════════════════════════════

class NAOSD(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)

        self.sched  = make_1step_sched(args.pretrained_model_name_or_path)
        self.sched2 = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        self.semantic_encoder = SemanticEncoder(3, 128, 8, 4, dropout=0.1)
        self.content_encoder  = ContentEncoder(3, 128, 64)

        vae, lora_vae_enc, lora_vae_dec, lora_vae_oth = initialize_vae(
            rank=args.lora_rank_vae,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            return_lora_module_names=True,
        )
        unet, lora_unet_enc, lora_unet_dec, lora_unet_oth = initialize_unet_sr(
            rank=args.lora_rank_unet,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            return_lora_module_names=True,
            args=args,
        )
        self.vae  = vae
        self.unet = unet

        self.lora_rank_vae  = args.lora_rank_vae
        self.lora_rank_unet = args.lora_rank_unet
        self.lora_vae_modules_encoder  = lora_vae_enc
        self.lora_vae_modules_decoder  = lora_vae_dec
        self.lora_vae_others           = lora_vae_oth
        self.lora_unet_modules_encoder = lora_unet_enc
        self.lora_unet_modules_decoder = lora_unet_dec
        self.lora_unet_others          = lora_unet_oth

        channel_list = self._probe_channels()
        self.cond_adapter = FusionConditionAdapter(
            channel_list=channel_list, cond_ch=128, ff_ch=64)

        if getattr(args, "pretrained_path", None) is not None:
            print("==> loading pretrained:", args.pretrained_path)
            sd = torch.load(args.pretrained_path, map_location="cpu")
            self.load_ckpt_from_state_dict(sd)

        self.unet.cuda()
        self.vae.cuda()
        self.semantic_encoder.cuda()
        self.content_encoder.cuda()
        self.cond_adapter.cuda()

        self.timesteps      = torch.tensor([args.time_step],       device="cuda").long()
        self.timestepsnoise = torch.tensor([args.time_step_noise], device="cuda").long()

    @torch.no_grad()
    def _probe_channels(self):
        unet = self.unet
        was_training = unet.training
        unet.eval()

        dummy_input = torch.zeros(1, 10, 64, 64)
        dummy_text  = torch.zeros(1, 77, 768)
        sample = unet.conv_in(dummy_input)
        t_emb  = unet.time_proj(torch.tensor([999]).long()).to(sample.dtype)
        emb    = unet.time_embedding(t_emb)

        channel_list = []
        down_block_res = (sample,)

        for down in unet.down_blocks:
            if getattr(down, "has_cross_attention", False):
                sample, res = down(hidden_states=sample, temb=emb,
                                   encoder_hidden_states=dummy_text)
            else:
                sample, res = down(hidden_states=sample, temb=emb)
            channel_list.append(sample.shape[1])
            down_block_res += res

        if unet.mid_block is not None:
            sample = unet.mid_block(sample, emb, encoder_hidden_states=dummy_text)
            channel_list.append(sample.shape[1])

        for up in unet.up_blocks:
            res = down_block_res[-len(up.resnets):]
            down_block_res = down_block_res[:-len(up.resnets)]
            if getattr(up, "has_cross_attention", False):
                sample = up(hidden_states=sample, temb=emb,
                            res_hidden_states_tuple=res,
                            encoder_hidden_states=dummy_text)
            else:
                sample = up(hidden_states=sample, temb=emb,
                            res_hidden_states_tuple=res)
            channel_list.append(sample.shape[1])

        print(f"[NAOSD] probe channels ({len(channel_list)} pts): {channel_list}")
        if was_training:
            unet.train()
        return channel_list

    def set_eval(self):
        for m in [self.unet, self.vae, self.semantic_encoder,
                  self.content_encoder, self.cond_adapter]:
            m.eval()
            m.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.semantic_encoder.train()
        self.content_encoder.train()
        self.cond_adapter.train()
        for n, p in self.unet.named_parameters():
            p.requires_grad = ("lora" in n or "conv_in" in n)
        for n, p in self.vae.named_parameters():
            p.requires_grad = ("lora" in n)
        for m in [self.semantic_encoder, self.content_encoder, self.cond_adapter]:
            for p in m.parameters():
                p.requires_grad = True

    @torch.no_grad()
    def encode_prompt(self, prompt):
        ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return self.text_encoder(ids.to(self.text_encoder.device))[0]

    def _extract_fusion_cond(self, xA, xB):
        fa  = self.semantic_encoder(xA)
        fb  = self.semantic_encoder(xB)
        f_f = self.content_encoder(xA, xB)
        return fa, fb, f_f

    def _unet_forward(self, unet_input, caption_enc, fa, fb, f_f, t=None):
        """t: LongTensor(B,) 或 (1,)，None 时用 self.timesteps"""
        if t is None:
            t = self.timesteps

        unet   = self.unet
        sample = unet.conv_in(unet_input)

        if t.shape[0] == sample.shape[0]:
            t_emb = unet.time_proj(t).to(sample.dtype)
            emb   = unet.time_embedding(t_emb)
        else:
            t_emb = unet.time_proj(t[:1]).to(sample.dtype)
            emb   = unet.time_embedding(t_emb).expand(sample.shape[0], -1)

        enc_hs = caption_enc.to(sample.dtype)

        adapter_idx = 0
        down_block_res = (sample,)

        for down in unet.down_blocks:
            if getattr(down, "has_cross_attention", False):
                sample, res = down(hidden_states=sample, temb=emb,
                                   encoder_hidden_states=enc_hs)
            else:
                sample, res = down(hidden_states=sample, temb=emb)
            sample = self.cond_adapter(sample, adapter_idx, fa, fb, f_f)
            adapter_idx += 1
            down_block_res += res

        if unet.mid_block is not None:
            sample = unet.mid_block(sample, emb, encoder_hidden_states=enc_hs)
            sample = self.cond_adapter(sample, adapter_idx, fa, fb, f_f)
            adapter_idx += 1

        for up in unet.up_blocks:
            res = down_block_res[-len(up.resnets):]
            down_block_res = down_block_res[:-len(up.resnets)]
            if getattr(up, "has_cross_attention", False):
                sample = up(hidden_states=sample, temb=emb,
                            res_hidden_states_tuple=res,
                            encoder_hidden_states=enc_hs)
            else:
                sample = up(hidden_states=sample, temb=emb,
                            res_hidden_states_tuple=res)
            sample = self.cond_adapter(sample, adapter_idx, fa, fb, f_f)
            adapter_idx += 1

        if unet.conv_norm_out is not None:
            sample = unet.conv_norm_out(sample)
            sample = unet.conv_act(sample)
        return unet.conv_out(sample)

    def forward_train(self, x0, xA, xB, t=None):
        """
        Mask-DiFuser 训练范式（Eq.12-18）
        返回 (eps_pred, eps_target, x0_hat, t)
        注意：返回 t 而非 latent_x0，供训练脚本计算 t_weight
        """
        B = x0.shape[0]

        with torch.no_grad():
            latent_x0 = (self.vae.encode(x0).latent_dist.sample()
                         * self.vae.config.scaling_factor)

        if t is None:
            t = torch.randint(
                0, self.sched2.config.num_train_timesteps,
                (B,), device=x0.device
            ).long()

        eps = torch.randn_like(latent_x0)
        xt  = self.sched2.add_noise(latent_x0, eps, t)

        fa, fb, f_f = self._extract_fusion_cond(xA, xB)

        lh, lw = xt.shape[-2:]
        xa_lat = F.interpolate(xA, (lh, lw), mode="bilinear", align_corners=False)
        xb_lat = F.interpolate(xB, (lh, lw), mode="bilinear", align_corners=False)
        unet_input  = torch.cat([xt, xa_lat, xb_lat], dim=1)
        caption_enc = self.encode_prompt([""] * B)
        eps_pred    = self._unet_forward(unet_input, caption_enc, fa, fb, f_f, t=t)

        # predict_x0_from_noise，clamp 防大 t 数值爆炸
        alphas_cumprod = self.sched2.alphas_cumprod.to(x0.device)
        alpha_t = alphas_cumprod[t].float()
        while alpha_t.dim() < latent_x0.dim():
            alpha_t = alpha_t.unsqueeze(-1)

        x0_hat_lat = (xt.float() - (1 - alpha_t).sqrt() * eps_pred.float()) \
                     / alpha_t.sqrt()
        x0_hat_lat = x0_hat_lat.clamp(-5, 5)

        x0_hat = self.vae.decode(
            x0_hat_lat / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        # 返回 t（不是 latent_x0），供训练脚本计算 t_weight
        return eps_pred, eps, x0_hat, t

    def forward(self, c_t, positive_prompt=None, negative_prompt=None,
                args=None, extra_cond=None):
        """one-step 推理接口，保持不变"""
        caption_enc     = self.encode_prompt(positive_prompt)
        neg_caption_enc = self.encode_prompt(negative_prompt)

        latent       = (self.vae.encode(c_t).latent_dist.sample()
                        * self.vae.config.scaling_factor)
        noise        = torch.randn_like(latent)
        latent_noisy = self.sched2.add_noise(latent, noise, self.timestepsnoise)

        xA = extra_cond[:, :3]
        xB = extra_cond[:, 3:6]
        fa, fb, f_f = self._extract_fusion_cond(xA, xB)

        lh, lw = latent_noisy.shape[-2:]
        xa_lat = F.interpolate(xA, (lh, lw), mode="bilinear", align_corners=False)
        xb_lat = F.interpolate(xB, (lh, lw), mode="bilinear", align_corners=False)
        unet_input = torch.cat([latent_noisy, xa_lat, xb_lat], dim=1)

        model_pred = self._unet_forward(unet_input, caption_enc, fa, fb, f_f, t=None)
        x_denoised = self.sched.step(
            model_pred, self.timesteps, latent_noisy, return_dict=True
        ).prev_sample
        output_image = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        return output_image, x_denoised, caption_enc, neg_caption_enc, noise

    def save_model(self, outf: str):
        sd = {
            "rank_unet": self.lora_rank_unet,
            "rank_vae":  self.lora_rank_vae,
            "vae_lora_encoder_modules":  self.lora_vae_modules_encoder,
            "vae_lora_decoder_modules":  self.lora_vae_modules_decoder,
            "vae_lora_others_modules":   self.lora_vae_others,
            "unet_lora_encoder_modules": self.lora_unet_modules_encoder,
            "unet_lora_decoder_modules": self.lora_unet_modules_decoder,
            "unet_lora_others_modules":  self.lora_unet_others,
            "adapter_channel_list": self.cond_adapter.channel_list,
            "state_dict_unet": {
                k: v for k, v in self.unet.state_dict().items()
                if "lora" in k or "conv_in" in k
            },
            "state_dict_vae": {
                k: v for k, v in self.vae.state_dict().items()
                if "lora" in k
            },
            "state_dict_semantic_encoder": self.semantic_encoder.state_dict(),
            "state_dict_content_encoder":  self.content_encoder.state_dict(),
            "state_dict_cond_adapter":     self.cond_adapter.state_dict(),
        }
        torch.save(sd, outf)
        print(f"[NAOSD] saved → {outf}")

    def load_ckpt_from_state_dict(self, sd: dict):
        if not (hasattr(self.unet, "peft_config") and self.unet.peft_config):
            self.unet.add_adapter(
                LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_encoder_modules"]),
                adapter_name="default_encoder")
            self.unet.add_adapter(
                LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_decoder_modules"]),
                adapter_name="default_decoder")
            self.unet.add_adapter(
                LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_others_modules"]),
                adapter_name="default_others")
        if "state_dict_unet" in sd:
            for n, p in self.unet.named_parameters():
                if n in sd["state_dict_unet"]:
                    p.data.copy_(sd["state_dict_unet"][n])

        if not (hasattr(self.vae, "peft_config") and self.vae.peft_config):
            self.vae.add_adapter(
                LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                           target_modules=sd["vae_lora_encoder_modules"]),
                adapter_name="default_encoder")
            self.vae.add_adapter(
                LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                           target_modules=sd["vae_lora_decoder_modules"]),
                adapter_name="default_decoder")
        if "state_dict_vae" in sd:
            for n, p in self.vae.named_parameters():
                if n in sd["state_dict_vae"]:
                    p.data.copy_(sd["state_dict_vae"][n])

        for key, mod in [
            ("state_dict_semantic_encoder", self.semantic_encoder),
            ("state_dict_content_encoder",  self.content_encoder),
            ("state_dict_cond_adapter",     self.cond_adapter),
        ]:
            if key in sd:
                mod.load_state_dict(sd[key], strict=True)
