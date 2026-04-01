"""
NAOSD.py  —  Stage-1 融合基础模型
路径: /private/home/wuhao/dnj/GDPO-main/GDPOSR/modelfile/NAOSD.py

核心修复（相对于上一版）：
  1. 不再预设 SD21_BLOCK_CHANNELS，改为在 __init__ 时
     通过 _probe_and_build_adapter() 动态探测每个 block 的真实输出 channel，
     并据此构建 LazyFusionConditionAdapter，完全规避 channel 不匹配问题。
  2. VAE encode 对象是 GT (c_t=x_tgt)，不是 masked 均值。
  3. unet_input = [latent_noisy(4ch) | xa_lat(3ch) | xb_lat(3ch)] = 10ch。
  4. FusionConditionAdapter zero-init，训练初期不扰动主干。
  5. CrossAttentionSpatial 含 NaN 防护（clamp + float32 softmax）。
  6. 注入策略：每个 down_block / mid_block / up_block 各注入一次（共9次），
     与 probe 逻辑严格对应。
"""

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
    patterns = ["conv1","conv2","conv_in","conv_shortcut","conv","conv_out",
                "to_k","to_q","to_v","to_out.0"]
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
    """
    扩展 conv_in 为 10 通道：
        前 4ch  → 复制 SD 原权重（latent 通道，保留生成先验）
        后 6ch  → 置零（训练初期不引入扰动）
    """
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet")

    old = unet.conv_in
    new_conv = nn.Conv2d(
        in_channels=10,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4, :, :].copy_(old.weight.data)
        if old.bias is not None:
            new_conv.bias.copy_(old.bias.data)
    unet.conv_in = new_conv

    unet.requires_grad_(False)
    unet.train()

    enc_mods, dec_mods, oth_mods = [], [], []
    patterns = ["to_k","to_q","to_v","to_out.0","conv","conv1","conv2",
                "conv_in","conv_shortcut","conv_out","proj_out","proj_in",
                "ff.net.2","ff.net.0.proj"]
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
# FusionConditionAdapter（probe 后构建）
# ══════════════════════════════════════════════

class FusionConditionAdapter(nn.Module):
    """
    在每个 UNet block（down/mid/up）之后注入 fusion 条件。
    channel_list 由 _probe_and_build_adapter() 动态决定，不预设。

    注入公式：h_new = h + delta
    delta 来自 fuse([h, ff_proj, cross_attn(h,fa), cross_attn(h,fb)])
    fuse 最后一层 Conv zero-init，训练初期 delta≈0。
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
                nn.Conv2d(ch, ch, 1),   # ← zero-init
            )
            nn.init.zeros_(fuse_block[-1].weight)
            nn.init.zeros_(fuse_block[-1].bias)
            self.fuse.append(fuse_block)

    def forward(self, h, idx, fa, fb, f_f):
        H, W = h.shape[-2:]
        ff   = F.interpolate(f_f, size=(H, W), mode="bilinear", align_corners=False)
        ff   = self.ff_proj[idx](ff)
        za   = self.fa_attn[idx](h, fa)
        zb   = self.fb_attn[idx](h, fb)
        delta = self.fuse[idx](torch.cat([h, ff, za, zb], dim=1))
        return h + delta


# ══════════════════════════════════════════════
# NAOSD  —  Stage-1 主类
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

        # ── fusion encoder modules ────────────
        self.semantic_encoder = SemanticEncoder(3, 128, 8, 4, dropout=0.1)
        self.content_encoder  = ContentEncoder(3, 128, 64)

        # ── base models ───────────────────────
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

        # ── 动态探测 channel，构建 adapter ────
        channel_list = self._probe_channels()
        self.cond_adapter = FusionConditionAdapter(
            channel_list=channel_list, cond_ch=128, ff_ch=64)

        # ── load checkpoint ───────────────────
        if getattr(args, "pretrained_path", None) is not None:
            print("==> loading pretrained:", args.pretrained_path)
            sd = torch.load(args.pretrained_path, map_location="cpu")
            self.load_ckpt_from_state_dict(sd)

        # ── move to cuda ──────────────────────
        self.unet.cuda()
        self.vae.cuda()
        self.semantic_encoder.cuda()
        self.content_encoder.cuda()
        self.cond_adapter.cuda()

        self.timesteps      = torch.tensor([args.time_step],       device="cuda").long()
        self.timestepsnoise = torch.tensor([args.time_step_noise], device="cuda").long()

    # ──────────────────────────────────────────
    # probe：干跑一次 UNet，每个 block 结束后记录 channel
    # ──────────────────────────────────────────
    @torch.no_grad()
    def _probe_channels(self):
        """
        注入策略：down_block × 4 + mid × 1 + up_block × 4 = 9 个注入点。
        每个注入点记录 block 主输出的 channel 数。
        """
        unet = self.unet
        was_training = unet.training
        unet.eval()

        # SD1.5 text embedding dim = 768
        dummy_input = torch.zeros(1, 10, 64, 64)
        dummy_text  = torch.zeros(1, 77, 768)

        sample = unet.conv_in(dummy_input)
        t_emb  = unet.time_proj(torch.tensor([999]).long()).to(sample.dtype)
        emb    = unet.time_embedding(t_emb)

        channel_list = []
        down_block_res_samples = (sample,)

        for down in unet.down_blocks:
            if getattr(down, "has_cross_attention", False):
                sample, res = down(hidden_states=sample, temb=emb,
                                   encoder_hidden_states=dummy_text)
            else:
                sample, res = down(hidden_states=sample, temb=emb)
            channel_list.append(sample.shape[1])
            down_block_res_samples += res

        if unet.mid_block is not None:
            sample = unet.mid_block(sample, emb, encoder_hidden_states=dummy_text)
            channel_list.append(sample.shape[1])

        for up in unet.up_blocks:
            res_samples = down_block_res_samples[-len(up.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up.resnets)]
            if getattr(up, "has_cross_attention", False):
                sample = up(hidden_states=sample, temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=dummy_text)
            else:
                sample = up(hidden_states=sample, temb=emb,
                            res_hidden_states_tuple=res_samples)
            channel_list.append(sample.shape[1])

        print(f"[NAOSD] probe channels ({len(channel_list)} injection points): {channel_list}")

        if was_training:
            unet.train()
        return channel_list

    # ──────────────────────────────────────────
    # train / eval
    # ──────────────────────────────────────────

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

    # ──────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────

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

    def _prepare_unet_input(self, extra_cond, latent_noisy):
        xa = extra_cond[:, :3]
        xb = extra_cond[:, 3:6]

        fa  = self.semantic_encoder(xa)
        fb  = self.semantic_encoder(xb)
        f_f = self.content_encoder(xa, xb)

        lh, lw = latent_noisy.shape[-2:]
        xa_lat = F.interpolate(xa, size=(lh, lw), mode="bilinear", align_corners=False)
        xb_lat = F.interpolate(xb, size=(lh, lw), mode="bilinear", align_corners=False)

        unet_input = torch.cat([latent_noisy, xa_lat, xb_lat], dim=1)  # (B,10,h,w)
        return fa, fb, f_f, unet_input

    def _unet_forward(self, unet_input, caption_enc, fa, fb, f_f):
        """
        注入顺序与 _probe_channels() 完全对应：
          idx 0~3 → down_blocks
          idx 4   → mid_block
          idx 5~8 → up_blocks
        """
        unet = self.unet

        sample = unet.conv_in(unet_input)
        t_emb  = unet.time_proj(self.timesteps).to(sample.dtype)
        emb    = unet.time_embedding(t_emb)
        enc_hs = caption_enc.to(sample.dtype)

        adapter_idx = 0
        down_block_res_samples = (sample,)

        # ── down ───────────────────────────────
        for down in unet.down_blocks:
            if getattr(down, "has_cross_attention", False):
                sample, res = down(hidden_states=sample, temb=emb,
                                   encoder_hidden_states=enc_hs)
            else:
                sample, res = down(hidden_states=sample, temb=emb)

            sample = self.cond_adapter(sample, adapter_idx, fa, fb, f_f)
            adapter_idx += 1
            down_block_res_samples += res

        # ── mid ────────────────────────────────
        if unet.mid_block is not None:
            sample = unet.mid_block(sample, emb, encoder_hidden_states=enc_hs)
            sample = self.cond_adapter(sample, adapter_idx, fa, fb, f_f)
            adapter_idx += 1

        # ── up ─────────────────────────────────
        for up in unet.up_blocks:
            res_samples = down_block_res_samples[-len(up.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up.resnets)]

            if getattr(up, "has_cross_attention", False):
                sample = up(hidden_states=sample, temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=enc_hs)
            else:
                sample = up(hidden_states=sample, temb=emb,
                            res_hidden_states_tuple=res_samples)

            sample = self.cond_adapter(sample, adapter_idx, fa, fb, f_f)
            adapter_idx += 1

        # ── output ─────────────────────────────
        if unet.conv_norm_out is not None:
            sample = unet.conv_norm_out(sample)
            sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)
        return sample

    # ──────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────

    def forward(self, c_t, positive_prompt=None, negative_prompt=None,
                args=None, extra_cond=None):
        caption_enc     = self.encode_prompt(positive_prompt)
        neg_caption_enc = self.encode_prompt(negative_prompt)

        # VAE encode GT
        latent = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        noise        = torch.randn_like(latent)
        latent_noisy = self.sched2.add_noise(latent, noise, self.timestepsnoise)

        fa, fb, f_f, unet_input = self._prepare_unet_input(extra_cond, latent_noisy)
        model_pred = self._unet_forward(unet_input, caption_enc, fa, fb, f_f)

        x_denoised = self.sched.step(
            model_pred, self.timesteps, latent_noisy, return_dict=True
        ).prev_sample

        output_image = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        return output_image, x_denoised, caption_enc, neg_caption_enc, noise

    # ──────────────────────────────────────────
    # save / load
    # ──────────────────────────────────────────

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
        # UNet LoRA
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

        # VAE LoRA
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

        # Fusion modules
        for key, mod in [
            ("state_dict_semantic_encoder", self.semantic_encoder),
            ("state_dict_content_encoder",  self.content_encoder),
            ("state_dict_cond_adapter",     self.cond_adapter),
        ]:
            if key in sd:
                mod.load_state_dict(sd[key], strict=True)