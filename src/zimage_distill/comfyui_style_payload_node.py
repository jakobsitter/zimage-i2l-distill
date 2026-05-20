from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .style_bridge import (
    load_style_bridge_checkpoint,
    load_weighted_style_embeddings,
    parse_style_embedding_spec,
    resolve_style_adapter,
    resolve_style_tokens,
)
from .style_embed import StyleEmbedding, mix_style_embeddings


STYLE_TOKEN_PAYLOAD_TYPE = "STYLE_TOKEN_PAYLOAD"


def load_style_token_payload(
    style_embedding_path: str | Path,
    bridge_checkpoint_path: str | Path,
) -> dict[str, Any]:
    raw_spec = str(style_embedding_path).strip()
    if not raw_spec:
        raise ValueError("style embedding file does not exist: ")

    weighted_paths = parse_style_embedding_spec(raw_spec)
    for path, _ in weighted_paths:
        if not path.is_file():
            raise ValueError(f"style embedding file does not exist: {path}")

    bridge = load_style_bridge_checkpoint(bridge_checkpoint_path)
    weighted_embeddings = load_weighted_style_embeddings(weighted_paths)
    mixed_embedding = weighted_embeddings[0][0] if len(weighted_embeddings) == 1 else mix_style_embeddings(weighted_embeddings)
    bridge_kind = bridge.checkpoint.get("adapter_kind", "style_token_bridge")

    if bridge_kind == "ip_adapter":
        resolved_tokens = resolve_style_adapter(weighted_embeddings, bridge, num_layers=30)
    else:
        resolved_tokens = resolve_style_tokens(weighted_embeddings, bridge)

    bridge_embedding_dim = getattr(bridge.adapter, "embedding_dim", bridge.checkpoint.get("embedding_dim"))
    bridge_hidden_size = getattr(bridge.adapter, "hidden_size", bridge.checkpoint.get("hidden_size"))
    bridge_num_style_tokens = getattr(bridge.adapter, "num_style_tokens", bridge.checkpoint.get("num_style_tokens"))

    payload: dict[str, Any] = {
        "type": "zimage_style_token_payload",
        "style_embedding_path": raw_spec,
        "bridge_checkpoint_path": str(Path(bridge_checkpoint_path).expanduser()),
        "bridge_embedding_dim": bridge_embedding_dim,
        "bridge_hidden_size": bridge_hidden_size,
        "bridge_num_style_tokens": bridge_num_style_tokens,
        "bridge_adapter_kind": bridge_kind,
        "style_embedding_vector": mixed_embedding.vector.detach().cpu(),
        "style_tokens": resolved_tokens.combined_tokens.detach().cpu(),
        "source_images": mixed_embedding.source_images,
        "mode": mixed_embedding.mode,
        "metadata": mixed_embedding.metadata,
    }
    if resolved_tokens.component_vectors.shape[0] > 1 or mixed_embedding.vectors is not None:
        payload["style_token_blocks"] = resolved_tokens.per_component_tokens.detach().cpu()
        payload["style_token_weights"] = resolved_tokens.per_component_weights.detach().cpu()
        payload["style_embedding_vectors"] = resolved_tokens.component_vectors.detach().cpu()
    if hasattr(resolved_tokens, "ipadapter_kwargs_list"):
        payload["ipadapter_kwargs_list"] = [
            {"ip_hidden_states": kwargs["ip_hidden_states"].detach().cpu(), "scale": kwargs["scale"]}
            for kwargs in resolved_tokens.ipadapter_kwargs_list
        ]
    return payload


def apply_style_token_payload(
    conditioning: list[list[Any]],
    style_token_payload: dict[str, Any],
) -> list[list[Any]]:
    if not isinstance(conditioning, list):
        raise ValueError("conditioning must be a list")
    if not isinstance(style_token_payload, dict):
        raise ValueError("style_token_payload must be a dictionary")
    if "style_tokens" not in style_token_payload:
        raise ValueError("style_token_payload must include style_tokens")

    style_tokens = style_token_payload["style_tokens"]
    if not isinstance(style_tokens, torch.Tensor):
        raise ValueError("style_tokens must be a torch.Tensor")
    if style_tokens.ndim != 2:
        raise ValueError("style_tokens must have shape [num_style_tokens, hidden_size]")

    payload_metadata = {
        "type": style_token_payload.get("type"),
        "style_embedding_path": style_token_payload.get("style_embedding_path"),
        "bridge_checkpoint_path": style_token_payload.get("bridge_checkpoint_path"),
        "bridge_embedding_dim": style_token_payload.get("bridge_embedding_dim"),
        "bridge_hidden_size": style_token_payload.get("bridge_hidden_size"),
        "bridge_num_style_tokens": style_token_payload.get("bridge_num_style_tokens"),
        "source_images": style_token_payload.get("source_images"),
        "mode": style_token_payload.get("mode"),
        "metadata": style_token_payload.get("metadata"),
    }
    if "style_embedding_vectors" in style_token_payload:
        payload_metadata["style_embedding_vectors"] = style_token_payload["style_embedding_vectors"]
    if "style_token_blocks" in style_token_payload:
        payload_metadata["style_token_blocks"] = style_token_payload["style_token_blocks"]
    if "style_token_weights" in style_token_payload:
        payload_metadata["style_token_weights"] = style_token_payload["style_token_weights"]
    if "ipadapter_kwargs_list" in style_token_payload:
        payload_metadata["ipadapter_kwargs_list"] = style_token_payload["ipadapter_kwargs_list"]

    updated_conditioning: list[list[Any]] = []
    for entry in conditioning:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("conditioning entries must be [tensor, metadata] pairs")
        cond, metadata = entry
        if not isinstance(cond, torch.Tensor):
            raise ValueError("conditioning tensor must be a torch.Tensor")
        if cond.ndim != 3:
            raise ValueError("conditioning tensor must have shape [batch, tokens, hidden_size]")
        if cond.shape[-1] != style_tokens.shape[-1]:
            raise ValueError("style token hidden size must match conditioning hidden size")

        injected_tokens = style_tokens.to(device=cond.device, dtype=cond.dtype).unsqueeze(0)
        updated_metadata = dict(metadata)
        updated_metadata["zimage_style_token_payload"] = payload_metadata
        updated_conditioning.append([torch.cat([cond, injected_tokens], dim=1), updated_metadata])
    return updated_conditioning


class LoadStyleTokenPayloadNode:
    CATEGORY = "zimage/style"
    FUNCTION = "load_style_token_payload"
    RETURN_TYPES = (STYLE_TOKEN_PAYLOAD_TYPE,)
    RETURN_NAMES = ("style_token_payload",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "style_embedding_path": ("STRING", {"multiline": False}),
                "bridge_checkpoint_path": ("STRING", {"multiline": False}),
            }
        }

    def load_style_token_payload(
        self,
        style_embedding_path: str,
        bridge_checkpoint_path: str,
    ) -> tuple[dict[str, Any]]:
        payload = load_style_token_payload(
            style_embedding_path,
            bridge_checkpoint_path,
        )
        return (payload,)


class ApplyStyleTokenPayloadNode:
    CATEGORY = "zimage/style"
    FUNCTION = "apply_style_token_payload"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_token_payload": (STYLE_TOKEN_PAYLOAD_TYPE,),
            }
        }

    def apply_style_token_payload(
        self,
        conditioning: list[list[Any]],
        style_token_payload: dict[str, Any],
    ) -> tuple[list[list[Any]]]:
        return (apply_style_token_payload(conditioning, style_token_payload),)


def _make_style_hook(style_tokens: torch.Tensor, scale: float = 1.0):
    """Return a post-forward hook that adds style cross-attention to the output.

    Uses per-layer ``to_k_ip`` / ``to_v_ip`` weights extracted from the
    fused QKV projection so each layer can attend to style differently.
    """

    def _ip_hook(module, args, output):
        x, x_mask, freqs_cis = args[:3]
        transformer_options = args[3] if len(args) > 3 else {}

        dev = x.device
        dt = x.dtype
        bsz, seqlen, _ = x.shape
        n_local_heads = module.n_local_heads
        n_local_kv_heads = module.n_local_kv_heads
        head_dim = module.head_dim
        q_dim = n_local_heads * head_dim
        kv_dim = n_local_kv_heads * head_dim

        # All weight access uses explicit .to(device) to survive ComfyUI offloading.
        qkv_w = module.qkv.weight.to(device=dev, dtype=dt)
        qkv_b = module.qkv.bias.to(device=dev, dtype=dt) if module.qkv.bias is not None else None
        # Split fused QKV into dedicated K and V projection weights for style tokens.
        # This avoids running the unnecessary Q projection on style tokens and
        # gives each layer its own dedicated style-projections.
        k_w = qkv_w[q_dim : q_dim + kv_dim, :]  # [kv_dim, dim]
        v_w = qkv_w[q_dim + kv_dim :, :]          # [kv_dim, dim]

        # --- style K, V via dedicated per-layer projections ---
        st = style_tokens.unsqueeze(0).expand(bsz, -1, -1).to(device=dev, dtype=dt)
        style_k = torch.nn.functional.linear(st, k_w, None)
        style_v = torch.nn.functional.linear(st, v_w, None)
        num_st = style_k.shape[1]
        style_k = style_k.view(bsz, num_st, n_local_kv_heads, head_dim)
        style_v = style_v.view(bsz, num_st, n_local_kv_heads, head_dim)
        kn_w = module.k_norm.weight.to(device=dev, dtype=dt) if hasattr(module.k_norm, "weight") else None
        if kn_w is not None:
            style_k = style_k * torch.rsqrt(style_k.pow(2).mean(-1, keepdim=True) + 1e-6) * kn_w

        # --- query from input x (unchanged — we still need the Q part) ---
        x_qkv = torch.nn.functional.linear(x, qkv_w, qkv_b)
        xq, _, _ = torch.split(x_qkv, [q_dim, kv_dim, kv_dim], dim=-1)
        xq = xq.view(bsz, seqlen, n_local_heads, head_dim)
        qn_w = module.q_norm.weight.to(device=dev, dtype=dt) if hasattr(module.q_norm, "weight") else None
        if qn_w is not None:
            xq = xq * torch.rsqrt(xq.pow(2).mean(-1, keepdim=True) + 1e-6) * qn_w

        from comfy.ldm.flux.math import apply_rope
        xq_roped, _ = apply_rope(xq, xq, freqs_cis)

        n_rep = n_local_heads // n_local_kv_heads
        if n_rep >= 1:
            style_k = style_k.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            style_v = style_v.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        from comfy.ldm.modules.attention import optimized_attention_masked
        ip_out = optimized_attention_masked(
            xq_roped.movedim(1, 2), style_k.movedim(1, 2), style_v.movedim(1, 2),
            n_local_heads, None, skip_reshape=True, transformer_options=transformer_options,
        )
        out_w = module.out.weight.to(device=dev, dtype=dt)
        out_b = module.out.bias.to(device=dev, dtype=dt) if module.out.bias is not None else None
        ip_out = torch.nn.functional.linear(ip_out, out_w, out_b)
        return output + scale * ip_out

    return _ip_hook


def _patch_zimage_model(model, style_tokens: torch.Tensor, scale: float = 1.0) -> None:
    diffusion_model = model.model.diffusion_model
    layers = getattr(diffusion_model, "layers", None) or getattr(diffusion_model, "joint_blocks", None)
    if layers is None:
        raise RuntimeError("Cannot find transformer blocks on Z-Image diffusion model")
    hook = _make_style_hook(style_tokens, scale)
    for layer in layers:
        if hasattr(layer, "attention"):
            layer.attention.register_forward_hook(hook)


class ZImageApplyStyleToModel:
    """Inject style tokens into Z-Image attention blocks as cross-attention.

    Place this node on the MODEL path between the model loader and the
    sampler.  It wraps every transformer layer so that during denoising,
    the query attends to learned style-token keys and values — closer to
    an IP-Adapter than to prompt concatenation.
    """

    CATEGORY = "zimage/style"
    FUNCTION = "apply_style_to_model"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "model": ("MODEL",),
                "style_token_payload": (STYLE_TOKEN_PAYLOAD_TYPE,),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            },
        }

    def apply_style_to_model(self, model, style_token_payload: dict, scale: float = 1.0) -> tuple:
        style_tokens = style_token_payload.get("style_tokens")
        if style_tokens is None:
            raise ValueError("style_token_payload must include style_tokens")
        new_model = model.clone()
        _patch_zimage_model(new_model, style_tokens, scale)
        return (new_model,)


def _reshape_tensor(x: torch.Tensor, heads: int) -> torch.Tensor:
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1).transpose(1, 2)
    return x.reshape(bs, heads, length, -1)


class _PerceiverAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(1280)
        self.norm2 = nn.LayerNorm(1280)
        self.to_q = nn.Linear(1280, 1280, bias=False)
        self.to_kv = nn.Linear(1280, 2560, bias=False)
        self.to_out = nn.Linear(1280, 1280, bias=False)

    def forward(self, x, latents, shift=None, scale=None):
        x = self.norm1(x)
        latents = self.norm2(latents)
        if shift is not None and scale is not None:
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        b = latents.shape[0]
        q = _reshape_tensor(self.to_q(latents), 20)
        kv = torch.cat((x, latents), dim=-2)
        k, v = _reshape_tensor(self.to_kv(kv), 20).chunk(2, dim=-1)
        s = 1.0 / (64 ** 0.25)  # dim_head**-0.25
        w = (q * s) @ (k * s).transpose(-2, -1)
        w = torch.softmax(w.float(), dim=-1).type(w.dtype)
        out = (w @ v).permute(0, 2, 1, 3).reshape(b, -1, 1280)
        return self.to_out(out)


class _SD3IPAdapterResampler(nn.Module):
    """Minimal resampler matching the SD3.5 IP-Adapter structure (4-layer Perceiver)."""

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = {k: v.float() for k, v in load_file(checkpoint_path, device="cpu").items()}
        else:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            sd = {}
            for k, v in raw["image_proj"].items():
                sd[f"image_proj.{k}"] = v.float()
            for k, v in raw["ip_adapter"].items():
                sd[f"ip_adapter.{k}"] = v.float()

        self.latents = nn.Parameter(sd["image_proj.latents"])  # [1, 64, 1280]
        self.proj_in = nn.Linear(1152, 1280, bias=False)
        self.proj_in.weight.data = sd["image_proj.proj_in.weight"]
        # proj_out / norm_out may need padding (SD3.5: 2432 → Z-Image: 3840)
        po_w = sd["image_proj.proj_out.weight"]  # [2432, 1280] or [3840, 1280]
        src_dim = po_w.shape[0]
        self.proj_out = nn.Linear(1280, 3840, bias=True)
        if src_dim < 3840:
            self.proj_out.weight.data = torch.nn.functional.pad(po_w, (0, 0, 0, 3840 - src_dim))
            self.proj_out.bias.data = torch.nn.functional.pad(sd["image_proj.proj_out.bias"], (0, 3840 - src_dim))
        else:
            self.proj_out.weight.data = po_w
            self.proj_out.bias.data = sd["image_proj.proj_out.bias"]
        no_w = sd["image_proj.norm_out.weight"]  # [2432] or [3840]
        self.norm_out = nn.LayerNorm(3840)
        if no_w.shape[0] < 3840:
            self.norm_out.weight.data = torch.nn.functional.pad(no_w, (0, 3840 - no_w.shape[0]))
            self.norm_out.bias.data = torch.nn.functional.pad(sd["image_proj.norm_out.bias"], (0, 3840 - no_w.shape[0]))
        else:
            self.norm_out.weight.data = no_w
            self.norm_out.bias.data = sd["image_proj.norm_out.bias"]

        self.layers = nn.ModuleList()
        for d in range(4):
            pfx = f"image_proj.layers.{d}"
            attn = _PerceiverAttention()
            attn.norm1.weight.data = sd[f"{pfx}.0.norm1.weight"]
            attn.norm2.weight.data = sd[f"{pfx}.0.norm2.weight"]
            attn.to_q.weight.data = sd[f"{pfx}.0.to_q.weight"]
            attn.to_kv.weight.data = sd[f"{pfx}.0.to_kv.weight"]
            attn.to_out.weight.data = sd[f"{pfx}.0.to_out.weight"]
            ff = nn.Sequential(
                nn.LayerNorm(1280), nn.Linear(1280, 5120, bias=False),
                nn.GELU(), nn.Linear(5120, 1280, bias=False),
            )
            # Key mapping: .1.0 = LayerNorm, .1.1 = Linear in, .1.3 = Linear out
            ff[0].weight.data = sd[f"{pfx}.1.0.weight"]
            ff[0].bias.data = sd[f"{pfx}.1.0.bias"]
            ff[1].weight.data = sd[f"{pfx}.1.1.weight"]
            ff[3].weight.data = sd[f"{pfx}.1.3.weight"]
            adaln = nn.Sequential(nn.SiLU(), nn.Linear(1280, 5120, bias=True))
            adaln[1].weight.data = sd[f"{pfx}.2.1.weight"]
            adaln[1].bias.data = sd[f"{pfx}.2.1.bias"]
            self.layers.append(nn.ModuleList([attn, ff, adaln]))

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(clip_features)
        latents = self.latents.expand(x.size(0), -1, -1)
        zero_emb = torch.zeros(x.size(0), 1280, device=x.device, dtype=x.dtype)
        for attn, ff, adaln in self.layers:
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaln(zero_emb).chunk(4, dim=1)
            latents = attn(x, latents, shift_msa, scale_msa) + latents
            res = latents
            latents = ff(latents) + res
        return self.norm_out(self.proj_out(latents))


_ip_adapter_cache: dict[str, object] = {}


def _load_ip_weights(ip_path: str) -> dict:
    if ip_path not in _ip_adapter_cache:
        from safetensors.torch import load_file
        _ip_adapter_cache[ip_path] = load_file(ip_path, device="cpu")
    return _ip_adapter_cache[ip_path]


class ZImageApplySD3IPAdapter:
    """Apply converted SD3.5 IP-Adapter weights to Z-Image attention blocks.

    Two modes:
    - **Path**: load a pre-computed ``.ip_tokens.safetensors`` file.
    - **Token string**: paste a JSON string with ``{"tokens": path, "strength": 1.0, ...}`` for multi-style blending.
    """

    CATEGORY = "zimage/style"
    FUNCTION = "apply_ipadapter"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "ip_token_path": ("STRING", {"multiline": False, "default": ""}),
                "ipadapter_weights": ("STRING", {
                    "multiline": False,
                    "default": "/home/jakob/comfy/ComfyUI/models/ipadapter/zimage_ip.safetensors",
                }),
                "weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
        }

    def apply_ipadapter(self, model, ip_token_path: str, ipadapter_weights: str, weight: float) -> tuple:
        import os, json
        if not os.path.isfile(ipadapter_weights):
            raise ValueError(f"IP-Adapter weights not found: {ipadapter_weights}")

        # Load pre-computed IP tokens
        tokens = self._load_tokens(ip_token_path.strip())
        if tokens is None:
            raise ValueError(f"No IP tokens found at: {ip_token_path}")

        ip_sd = _load_ip_weights(ipadapter_weights)
        new_model = model.clone()
        self._patch(new_model, tokens, ip_sd, weight)
        return (new_model,)

    @staticmethod
    def _load_tokens(path_or_spec: str) -> torch.Tensor | None:
        import json, os
        from safetensors.torch import load_file

        if not path_or_spec:
            return None
        # Support JSON spec for multi-image: {"embeddings": [{"path": ..., "strength": 1.0}, ...]}
        if path_or_spec.startswith("{") or path_or_spec.startswith("["):
            spec = json.loads(path_or_spec)
            items = spec if isinstance(spec, list) else spec.get("embeddings", [spec])
            blended = None
            total_w = 0.0
            for item in items:
                p = item["path"] if isinstance(item, dict) else item
                s = float(item.get("strength", 1.0)) if isinstance(item, dict) else 1.0
                sd = load_file(str(p), device="cpu")
                t = sd["ip_tokens"]
                if blended is None:
                    blended = torch.zeros_like(t)
                blended = blended + s * t
                total_w += s
            return blended / total_w if total_w > 0 else blended

        p = Path(path_or_spec).expanduser()
        if p.is_file():
            sd = load_file(str(p), device="cpu")
            return sd["ip_tokens"]
        return None

    def _patch(self, model, ip_tokens, ip_sd, weight):
        while ip_tokens.ndim > 2:
            ip_tokens = ip_tokens.squeeze(0)  # squeeze any extra leading dims → [64, 3840]
        diffusion = model.model.diffusion_model
        layers = getattr(diffusion, "layers", None)
        if layers is None:
            raise RuntimeError("Cannot find transformer layers")

        for li, layer in enumerate(layers):
            if not hasattr(layer, "attention") or li >= 30:
                continue
            pfx = f"ip_adapter.{li}."
            k_w = ip_sd[f"{pfx}to_k_ip.weight"]
            v_w = ip_sd[f"{pfx}to_v_ip.weight"]
            qn_w = ip_sd.get(f"{pfx}norm_q.weight", None)
            kn_w = ip_sd.get(f"{pfx}norm_k.weight", None)
            ikn_w = ip_sd.get(f"{pfx}norm_ip_k.weight", None)

            hook = _make_trained_ip_hook(ip_tokens, k_w, v_w, qn_w, kn_w, ikn_w, weight)
            layer.attention.register_forward_hook(hook)



def _make_trained_ip_hook(ip_tokens, k_w, v_w, qn_w, kn_w, ikn_w, scale):
    def _hook(module, args, output):
        x, x_mask, freqs_cis = args[:3]
        transformer_options = args[3] if len(args) > 3 else {}
        dev, dt = x.device, x.dtype
        bsz, seqlen, _ = x.shape
        nh = module.n_local_heads
        nkh = module.n_local_kv_heads
        hd = module.head_dim
        qd, kd = nh * hd, nkh * hd

        qkv_w = module.qkv.weight.to(device=dev, dtype=dt)
        qkv_b = module.qkv.bias.to(device=dev, dtype=dt) if module.qkv.bias is not None else None

        # Style K,V via trained IP weights — cast to model dtype at runtime
        st = ip_tokens.unsqueeze(0).expand(bsz, -1, -1).to(device=dev, dtype=dt)
        sk = torch.nn.functional.linear(st, k_w.to(device=dev, dtype=dt), None)
        sv = torch.nn.functional.linear(st, v_w.to(device=dev, dtype=dt), None)
        nst = sk.shape[1]
        sk = sk.view(bsz, nst, nkh, hd)
        sv = sv.view(bsz, nst, nkh, hd)
        if kn_w is not None:
            kw = kn_w.to(device=dev, dtype=dt)
            sk = sk * torch.rsqrt(sk.pow(2).mean(-1, keepdim=True) + 1e-6) * kw

        # Query from input
        xqkv = torch.nn.functional.linear(x, qkv_w, qkv_b)
        xq, _, _ = torch.split(xqkv, [qd, kd, kd], dim=-1)
        xq = xq.view(bsz, seqlen, nh, hd)
        if qn_w is not None:
            xq = xq * torch.rsqrt(xq.pow(2).mean(-1, keepdim=True) + 1e-6) * qn_w.to(device=dev, dtype=dt)

        from comfy.ldm.flux.math import apply_rope
        xq, _ = apply_rope(xq, xq, freqs_cis)
        nrep = nh // nkh
        if nrep >= 1:
            sk = sk.unsqueeze(3).repeat(1, 1, 1, nrep, 1).flatten(2, 3)
            sv = sv.unsqueeze(3).repeat(1, 1, 1, nrep, 1).flatten(2, 3)

        from comfy.ldm.modules.attention import optimized_attention_masked
        ip = optimized_attention_masked(
            xq.movedim(1, 2), sk.movedim(1, 2), sv.movedim(1, 2),
            nh, None, skip_reshape=True, transformer_options=transformer_options,
        )
        ow = module.out.weight.to(device=dev, dtype=dt)
        ob = module.out.bias.to(device=dev, dtype=dt) if module.out.bias is not None else None
        ip = torch.nn.functional.linear(ip, ow, ob)
        return output + scale * ip
    return _hook


class ZImageApplyLoRA:
    """Generate and apply style LoRA weights directly from reference images.

    Paste one or more reference image paths (comma-separated).  The node
    runs the student encoder + i2l decoder on CPU, converts the weights
    to ComfyUI's fused-QKV format, and patches the model — all in one step.
    """

    CATEGORY = "zimage/style"
    FUNCTION = "apply_lora"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "reference_images": ("STRING", {"multiline": True, "default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
        }

    def apply_lora(self, model, reference_images: str, strength: float) -> tuple:
        import os
        paths = [p.strip() for p in reference_images.split(",") if p.strip()]
        if not paths:
            raise ValueError("At least one reference image path is required")
        for p in paths:
            if not os.path.isfile(p):
                raise ValueError(f"Reference image not found: {p}")

        from .lora_export import export_lora
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        tmp.close()
        try:
            student_ckpt = Path("/home/jakob/Developer/world-model-studio/zimage-i2l-distill/checkpoints/student_dual-dinov3s-siglip2b.pt")
            export_lora(
                reference_images=[Path(p) for p in paths],
                student_checkpoint=student_ckpt,
                decoder_checkpoint=None,
                strengths=None,
                output=Path(tmp.name),
                device="cpu",
            )
            patches = _load_and_convert_lora(Path(tmp.name), strength)
        finally:
            os.unlink(tmp.name)

        new_model = model.clone()
        for key, delta in patches.items():
            new_model.add_patches({key: (delta,)}, strength_patch=1.0, strength_model=1.0)
        return (new_model,)


def _load_and_convert_lora(lora_path: Path, strength: float) -> dict[str, torch.Tensor]:
    """Load DiffSynth LoRA and convert to ComfyUI fused-QKV patch dict."""
    from safetensors.torch import load_file
    from collections import defaultdict

    sd = load_file(str(lora_path), device="cpu")
    groups = defaultdict(dict)
    for k, v in sd.items():
        base = re.sub(r"\.lora_[AB]\.default\.weight$", "", k)
        groups[base][k.split(".")[-3]] = v.float()

    patches: dict[str, torch.Tensor] = {}
    qkv_groups: dict[str, dict[str, dict[str, torch.Tensor]]] = defaultdict(dict)

    for base, parts in groups.items():
        if "lora_A" not in parts or "lora_B" not in parts:
            continue
        new_base = base
        new_base = new_base.replace("layers.", "diffusion_model.layers.")
        new_base = new_base.replace("context_refiner.", "diffusion_model.context_refiner.")
        new_base = new_base.replace("noise_refiner.", "diffusion_model.noise_refiner.")
        new_base = new_base.replace(".to_out.0", ".out")

        if ".to_q" in new_base or ".to_k" in new_base or ".to_v" in new_base:
            fusion_key = new_base.rsplit(".to_", 1)[0] + ".qkv.weight"
            which = new_base.rsplit(".to_", 1)[1]
            for lt in ("lora_A", "lora_B"):
                qkv_groups[fusion_key][which][lt] = parts[lt]
        else:
            delta = (parts["lora_B"] @ parts["lora_A"]) * strength
            patches[new_base + ".weight"] = delta

    for fusion_key, which_parts in qkv_groups.items():
        q = which_parts.get("q", {})
        k = which_parts.get("k", {})
        v = which_parts.get("v", {})
        if "lora_A" in q and "lora_A" in k and "lora_A" in v:
            delta = torch.cat([
                q["lora_B"] @ q["lora_A"],
                k["lora_B"] @ k["lora_A"],
                v["lora_B"] @ v["lora_A"],
            ], dim=0) * strength
            patches[fusion_key] = delta

    return patches


NODE_CLASS_MAPPINGS = {
    "ZImageLoadStyleTokenPayload": LoadStyleTokenPayloadNode,
    "ZImageApplyStyleTokenPayload": ApplyStyleTokenPayloadNode,
    "ZImageApplyStyleToModel": ZImageApplyStyleToModel,
    "ZImageApplySD3IPAdapter": ZImageApplySD3IPAdapter,
    "ZImageApplyLoRA": ZImageApplyLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoadStyleTokenPayload": "ZImage Load Style Token Payload",
    "ZImageApplyStyleTokenPayload": "ZImage Apply Style Token Payload",
    "ZImageApplyStyleToModel": "ZImage Apply Style To Model",
    "ZImageApplySD3IPAdapter": "ZImage Apply SD3 IP Adapter",
    "ZImageApplyLoRA": "ZImage Apply LoRA",
}
