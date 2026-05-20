"""Export IP-Adapter tokens from reference images using SD3.5 IP-Adapter.

Pre-computes the resampler output (64 tokens × 3840 dim) so ComfyUI
can inject them without loading SigLIP at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import SiglipImageProcessor, SiglipVisionModel


def load_siglip() -> tuple[SiglipVisionModel, SiglipImageProcessor]:
    model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
    model.eval()
    processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    return model, processor


def build_resampler(ipa_checkpoint: Path) -> torch.nn.Module:
    from .comfyui_style_payload_node import _SD3IPAdapterResampler
    return _SD3IPAdapterResampler(str(ipa_checkpoint))


def encode_reference_images(
    image_paths: list[Path],
    *,
    siglip_model: SiglipVisionModel,
    processor: SiglipImageProcessor,
    resampler: torch.nn.Module,
    device: str,
) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = siglip_model(**inputs, output_hidden_states=True)
        # penultimate_hidden_states: [B, N, 1152]
        clip_feats = outputs.hidden_states[-2]
        ip_tokens = resampler(clip_feats)  # [B, 64, 3840]
    return ip_tokens.cpu()


def save_ip_tokens(tokens: torch.Tensor, output_path: Path, source_images: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    metadata = json.dumps({"source_images": source_images})
    save_file({"ip_tokens": tokens}, str(output_path), metadata={"info": metadata})
    print(f"Saved {tokens.shape} to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export IP-Adapter tokens from reference images.")
    parser.add_argument("--reference-image", type=Path, required=True, action="append", dest="reference_images",
                        help="Reference image path (repeat for multiple images).")
    parser.add_argument("--ipa-checkpoint", type=Path, required=True,
                        help="Path to SD3.5 IP-Adapter checkpoint (ip-adapter.bin).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .safetensors path for the IP tokens.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device for SigLIP + resampler.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = args.device

    print(f"Loading SigLIP on {device}...")
    siglip_model, processor = load_siglip()
    siglip_model = siglip_model.to(device)

    print(f"Loading resampler from {args.ipa_checkpoint}...")
    resampler = build_resampler(args.ipa_checkpoint).to(device)

    print(f"Encoding {len(args.reference_images)} reference image(s)...")
    tokens = encode_reference_images(
        args.reference_images,
        siglip_model=siglip_model,
        processor=processor,
        resampler=resampler,
        device=device,
    )
    # If multiple images, pool: mean of per-image tokens
    if tokens.shape[0] > 1:
        tokens = tokens.mean(dim=0, keepdim=False)  # [64, 3840]

    source_names = [str(p) for p in args.reference_images]
    save_ip_tokens(tokens, args.output, source_names)

    # Cleanup
    del siglip_model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
