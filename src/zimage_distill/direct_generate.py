from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffsynth.diffusion.base_pipeline import PipelineUnit

from .generate import DEFAULT_NEGATIVE_PROMPT, _normalize_cuda_device, load_generation_pipeline
from .style_bridge import load_style_bridge_checkpoint, resolve_style_adapter, resolve_style_tokens
from .style_embed import StyleEmbedding


class StyleTokenInjectionUnit(PipelineUnit):
    def __init__(self, style_tokens: torch.Tensor) -> None:
        super().__init__(take_over=True)
        self.style_tokens = style_tokens

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        prompt_embeds = inputs_posi.get("prompt_embeds")
        if prompt_embeds is None:
            return inputs_shared, inputs_posi, inputs_nega

        if not isinstance(prompt_embeds, list):
            prompt_embeds = [prompt_embeds]

        inputs_posi["prompt_embeds"] = [
            torch.cat(
                [self.style_tokens.to(device=prompt_embed.device, dtype=prompt_embed.dtype), prompt_embed],
                dim=0,
            )
            for prompt_embed in prompt_embeds
        ]
        return inputs_shared, inputs_posi, inputs_nega


def insert_style_tokens(pipe, style_tokens: torch.Tensor) -> StyleTokenInjectionUnit:
    unit = StyleTokenInjectionUnit(style_tokens)
    for index, existing_unit in enumerate(pipe.units):
        if existing_unit.__class__.__name__ == "ZImageUnit_PromptEmbedder":
            pipe.units.insert(index + 1, unit)
            return unit
    raise RuntimeError("prompt embedder unit was not found")

def generate_image_with_style_embeddings(
    pipe,
    weighted_style_embeddings: list[tuple[StyleEmbedding, float]],
    bridge_checkpoint_path: str | Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    cfg_scale: float,
    num_inference_steps: int,
    sigma_shift: float,
):
    bridge = load_style_bridge_checkpoint(bridge_checkpoint_path)
    if bridge.checkpoint.get("adapter_kind") == "ip_adapter":
        num_layers = len(getattr(getattr(pipe, "dit", None), "layers", [])) or 30
        resolved = resolve_style_adapter(weighted_style_embeddings, bridge, num_layers=num_layers)
        ipadapter_kwargs_list = [
            {**ipadapter_kwargs, "ip_hidden_states": ipadapter_kwargs["ip_hidden_states"].to(device=pipe.device)}
            for ipadapter_kwargs in resolved.ipadapter_kwargs_list
        ]
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            ipadapter_kwargs_list=ipadapter_kwargs_list,
        )

    resolved = resolve_style_tokens(weighted_style_embeddings, bridge)
    style_tokens = resolved.combined_tokens.to(device=pipe.device)
    insert_style_tokens(pipe, style_tokens)
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
    )



def generate_image_with_style_embedding(
    pipe,
    style_embedding: StyleEmbedding,
    bridge_checkpoint_path: str | Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    cfg_scale: float,
    num_inference_steps: int,
    sigma_shift: float,
):
    return generate_image_with_style_embeddings(
        pipe,
        [(style_embedding, 1.0)],
        bridge_checkpoint_path,
        prompt,
        negative_prompt,
        seed,
        cfg_scale,
        num_inference_steps,
        sigma_shift,
    )


def _load_weighted_style_embeddings(style_embedding_paths: list[Path], style_strengths: list[float]) -> list[tuple[StyleEmbedding, float]]:
    if not style_embedding_paths:
        raise ValueError("style_embedding_paths must not be empty")
    if style_strengths and len(style_strengths) not in {1, len(style_embedding_paths)}:
        raise ValueError("style strengths must be omitted, specified once, or match the number of style embeddings")

    if not style_strengths:
        strengths = [1.0] * len(style_embedding_paths)
    elif len(style_strengths) == 1 and len(style_embedding_paths) > 1:
        strengths = style_strengths * len(style_embedding_paths)
    else:
        strengths = style_strengths

    return [(StyleEmbedding.load(path), strength) for path, strength in zip(style_embedding_paths, strengths)]



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zimage-distill-direct-generate",
        description="Generate an image using a saved style embedding and direct conditioning.",
    )
    parser.add_argument("--style-embedding", action="append", required=True, type=Path, help="Style embedding path. Repeat for multiple references.")
    parser.add_argument("--style-strength", action="append", type=float, default=[], help="Per-style strength. Repeat in the same order as --style-embedding.")
    parser.add_argument("--bridge-checkpoint", required=True, type=Path, help="Trained bridge checkpoint path.")
    parser.add_argument("--prompt", required=True, help="Text prompt for generation.")
    parser.add_argument("--output", required=True, type=Path, help="Output image path.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sigma-shift", type=float, default=8.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = _normalize_cuda_device(args.device)

    weighted_style_embeddings = _load_weighted_style_embeddings(args.style_embedding, args.style_strength)
    pipe = load_generation_pipeline(device)

    print(f"Generating with {len(weighted_style_embeddings)} style embedding(s): {', '.join(str(path) for path in args.style_embedding)}")
    print("Starting direct generation...")
    image = generate_image_with_style_embeddings(
        pipe,
        weighted_style_embeddings,
        args.bridge_checkpoint,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        sigma_shift=args.sigma_shift,
    )
    print(f"Generated image: type={type(image).__name__}, size={getattr(image, 'size', None)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {out}")
    image.save(str(out))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
