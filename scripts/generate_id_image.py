import argparse
import random
from pathlib import Path
from typing import Literal

import PIL.Image
import torch
from thirdparty.InfiniteYou.pipelines.pipeline_infu_flux import InfUFluxPipeline

from src.defs import ROOT


def main(
    prompt: str,
    id_image_path: Path,
    output_image_path: Path | None = None,
    model_version: Literal["sim_stage1", "aes_stage2"] = "sim_stage1",
    enable_realism_lora: bool = True,
    enable_anti_blur_lora: bool = False,
    width: int = 864,
    height: int = 1152,
    num_steps: int = 50,
    optimize_vram: bool = False,
    guidance_scale: float = 3.5,
    seed: int = -1,
    infusenet_conditioning_scale: float = 1.0,
    infusenet_guidance_start: float = 0.0,
    infusenet_guidance_end: float = 1.0,
) -> None:
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    id_image = PIL.Image.open(id_image_path).convert("RGB")

    model_dir = ROOT / "models" / "InfiniteYou"
    infu_model_path = model_dir / "infu_flux_v1.0" / model_version
    insightface_root_path = model_dir / "supports" / "insightface"
    pipe = InfUFluxPipeline(
        base_model_path="black-forest-labs/FLUX.1-dev",
        infu_model_path=str(infu_model_path),
        insightface_root_path=str(insightface_root_path),
        infu_flux_version="v1.0",
        model_version=model_version,
    )

    lora_dir = model_dir / "supports" / "optional_loras"
    loras = []
    if enable_realism_lora:
        loras.append([str(lora_dir / "flux_realism_lora.safetensors"), "realism", 1.0])
    if enable_anti_blur_lora:
        loras.append(
            [
                str(lora_dir / "flux_anti_blur_lora.safetensors"),
                "anti_blur",
                1.0,
            ]
        )
    pipe.load_loras(loras)

    if optimize_vram:
        pipe.pipe.vae.enable_slicing()
        pipe.pipe.vae.enable_tiling()
        pipe.pipe.enable_model_cpu_offload()

    with torch.inference_mode():
        image = pipe(
            id_image=id_image,
            prompt=prompt,
            control_image=None,
            seed=seed,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            infusenet_conditioning_scale=infusenet_conditioning_scale,
            infusenet_guidance_start=infusenet_guidance_start,
            infusenet_guidance_end=infusenet_guidance_end,
            width=width,
            height=height,
        )

    if output_image_path is None:
        output_image_path = (
            ROOT / "results" / "stage_one" / f"{id_image_path.stem}_{prompt[:50]}.png"
        )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--id_image_path", type=str, required=True)
    parser.add_argument("--output_image_path", type=str, default=None)
    parser.add_argument("--model_version", type=str, default="sim_stage1")
    parser.add_argument("--enable_realism_lora", action="store_true")
    parser.add_argument("--enable_anti_blur_lora", action="store_true")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--optimize_vram", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--infusenet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--infusenet_guidance_start", type=float, default=0.0)
    parser.add_argument("--infusenet_guidance_end", type=float, default=1.0)

    args = parser.parse_args()
    main(
        prompt=args.prompt,
        id_image_path=Path(args.id_image_path),
        output_image_path=Path(args.output_image_path) if args.output_image_path else None,
        model_version=args.model_version,
        enable_realism_lora=args.enable_realism_lora,
        enable_anti_blur_lora=args.enable_anti_blur_lora,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        optimize_vram=args.optimize_vram,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        infusenet_conditioning_scale=args.infusenet_conditioning_scale,
        infusenet_guidance_start=args.infusenet_guidance_start,
        infusenet_guidance_end=args.infusenet_guidance_end,
    )
