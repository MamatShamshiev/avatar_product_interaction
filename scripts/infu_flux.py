import argparse
from pathlib import Path
from typing import Literal

import PIL.Image
import torch
from thirdparty.InfiniteYou.pipelines.pipeline_infu_flux import InfUFluxPipeline

from src.defs import ROOT


def main(
    prompt: str,
    id_image_path: Path,
    model_version: Literal["sim_stage1", "aes_stage2"] = "sim_stage1",
    enable_realism_lora: bool = True,
    enable_anti_blur_lora: bool = False,
    width: int = 864,
    height: int = 1152,
    num_steps: int = 50,
    optimize_vram: bool = False,
):
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

    guidance_scale = 3.5
    seed = 0
    infusenet_conditioning_scale = 1.0
    infusenet_guidance_start = 0.0
    infusenet_guidance_end = 1.0

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

        results_dir = ROOT / "results" / "InfiniteYou"
        results_dir.mkdir(parents=True, exist_ok=True)
        image.save(results_dir / f"{id_image_path.stem}_{prompt}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--id_image_path", type=str, required=True)
    parser.add_argument("--model_version", type=str, default="sim_stage1")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--optimize_vram", type=bool, action="store_true")
    args = parser.parse_args()
    breakpoint()
    main(
        prompt=args.prompt,
        id_image_path=Path(args.id_image_path),
        model_version=args.model_version,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        optimize_vram=args.optimize_vram,
    )
