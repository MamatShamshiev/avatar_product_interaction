import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from controlnet_flux import FluxControlNetModel
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from PIL import Image
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from src.attn_processor import CustomFluxAttnProcessor
from src.defs import ROOT
from src.integrate_product import make_diptych, mask_id_image, segment_image


def main(
    product_image_path: Path,
    id_image_path: Path,
    subject_name: str,
    target_prompt: str,
    output_image_path: Path | None = None,
    width: int = 768,
    height: int = 1024,
    num_steps: int = 30,
    context_px: int = 0,
    mask_face: bool = False,
    seed: int = -1,
    attn_enforce: float = 1.0,
    ctrl_scale: float = 0.95,
    guidance_scale: float = 3.5,
    true_guidance_scale: float = 3.5,
    save_debug_images: bool = False,
    optimize_vram: bool = False,
):
    if output_image_path is None:
        results_dir = ROOT / "results" / "stage_two"
        output_image_path = results_dir / f"{subject_name}.png"
    else:
        results_dir = output_image_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    detector_id = "IDEA-Research/grounding-dino-base"
    segmenter_id = "facebook/sam-vit-large"
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to("cuda")
    segment_processor = AutoProcessor.from_pretrained(segmenter_id)
    object_detector = pipeline(
        model=detector_id,
        task="zero-shot-object-detection",
        device=torch.device("cuda"),
    )

    face_model = FaceAnalysis(
        name="antelopev2",
        root=str(ROOT / "models" / "InfiniteYou" / "supports" / "insightface"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_model.prepare(ctx_id=0, det_size=(640, 640))

    diptych_size = (width * 2, height)
    diptych_text_prompt = (
        f"The two-panel image showcases the same {subject_name}. "
        f"[LEFT] the left panel is showing the {subject_name}. "
        f"[RIGHT] the right panel is showing the same {subject_name} but as {target_prompt}"
    )

    product_image = load_image(str(product_image_path)).resize((width, height)).convert("RGB")
    id_image = load_image(str(id_image_path)).resize((width, height)).convert("RGB")

    product_image_segmented, _ = segment_image(
        product_image,
        object_detector,
        segmentator,
        segment_processor,
        subject_name,
    )
    if save_debug_images:
        product_image_segmented.save(results_dir / "product_image_segmented.png")

    if not mask_face:
        face_infos = face_model.get(cv2.cvtColor(np.array(id_image), cv2.COLOR_RGB2BGR))
        face_info = max(
            face_infos, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1])
        )
        face_bbox = face_info["bbox"].astype(np.int32)
    else:
        face_bbox = None

    id_image_masked, mask = mask_id_image(
        id_image,
        object_detector,
        subject_name,
        face_bbox=face_bbox,
        context_px=context_px,
    )
    if save_debug_images:
        id_image_masked.save(results_dir / "id_image_masked.png")

    diptych_image_prompt = make_diptych(product_image_segmented, id_image_masked)
    if save_debug_images:
        diptych_image_prompt.save(results_dir / "diptych_image_prompt.png")

    mask_image = np.concatenate([np.zeros((height, width)), mask * 255], axis=1)
    mask_image = Image.fromarray(mask_image.astype(np.uint8)).convert("RGB")
    if save_debug_images:
        mask_image.save(results_dir / "mask_image.png")

    # Build pipeline
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(
        "ali-vilab/In-Context-LoRA", weight_name="visual-identity-design.safetensors"
    )
    if attn_enforce != 1.0:
        new_attn_procs = {}
        for k in pipe.transformer.attn_processors:
            new_attn_procs[k] = CustomFluxAttnProcessor(
                height=height // 16, width=width // 16 * 2, attn_enforce=args.attn_enforce
            )
        pipe.transformer.set_attn_processor(new_attn_procs)

    if optimize_vram:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=diptych_text_prompt,
        height=diptych_size[1],
        width=diptych_size[0],
        control_image=diptych_image_prompt,
        control_mask=mask_image,
        num_inference_steps=num_steps,
        generator=generator,
        controlnet_conditioning_scale=ctrl_scale,
        guidance_scale=guidance_scale,
        negative_prompt="",
        true_guidance_scale=true_guidance_scale,
    ).images[0]

    result = result.crop((width, 0, width * 2, height))
    result.save(output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product_image_path", type=str, required=True)
    parser.add_argument("--id_image_path", type=str, required=True)
    parser.add_argument("--subject_name", type=str, required=True)
    parser.add_argument("--target_prompt", type=str, required=True)
    parser.add_argument("--output_image_path", type=str, default=None)
    parser.add_argument("--attn_enforce", type=float, default=1.0)
    parser.add_argument("--ctrl_scale", type=float, default=0.95)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--context_px", type=int, default=0)
    parser.add_argument("--mask_face", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--true_guidance_scale", type=float, default=3.5)
    parser.add_argument("--save_debug_images", action="store_true")
    parser.add_argument("--optimize_vram", action="store_true")
    args = parser.parse_args()

    main(
        product_image_path=Path(args.product_image_path),
        id_image_path=Path(args.id_image_path),
        subject_name=args.subject_name,
        target_prompt=args.target_prompt,
        output_image_path=Path(args.output_image_path) if args.output_image_path else None,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        context_px=args.context_px,
        mask_face=args.mask_face,
        seed=args.seed,
        attn_enforce=args.attn_enforce,
        ctrl_scale=args.ctrl_scale,
        guidance_scale=args.guidance_scale,
        true_guidance_scale=args.true_guidance_scale,
        save_debug_images=args.save_debug_images,
        optimize_vram=args.optimize_vram,
    )
