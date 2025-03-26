import argparse

import numpy as np
import torch
from controlnet_flux import FluxControlNetModel
from diffusers.utils import load_image
from PIL import Image
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from thirdparty.DiptychPrompting.diptych_prompting_inference import (
    DetectionResult,
    grounded_segmentation,
)
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from src.defs import ROOT


def detect(
    object_detector,
    image: Image.Image,
    label: str,
    threshold: float = 0.3,
    detector_id: str | None = None,
) -> DetectionResult:
    """Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion."""
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    results = object_detector(image, candidate_labels=[f"{label}."], threshold=threshold)
    result = max(
        (DetectionResult.from_dict(result) for result in results),
        key=lambda x: x.score,
    )

    return result


def grounded_detection(
    detect_pipeline,
    image: Image.Image | str,
    label: str,
    threshold: float = 0.3,
    detector_id: str | None = None,
    context_px: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(image, str):
        image = load_image(image)

    detection = detect(detect_pipeline, image, label, threshold, detector_id)
    box = detection.box.xyxy

    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1 - context_px : y2 + context_px, x1 - context_px : x2 + context_px] = 1

    return np.array(image), mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image_path", type=str, required=True)
    parser.add_argument("--id_image_path", type=str, required=True)
    parser.add_argument("--subject_name", type=str, required=True)
    parser.add_argument("--target_prompt", type=str, required=True)
    parser.add_argument("--attn_enforce", type=float, default=1.0)
    parser.add_argument("--ctrl_scale", type=float, default=0.95)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--pixel_offset", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=30)

    args = parser.parse_args()

    results_dir = ROOT / "results" / "diptych"
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

    def segment_image(image, object_name, reverse: bool = False) -> tuple[Image.Image, np.ndarray]:
        image_array, detections = grounded_segmentation(
            object_detector,
            segmentator,
            segment_processor,
            image=image,
            labels=[object_name],
            threshold=0.3,
            polygon_refinement=True,
        )
        detection = max(detections, key=lambda x: x.score)
        mask = detection.mask / 255

        mask = np.expand_dims(mask, axis=-1)
        if reverse:
            segment_result = image_array * (1 - mask)
        else:
            segment_result = image_array * mask + np.ones_like(image_array) * (1 - mask) * 255
        segmented_image = Image.fromarray(segment_result.astype(np.uint8))
        return segmented_image, mask.squeeze(-1)

    def segment_image_w_bbox(image, object_name) -> tuple[Image.Image, np.ndarray]:
        image_array, mask = grounded_detection(
            object_detector,
            image=image,
            label=object_name,
            threshold=0.3,
        )
        mask = np.expand_dims(mask, axis=-1)
        segment_result = image_array * (1 - mask) + np.ones_like(image_array) * mask * 128
        segmented_image = Image.fromarray(segment_result.astype(np.uint8))
        return segmented_image, mask.squeeze(-1)

    def make_diptych(ref_image, id_image) -> Image.Image:
        ref_image = np.array(ref_image)
        id_image = np.array(id_image)
        diptych = np.concatenate([ref_image, id_image], axis=1)
        diptych = Image.fromarray(diptych)
        return diptych

    # Load image and mask
    width = args.width + args.pixel_offset * 2
    height = args.height + args.pixel_offset * 2
    size = (width * 2, height)

    subject_name = args.subject_name
    base_prompt = f"a photo of {subject_name}"
    diptych_text_prompt = f"A diptych with two side-by-side images of same {subject_name}. On the left, {base_prompt}. On the right, {args.target_prompt}"

    reference_image = load_image(args.ref_image_path).resize((width, height)).convert("RGB")
    id_image = load_image(args.id_image_path).resize((width, height)).convert("RGB")

    reference_image_segmented, _ = segment_image(reference_image, subject_name)
    reference_image_segmented.save(results_dir / "reference_image_segmented.png")

    # id_image_masked, mask = segment_image_w_bbox(id_image, subject_name)
    id_image_masked, mask = segment_image(id_image, subject_name, reverse=True)
    id_image_masked.save(results_dir / "id_image_masked.png")

    diptych_image_prompt = make_diptych(reference_image_segmented, id_image_masked)
    diptych_image_prompt.save(results_dir / "diptych_image_prompt.png")
    mask_image = np.concatenate([np.zeros((height, width)), mask * 255], axis=1)
    mask_image = Image.fromarray(mask_image.astype(np.uint8)).convert("RGB")
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
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    base_attn_procs = pipe.transformer.attn_processors.copy()

    # new_attn_procs = base_attn_procs.copy()
    # for i, (k, v) in enumerate(new_attn_procs.items()):
    #     new_attn_procs[k] = CustomFluxAttnProcessor2_0(
    #         height=height // 16, width=width // 16 * 2, attn_enforce=args.attn_enforce
    #     )
    # pipe.transformer.set_attn_processor(new_attn_procs)

    generator = torch.Generator(device="cuda").manual_seed(42)
    # Inpaint
    result = pipe(
        prompt=diptych_text_prompt,
        height=size[1],
        width=size[0],
        control_image=diptych_image_prompt,
        control_mask=mask_image,
        num_inference_steps=args.num_steps,
        generator=generator,
        controlnet_conditioning_scale=args.ctrl_scale,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=3.5,
    ).images[0]

    result = result.crop((width, 0, width * 2, height))
    result = result.crop(
        (
            args.pixel_offset,
            args.pixel_offset,
            width - args.pixel_offset,
            height - args.pixel_offset,
        )
    )
    result.save(results_dir / "result.png")
