import argparse
import random

import cv2
import numpy as np
import torch
from controlnet_flux import FluxControlNetModel
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from PIL import Image
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from thirdparty.DiptychPrompting.diptych_prompting_inference import (
    DetectionResult,
    grounded_segmentation,
)
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from src.attn_processor import CustomFluxAttnProcessor
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image_path", type=str, required=True)
    parser.add_argument("--id_image_path", type=str, required=True)
    parser.add_argument("--subject_name", type=str, required=True)
    parser.add_argument("--target_prompt", type=str, required=True)
    parser.add_argument("--attn_enforce", type=float, default=1.0)
    parser.add_argument("--ctrl_scale", type=float, default=0.95)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--pixel_offset", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--context_px", type=int, default=0)
    parser.add_argument("--mask_face", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
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

    face_model = FaceAnalysis(
        name="antelopev2",
        root=str(ROOT / "models" / "InfiniteYou" / "supports" / "insightface"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_model.prepare(ctx_id=0, det_size=(640, 640))

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

    def mask_id_image(
        image, object_name, face_bbox: np.ndarray | None = None, context_px: int = 0
    ) -> tuple[Image.Image, np.ndarray]:
        detection = detect(object_detector, image, object_name, threshold=0.3)
        object_box = detection.box.xyxy
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        x1, y1, x2, y2 = object_box
        mask[y1 - context_px : y2 + context_px, x1 - context_px : x2 + context_px] = 1

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox.astype(np.int32)
            mask[y1:y2, x1:x2] = 0

        image_array = np.array(image)
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
    # diptych_text_prompt = f"A diptych with two side-by-side images of same {subject_name}. On the left, {base_prompt}. On the right, {args.target_prompt}"
    diptych_text_prompt = f"The two-panel image showcases the same {subject_name}. [LEFT] the left panel is showing the {subject_name}. [RIGHT] the right panel is showing the {subject_name} as {args.target_prompt}"

    reference_image = load_image(args.ref_image_path).resize((width, height)).convert("RGB")
    id_image = load_image(args.id_image_path).resize((width, height)).convert("RGB")

    reference_image_segmented, _ = segment_image(reference_image, subject_name)
    reference_image_segmented.save(results_dir / "reference_image_segmented.png")

    if not args.mask_face:
        face_infos = face_model.get(cv2.cvtColor(np.array(id_image), cv2.COLOR_RGB2BGR))
        face_info = max(
            face_infos, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1])
        )
        face_bbox = face_info["bbox"].astype(np.int32)
    else:
        face_bbox = None

    id_image_masked, mask = mask_id_image(
        id_image, subject_name, face_bbox=face_bbox, context_px=args.context_px
    )
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
    )
    pipe.load_lora_weights(
        "ali-vilab/In-Context-LoRA", weight_name="visual-identity-design.safetensors"
    )

    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    pipe.to("cuda")

    if args.attn_enforce != 0.0:
        new_attn_procs = {}
        for k in pipe.transformer.attn_processors:
            new_attn_procs[k] = CustomFluxAttnProcessor(
                height=height // 16, width=width // 16 * 2, attn_enforce=args.attn_enforce
            )
        pipe.transformer.set_attn_processor(new_attn_procs)

    seed = random.randint(0, 2**32 - 1) if args.seed == -1 else args.seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
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
