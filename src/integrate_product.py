from typing import cast

import numpy as np
from PIL import Image
from thirdparty.DiptychPrompting.diptych_prompting_inference import (
    DetectionResult,
    grounded_segmentation,
)
from transformers import AutoModelForMaskGeneration, AutoProcessor, Pipeline


def segment_image(
    image: Image.Image,
    object_detector: Pipeline,
    segmentator: AutoModelForMaskGeneration,
    segment_processor: AutoProcessor,
    object_name: str,
    reverse: bool = False,
) -> tuple[Image.Image, np.ndarray]:
    image_array, detections = grounded_segmentation(
        object_detector,
        segmentator,
        segment_processor,
        image=image,
        labels=[object_name],
        threshold=0.3,
        polygon_refinement=True,
    )
    if len(detections) == 0:
        raise ValueError(f"No {object_name} found in the image")
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
    image: Image.Image,
    object_detector: Pipeline,
    object_name: str,
    face_bbox: np.ndarray | None = None,
    context_px: int = 0,
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


def make_diptych(ref_image: Image.Image, id_image: Image.Image) -> Image.Image:
    diptych = np.concatenate([np.array(ref_image), np.array(id_image)], axis=1)
    return Image.fromarray(diptych)


def detect(
    object_detector: Pipeline,
    image: Image.Image,
    label: str,
    threshold: float = 0.3,
    detector_id: str | None = None,
) -> DetectionResult:
    """Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion."""
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    results = cast(
        list[DetectionResult],
        object_detector(image, candidate_labels=[f"{label}."], threshold=threshold),
    )
    if len(results) == 0:
        raise ValueError(f"No {label} found in the image")
    detections = [DetectionResult.from_dict(result) for result in results]
    return max(detections, key=lambda x: x.score)
