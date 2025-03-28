# Avatar Product Interaction

## Description

This repository contains a pipeline designed to automatically generate realistic images of an avatar interacting with a product. Given an input product image and a portrait of a person (extracted from video frames), the solution synthesizes high-quality images where the avatar naturally uses or showcases the product.


## Pipeline Overview

The implemented pipeline leverages open-source models and operates in two sequential stages.

### Stage 1: Identity-Conditioned Image Generation

This stage generates a base image of the avatar while preserving identity features. The model takes an ID image of the person and a text prompt describing the desired appearance, pose, or context. It synthesizes a high-quality image where the avatar closely resembles the provided ID while adhering to the textual description.

This stage is built upon the [InfiniteYou](https://github.com/bytedance/InfiniteYou) framework, leveraging DiTs (FLUX) for flexible, high-fidelity, and identity-preserved image generation. The process requires a reference identity image and a descriptive textual prompt that specifies the desired appearance, pose, and context, especially tailored to showcase or interact with the intended product.

Guidelines for Optimal Results:

- Prompting: Create prompts clearly specifying how the avatar should showcase, hold, or wear the product. Generally, good images are produced immediately, but if the product is not distinctly visible, provide more detailed prompts explicitly mentioning the product’s position or interaction.

- Adjustments: Typically, significant adjustments are unnecessary. However, if the generated image does not align closely with the provided prompt, try increasing `--infusenet_guidance_start` slightly (e.g., set to 0.1). If results are still unsatisfactory, slightly decrease `--infusenet_conditioning_scale` (e.g., set to 0.9).

Example command:
```bash
python -m scripts.generate_id_image --id_image_path data/avatars/1.png --prompt 'A young woman wearing a t-shirt on a monotone background, 4K, high quality, photorealistic' --output_image_path 'results/stage_one/1_t-shirt.png' --optimize_vram
```

### Stage 2: Product-Integrated Image Editing

In this stage, the generated avatar image is refined to incorporate the target product. The model receives the output from Stage 1, along with the product name and an additional text prompt. It seamlessly modifies the image, ensuring that the product appears naturally positioned and integrated into the scene while maintaining realism and consistency.

This stage leverages the approach described in the paper ["Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator"](https://arxiv.org/pdf/2411.15466) using [Diptych Prompting](https://diptychprompting.github.io/). This method reframes subject-driven image generation as an inpainting task, employing large-scale text-to-image models for precise, zero-shot integration and alignment of subjects (products in this context).

This stage receives the base avatar image (output from Stage 1), a reference product image, the product name, and a descriptive text prompt. Leveraging the product name, the pipeline utilizes Grounding DINO and Segment Anything (SAM) to segment the product in both the reference product image and the base avatar image.
Next, a diptych image is created. The left panel contains the reference product image, while the right panel holds the base avatar image with a masked region indicating the area for inpainting. FLUX, equipped with a ControlNet module, then performs text-conditioned inpainting on the masked area in the right panel, referencing the product from the left panel.

Guidelines for Optimal Results:
- If the integrated product's appearance differs significantly from the reference image, consider increasing the `attn_enforce` parameter slightly (e.g., set to 1.1) to reinforce product alignment.

- If insufficient space is available in the base avatar image to incorporate the product naturally, increase the `context_px` parameter to provide more area for inpainting the product seamlessly.

- By default, avatar identity preservation is enforced by preventing inpainting over facial regions. However, for certain products like glasses, you can disable this restriction using the `--mask_face` flag.

- For optimal results, generating multiple images using different random seeds can be beneficial, allowing selection from various outputs.

Example command:
```bash
PYTHONPATH='thirdparty/DiptychPrompting/' python -m scripts.integrate_product --product_image_path data/products/t-shirt-2.jpeg --id_image_path results/stage_one/1_t-shirt.png --product_name 't-shirt' --target_prompt 'a woman wearing a t-shirt' --output_image_path 'results/stage_two/1_t-shirt.png' --optimize_vram
```

### Memory Requirements

The pipeline requires significant GPU memory, particularly for the Flux model.
The pipeline requires 40GB of VRAM with enabled `--optimize_vram` flag.
If you don't have enough VRAM, you may set lower height and width.

## Installation

Make sure to clone the submodules:
```bash
make sync-submodules
```

### Docker (Recommended)
```bash
make docker-build
make docker-run
```

### Conda Environment
```bash
conda create -n avatar_product python=3.10
conda activate avatar_product

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt
```

### Login to Hugging Face
```bash
huggingface-cli login --token $HF_TOKEN
```
