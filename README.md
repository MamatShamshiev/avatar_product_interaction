# Avatar-Product Interaction Image Generation

![Avatar Product Interaction](assets/teaser.png)

## Table of Contents
- [Introduction](#introduction)
- [Pipeline Overview](#pipeline-overview)
  - [Stage 1: Identity-Conditioned Image Generation](#stage-1-identity-conditioned-image-generation)
  - [Stage 2: Product-Integrated Image Editing](#stage-2-product-integrated-image-editing)
- [Limitations](#limitations)
- [Further Application: Video Generation](#further-application-video-generation)
- [Alternative Approaches Considered](#alternative-approaches-considered)
- [Installation](#installation)
  - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
  - [Option 2: Conda Environment](#option-2-conda-environment)
- [Memory Requirements](#memory-requirements)

## Introduction

This repository contains a pipeline designed to automatically generate realistic images of an avatar interacting with a product. Given an input product image and a portrait of a person, the solution synthesizes high-quality images where the avatar naturally uses or showcases the product.

A gallery of examples is available in the [assets/demo](assets/demo) folder.

## Pipeline Overview

The implemented pipeline leverages open-source models and operates in two sequential stages.

![Pipeline Overview](assets/pipeline.png)

### Stage 1: Identity-Conditioned Image Generation

This stage generates a base image of the avatar while preserving identity features. The model takes an ID image of the person and a text prompt describing the desired appearance, pose, or context. It synthesizes a high-quality image where the avatar closely resembles the provided ID while adhering to the textual description.

This stage is built upon the [InfiniteYou](https://github.com/bytedance/InfiniteYou) framework, leveraging DiTs (FLUX) for flexible, high-fidelity, and identity-preserved image generation. The process requires a reference identity image and a descriptive textual prompt that specifies the desired appearance, pose, and context, especially tailored to showcase or interact with the intended product.

Guidelines for Optimal Results:

- Prompting: Create prompts clearly specifying how the avatar should showcase, hold, or wear the product. Good images are generally produced immediately, but if the product is not distinctly visible, provide more detailed prompts explicitly mentioning its position or interaction.

- Adjustments: Typically, significant adjustments are unnecessary. However, if the generated image does not align closely with the provided prompt, try increasing `--infusenet_guidance_start` slightly (e.g., set to 0.1). If results are still unsatisfactory, slightly decrease `--infusenet_conditioning_scale` (e.g., set to 0.9).

Example command:
```bash
python -m scripts.generate_id_image --id_image_path data/avatars/1.png --prompt 'A young woman wearing a t-shirt on a monotone background, 4K, high quality, photorealistic' --output_image_path 'results/stage_one/1_t-shirt.png' --optimize_vram
```

### Stage 2: Product-Integrated Image Editing

![Stage 2](assets/stage_2.png)

In this stage, the generated avatar image is refined to incorporate the target product. The stage receives the output from Stage 1, along with the product name and an additional target text prompt.

This stage leverages the approach described in the paper ["Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator"](https://arxiv.org/abs/2411.15466) using Diptych Prompting. This method reframes subject-driven image generation as an inpainting task, employing large-scale text-to-image models for precise, zero-shot integration and alignment of subjects (products in this context).

This stage receives the base avatar image (output from Stage 1), a reference product image, the product name, and a descriptive text prompt. Leveraging the product name, the pipeline utilizes [Grounding DINO](https://arxiv.org/abs/2303.05499) and [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) to detect and segment the product in both the reference product image and the base avatar image.
Next, a diptych image is created. The left panel contains the reference product image, while the right panel holds the base avatar image with a masked region indicating the area for inpainting. FLUX, equipped with a ControlNet module, then performs text-conditioned inpainting on the masked area in the right panel, referencing the product from the left panel.

Guidelines for Optimal Results:
- If the integrated product's appearance differs significantly from the reference image, consider increasing the `attn_enforce` parameter slightly (e.g., set to 1.1) to reinforce product alignment. Note that setting this parameter to a value other than 1.0 increases memory usage, as flash attention implementation becomes unavailable and a custom attention implementation is used instead.

- If insufficient space is available in the base avatar image to incorporate the product naturally, increase the `context_px` parameter to provide more area for inpainting the product seamlessly.

- By default, avatar identity preservation is enforced by preventing inpainting over facial regions. However, for certain products like glasses, you can disable this restriction using the `--mask_face` flag.

- For optimal results, generating multiple images using different random seeds can be beneficial, allowing selection from various outputs.

Example command:
```bash
PYTHONPATH='thirdparty/DiptychPrompting/' python -m scripts.integrate_product --product_image_path data/products/t-shirt-2.jpeg --id_image_path results/stage_one/1_t-shirt.png --product_name 't-shirt' --target_prompt 'a woman wearing a t-shirt' --output_image_path 'results/stage_two/1_t-shirt.png' --optimize_vram
```


## Limitations
- **Product Appearance Variations**: The generated product may not be an exact replica of the reference image. For instance, subtle details — such as logos, textures, or small design elements — might differ, as the pipeline relies on zero-shot subject-driven image generation rather than direct copy-pasting of the product.
Training the model with additional subject-driven adaptation methods or integrating more explicit structure-preserving techniques could improve product fidelity.

- **Challenges with Face-Worn Products**: Items like glasses, which require precise integration with facial features, pose a challenge. Since the second stage does not have direct access to the original ID image, inpainting over the face can alter the avatar's identity. This is an inherent limitation of the pipeline's current structure.

Examples of these limitations:
1. The Baby Yoda design in the [generated image](assets/demo/1/mug.png) may differ slightly from the [original product](data/products/mug.webp).
2. The hat in the [generated image](assets/demo/2/hat.png) differs in design from the [original product](data/products/hat.webp).
3. Intricate patterns on a t-shirt on the [generated image](assets/demo/1/t-shirt.png) might look slightly different from the [original product](data/products/t-shirt.webp).
4. The sunglasses in the [generated image](assets/demo/2/sunglasses.png) show slight variations from the [original product](data/products/glasses.jpg).


## Further Application: Video Generation
The images generated by this pipeline can serve as inputs for subsequent video generation, enabling the creation of dynamic avatar-product interaction sequences. While video generation is not the focus of this repository, this pipeline provides a strong foundation by ensuring high-quality identity-preserved static images.

As an example, here is a [short video](assets/demo/videos/1_jacket.mp4) generated using [MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance](https://github.com/tencent/MimicMotion), where the [input image](assets/demo/1/jacket.png) was generated by this pipeline.

Note: In this example, the identity of the person in the video is poorly preserved, highlighting a common challenge in motion-based generation. However, this demonstrates how the current pipeline can serve as a first step for future video-based applications, though additional techniques may be required to maintain identity consistency.

## Alternative Approaches Considered
During the development of this pipeline, multiple open-source models and frameworks were tested. Below are three of the most notable alternatives considered:

### PhotoMaker

[PhotoMaker](https://github.com/TencentARC/PhotoMaker) is a tuning-free text-to-image model that enables personalized generation without additional fine-tuning. It takes a few images of a person and a text prompt, utilizing a pretrained identity encoder and LoRA on top of Stable Diffusion [RealVisXL_V4.0](https://github.com/TencentARC/RealVisXL) checkpoint. It natively supports IP-Adapter, T2I-Adapter, ControlNet, and LoRAs.

During initial testing, the image quality and identity preservation were reasonable, but since it is based on Stable Diffusion XL, it performed worse than FLUX, which was ultimately selected for this pipeline..
Additionally, various conditioning techniques, such as ControlNet keypoints and IP-Adapter, were explored to improve control over generation, but the results remained unsatisfactory.

### InteractDiffusion

[InteractDiffusion](https://github.com/jiuntian/interactdiffusion) is a model that operates on a triplet label (person, interaction, subject). The user additionally provides bounding boxes for the person and subject, and the framework infers the interaction region. It offers good control over scene composition, making it useful for structured interactions.
However, it lacks out-of-the-box mechanisms for identity or subject conditioning and, like PhotoMaker, it is based on Stable Diffusion XL.

### OminiControl

[OminiControl](https://github.com/Yuanshi9815/OminiControl) is a flexible control framework built specifically for Diffusion Transformer (DiT) models, offering support for both subject-driven control and spatial control.
Since it is based on FLUX, it inherits the advantages of high-quality generation.

During the initial testing, preservation was decent for simple objects like plain t-shirts (without prints) and basic mugs (without text).

However, for more complex objects, such as t-shirts with patterns or phones, the details were poorly preserved, making it unsuitable for the required product integration task.

## Installation

First, clone the submodules:
```bash
make sync-submodules
```

Next, there are two options for setting up the environment:

### Option 1: Docker (Recommended)
```bash
make docker-build
make docker-run
```

### Option 2: Conda Environment
```bash
conda create -n avatar_product python=3.10
conda activate avatar_product

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt
```

### Login to Hugging Face
Finally, login to Hugging Face to be able to download the models:
```bash
huggingface-cli login --token $HF_TOKEN
```

## Memory Requirements

The pipeline requires significant GPU memory, particularly for the Flux model.
The pipeline requires 40GB of VRAM with the `--optimize_vram` flag enabled.
If you don't have enough VRAM, you may reduce the height and width.
