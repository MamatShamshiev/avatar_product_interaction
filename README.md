# Avatar Product Interaction

## Installation

First, make sure to clone the submodules:
```bash
make sync-submodules
```

### Conda Environment
```bash
conda create -n avatar_product python=3.10
conda activate avatar_product

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install -r thirdparty/InfiniteYou/requirements.txt
```

### Docker
```bash
make docker-build
make docker-run
```

### Login to Hugging Face
```bash
huggingface-cli login --token $HF_TOKEN
```


## First Stage: Identity-Conditioned Image Generation

This stage generates a base image of the avatar while preserving identity features. The model takes an ID image of the person and a text prompt describing the desired appearance, pose, or context. It synthesizes a high-quality image where the avatar closely resembles the provided ID while adhering to the textual description.

```bash
python -m scripts.generate_id_image --id_image_path data/avatars/1.png --prompt 'A young woman wearing a t-shirt on a monotone background, 4K, high quality, photorealistic' --output_image_path 'results/stage_one/1_t-shirt.png'
```

Note: if your GPU memory is limited, you can use the `--optimize_vram` flag to reduce the memory usage or set lower height and width.

## Second Stage: Product-Integrated Image Editing

In this stage, the generated avatar image is refined to incorporate the target product. The model receives the output from Stage 1, along with the product name and an additional text prompt. It seamlessly modifies the image, ensuring that the product appears naturally positioned and integrated into the scene while maintaining realism and consistency.

```bash
PYTHONPATH='thirdparty/DiptychPrompting/' python -m scripts.integrate_product --product_image_path data/products/t-shirt-2.jpeg --id_image_path results/stage_one/1_t-shirt.png --subject_name 't-shirt' --target_prompt 'a woman wearing a t-shirt' --output_image_path 'results/stage_two/1_t-shirt.png'
```

Note: if your GPU memory is limited, you can use the `--optimize_vram` flag to reduce the memory usage or set lower height and width.