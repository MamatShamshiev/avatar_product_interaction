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


## First Stage: Avatar pose generation

```bash
python -m scripts.infu_flux --id_image_path /path/to/avatar.png --prompt 'A young woman wearing a t-shirt on a monotone background, 4K, high quality, photorealistic'
```

## Second Stage: Product-Driven image editing

```bash
PYTHONPATH='thirdparty/DiptychPrompting/' python -m scripts.diptych --ref_image_path /path/to/products/t-shirt.png --id_image_path /path/to/stage1/output.png --subject_name 't-shirt' --target_prompt 'a woman wearing a t-shirt'
```