DOCKER_IMAGE_NAME = avatar-product
DOCKER_TAG = latest

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

docker-run:
	docker run -it --gpus all -v $(PWD):/workspace $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) /bin/bash

sync-submodules:
	git submodule update --init --recursive
