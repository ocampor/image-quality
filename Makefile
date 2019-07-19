include .env

version=${VERSION}
python-docker-version=3.7.4-slim

build-images:
	echo `Building ocampor/image-quality:${version} with python version ${python-docker-version}`
	docker-compose build \
		--build-arg VERSION=${version} \
		--build-arg PYTHON_DOCKER_VERSION=${python-docker-version} \
		image-quality