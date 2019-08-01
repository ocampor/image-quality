git-commit := `git rev-parse --short HEAD`
version := 1.0.1-a${git-commit}
python-docker-version := 3.7.4-slim

build-images:
	@echo "Building ocampor/image-quality:${VERSION} with python version ${python-docker-version}"
	@env VERSION=${version} docker-compose build \
			--build-arg VERSION=${version} \
			--build-arg PYTHON_DOCKER_VERSION=${python-docker-version} \
			image-quality

run-test: build-images
	@env VERSION=${version} docker-compose run pytest
