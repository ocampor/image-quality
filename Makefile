git-commit := `git rev-parse --short HEAD`
version := `python -c "import imquality; print(imquality.__version__)"`
version-dev := ${version}-a${git-commit}
python-docker-version := 3.7.4-slim

build-images:
	@echo "Building ocampor/image-quality:${version-dev} with python version ${python-docker-version}"
	@env VERSION=${version-dev} docker-compose build \
			--build-arg PYTHON_DOCKER_VERSION=${python-docker-version} \
			image-quality

run-tests: build-images
	@env VERSION=${version-dev} docker-compose -f docker-compose.yml -f docker-compose.test.yml run image-quality
