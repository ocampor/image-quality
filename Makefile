git-commit := `git rev-parse --short HEAD`
version := `python -c "import imquality; print(imquality.__version__)"`
version-dev := ${version}-a${git-commit}
python-docker-version := 3.7.4-slim

build-images:
	@echo "Building ocampor/image-quality:${version-dev} with python version ${python-docker-version}"
	@env VERSION=${version-dev} PYTHON_DOCKER_VERSION=${python-docker-version} docker-compose build image-quality

run-tests:
	@echo "Running tests with python version ${python-docker-version}"
	@env VERSION=${version-dev} PYTHON_DOCKER_VERSION=${python-docker-version} \
		docker-compose -f docker-compose.yml -f docker-compose.test.yml build image-quality
	@env VERSION=${version-dev} PYTHON_DOCKER_VERSION=${python-docker-version} \
		docker-compose -f docker-compose.yml -f docker-compose.test.yml run image-quality
