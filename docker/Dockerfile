ARG PYTHON_DOCKER_VERSION=3.7.4-slim

FROM python:${PYTHON_DOCKER_VERSION}

COPY . /src
WORKDIR /src

RUN pip install -e /src