ARG PYTHON_DOCKER_VERSION

FROM python:${PYTHON_DOCKER_VERSION}

ARG VERSION
ENV VERSION ${VERSION}

# Install libsvm
RUN apt-get update && apt-get install -y git build-essential
ENV LIBSVM_PATH /opt/libsvm
RUN git clone 'https://github.com/cjlin1/libsvm.git' ${LIBSVM_PATH}
RUN make -C ${LIBSVM_PATH}
RUN make -C ${LIBSVM_PATH}/python
ENV PYTHONPATH ${PYTHONPATH}:${LIBSVM_PATH}/python

# OpenCV dependencies
RUN apt-get install -y libglib2.0-0 libsm6 libfontconfig1 libxrender1

COPY . /src
WORKDIR /src

# Install twine to validate dist package
RUN pip install --upgrade twine
RUN pip install -e .[dev]
RUN python setup.py sdist && python -m twine check dist/*
