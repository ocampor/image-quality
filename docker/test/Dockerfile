ARG PYTHON_DOCKER_VERSION
ARG WORKDIR=/src

FROM python:${PYTHON_DOCKER_VERSION} as base
ARG WORKDIR
# Install libsvm
RUN apt-get update && apt-get install -y g++
# OpenCV dependencies
RUN apt-get install -y libglib2.0-0 libsm6 libfontconfig1 libxrender1 libxext6
# Install package
ADD imquality ${WORKDIR}/imquality
ADD setup.py README.rst MANIFEST.in ${WORKDIR}/
WORKDIR ${WORKDIR}
RUN pip install -e .[dev]

FROM base as tests
ARG WORKDIR
ADD tests ${WORKDIR}/tests
ADD scripts/run-tests.sh ${WORKDIR}/
CMD ["/bin/bash", "run-tests.sh"]
