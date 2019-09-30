FROM python:3.7 AS base
# Install requirements
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
# Set user
RUN useradd --create-home --shell /bin/bash -G root ocampor
USER ocampor
WORKDIR /home/ocampor
# Start jupyter notebook
EXPOSE 8000
ADD entrypoint.sh entrypoint.sh
CMD ["/bin/bash", "entrypoint.sh"]

FROM base AS tensorflow
USER root
RUN pip install tensorflow==2.0.0
USER ocampor
