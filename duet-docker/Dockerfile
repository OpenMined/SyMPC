FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y git gcc
RUN pip install notebook
RUN pip install git+https://github.com/openmined/PySyft@dev#subdirectory=packages/syft

WORKDIR /notebooks
ARG PORT
ENV PORT "${PORT}"
ENV PORT=${PORT:-8888}

ENTRYPOINT jupyter notebook --port="$PORT" --ip=0.0.0.0 --allow-root
