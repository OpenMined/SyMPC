FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y git gcc
RUN pip install notebook
RUN pip install git+https://github.com/openmined/PySyft@dev#subdirectory=packages/syft

WORKDIR /notebooks
