FROM python:3.7-slim-buster

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install git clang build-essential

COPY ./ SyMPC/

WORKDIR /SyMPC

RUN pip3 install setuptools_scm
RUN pip3 install -r requirements.txt && pip3 install -r requirements.dev.txt
RUN pip3 install -e .

ENTRYPOINT [ "/bin/bash" ]
