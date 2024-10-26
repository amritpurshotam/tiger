# full list of supported tags: https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
# latest pytorch requirements: https://pytorch.org/get-started/locally/
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# installs python 3.10.12 at time of writing
RUN apt-get update -y \
    && apt-get -y install software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get -y install python3.13 python3.13-distutils python3.13-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && apt-get -y install python3-pip

WORKDIR /app/

COPY requirements.txt scripts/requirements-nb.txt /app/

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-nb.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--ServerApp.allow_origin='*'"]