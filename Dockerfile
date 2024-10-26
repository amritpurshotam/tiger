# full list of supported tags: https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
# latest pytorch requirements: https://pytorch.org/get-started/locally/
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update -y
RUN apt-get -y install python3 \ 
    && apt-get -y install python3-pip

WORKDIR /app/

COPY scripts/ /app/scripts

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r scripts/requirements-nb.txt

COPY requirements.txt setup.py /app/

RUN pip install --no-cache-dir -e .

EXPOSE 8888

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--ServerApp.allow_origin='*'"]