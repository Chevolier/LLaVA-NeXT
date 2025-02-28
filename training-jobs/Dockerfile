FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV NVIDIA_VISIBLE_DEVICES="all"

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN MAX_JOBS=4 pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir