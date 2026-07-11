FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates pkg-config \
      libglfw3-dev libgl1-mesa-dev libgl1-mesa-dri mesa-utils \
      libx11-dev \
      python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*
# PPO track: CPU torch (a 5k-param net needs no GPU; CPU wheels keep the image lean-ish).
# Versions unpinned by design — pin in this file if a torch/numpy bump ever breaks training.
RUN pip3 install --no-cache-dir numpy && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Dev-machine black-window workaround (imgui GPU fallback; inherited from ui_pong).
COPY glx_present_shim.c /opt/glx_present_shim.c
RUN cc -shared -fPIC /opt/glx_present_shim.c -o /usr/local/lib/glx_present_shim.so
WORKDIR /work
