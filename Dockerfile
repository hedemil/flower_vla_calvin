FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ===============================
# System deps
# ===============================
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libgomp1 \
    libegl1 \
    libegl1-mesa-dev \
    libglvnd0 \
    libglvnd-dev \
    libosmesa6 \
    libosmesa6-dev \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    freeglut3-dev \
    ffmpeg \
    mesa-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN python -m pip install --upgrade pip setuptools wheel

# ===============================
# MuJoCo rendering config
# ===============================
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# ===============================
# Base Python deps
# ===============================
RUN pip install \
    pillow \
    termcolor \
    tqdm \
    imageio \
    imageio-ffmpeg \
    moviepy \
    mujoco==2.3.7 \
    glfw

# Install PyTorch first (CUDA 11.8)
RUN pip install torch==2.2.2 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# ===============================
# Copy project and install submodules
# ===============================
WORKDIR /workspace/flower_vla_calvin
COPY . .

# ===============================
# Install tacto
# ===============================
WORKDIR /workspace/flower_vla_calvin/calvin_env/tacto
RUN pip install -e .

# ===============================
# Install calvin_env
# ===============================
WORKDIR /workspace/flower_vla_calvin/calvin_env
RUN pip install -e .

# ===============================
# Install LIBERO with specific requirements
# ===============================
WORKDIR /workspace/flower_vla_calvin/LIBERO
RUN pip install -r requirements.txt && \
    pip install -e . && \
    pip install numpy~=1.23

# ===============================
# Install pyhash with specific setuptools
# ===============================
WORKDIR /workspace/flower_vla_calvin
RUN pip install setuptools==57.5.0 && \
    cd pyhash-0.9.3 && \
    python setup.py build && \
    python setup.py install && \
    cd ..

# ===============================
# Install main project requirements
# ===============================
WORKDIR /workspace/flower_vla_calvin
RUN pip install -r requirements.txt

# ===============================
# Upgrade transformers for Florence-2 compatibility
# ===============================
RUN pip install --upgrade transformers>=4.37.0

# ===============================
# Environment variables
# ===============================
ENV flower_calvin_ROOT=/workspace/flower_vla_calvin

# Set default working directory
WORKDIR /workspace/flower_vla_calvin

CMD ["/bin/bash"]
