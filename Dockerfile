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
RUN pip install torch==2.6.0 torchvision torchaudio \
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
# Create LIBERO config file first
# ===============================
RUN mkdir -p /root/.libero && \
    echo "benchmark_root: /workspace/flower_vla_calvin/LIBERO/libero/libero" > /root/.libero/config.yaml && \
    echo "bddl_files: /workspace/flower_vla_calvin/LIBERO/libero/libero/bddl_files" >> /root/.libero/config.yaml && \
    echo "init_states: /workspace/flower_vla_calvin/LIBERO/libero/libero/init_files" >> /root/.libero/config.yaml && \
    echo "datasets: /workspace/flower_vla_calvin/LIBERO/libero/datasets" >> /root/.libero/config.yaml && \
    echo "assets: /workspace/flower_vla_calvin/LIBERO/libero/libero/assets" >> /root/.libero/config.yaml && \
    echo "✓ LIBERO config created"

# ===============================
# Install LIBERO requirements first (to get dependencies)
# ===============================
WORKDIR /workspace/flower_vla_calvin/LIBERO
RUN echo "Installing LIBERO requirements..." && \
    pip install -r requirements.txt && \
    pip install -e . && \
    pip install numpy~=1.23 && \
    echo "✓ LIBERO requirements installed"

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
RUN pip install transformers==4.46.3

# ===============================
# Set working directory for remaining installations
# ===============================
WORKDIR /workspace/flower_vla_calvin

# ===============================
# Environment variables
# ===============================
ENV flower_calvin_ROOT=/workspace/flower_vla_calvin
ENV PYTHONPATH=/workspace/flower_vla_calvin/LIBERO:${PYTHONPATH}

# Verify LIBERO can be imported (from same working directory)
RUN echo "Verifying LIBERO installation..." && \
    python -c "from libero.libero import benchmark, get_libero_path; print('✓ LIBERO imported successfully')" || \
    (echo "ERROR: LIBERO import failed!" && exit 1)

# ===============================
# Final verification of critical imports
# ===============================
RUN echo "Running final verification of critical packages..." && \
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" && \
    python -c "import calvin_env; print('✓ calvin_env')" && \
    python -c "from libero.libero import benchmark, get_libero_path; print('✓ LIBERO')" && \
    python -c "import mujoco; print(f'✓ MuJoCo {mujoco.__version__}')" && \
    echo "All critical packages verified successfully!"

CMD ["/bin/bash"]
