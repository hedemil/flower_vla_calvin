FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ===============================
# System deps
# ===============================
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
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

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python -m pip install --upgrade pip setuptools wheel

# ===============================
# MuJoCo rendering config
# ===============================
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# ===============================
# Install PyTorch first as a separate cached layer (CUDA 12.8)
# ===============================
RUN pip install torch==2.10.0 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu128

# ===============================
# Copy project
# ===============================
WORKDIR /workspace/flower_vla_calvin
COPY . .

# ===============================
# Install main requirements (matches Leonardo environment)
# ===============================
RUN pip install -r requirements_leonardo.txt

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
# Create LIBERO config file
# ===============================
RUN mkdir -p /appuser/.libero && \
    echo "benchmark_root: /workspace/flower_vla_calvin/LIBERO/libero/libero" > /appuser/.libero/config.yaml && \
    echo "bddl_files: /workspace/flower_vla_calvin/LIBERO/libero/libero/bddl_files" >> /appuser/.libero/config.yaml && \
    echo "init_states: /workspace/flower_vla_calvin/LIBERO/libero/libero/init_files" >> /appuser/.libero/config.yaml && \
    echo "datasets: /workspace/flower_vla_calvin/LIBERO/libero/datasets" >> /appuser/.libero/config.yaml && \
    echo "assets: /workspace/flower_vla_calvin/LIBERO/libero/libero/assets" >> /appuser/.libero/config.yaml && \
    echo "✓ LIBERO config created"

# ===============================
# Install LIBERO (no-deps to avoid overriding pinned versions)
# ===============================
WORKDIR /workspace/flower_vla_calvin/LIBERO
RUN pip install --no-deps -e . && \
    echo "✓ LIBERO installed"

# ===============================
# Install pyhash from source (CALVIN dependency, requires old setuptools)
# ===============================
WORKDIR /workspace/flower_vla_calvin
RUN pip install setuptools==57.5.0 && \
    cd pyhash-0.9.3 && \
    python setup.py build && \
    python setup.py install && \
    cd .. && \
    pip install --upgrade setuptools

# ===============================
# Environment variables
# ===============================
ENV flower_calvin_ROOT=/workspace/flower_vla_calvin
ENV PYTHONPATH=/workspace/flower_vla_calvin/LIBERO:${PYTHONPATH}
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HOME=/appuser

# Verify LIBERO can be imported
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
