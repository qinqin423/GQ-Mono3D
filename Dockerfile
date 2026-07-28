# 使用Ubuntu 22.04作为基础镜像（国内镜像源更稳定）
FROM ubuntu:22.04

# 设置环境变量避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 配置国内apt源（清华源）
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 1. 添加NVIDIA CUDA仓库的GPG密钥和仓库配置
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    ca-certificates && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

# 2. 安装完整的CUDA 12.1工具链（包括nvcc编译器）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    ca-certificates \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    cuda-toolkit-12-1 \
    cuda-compiler-12-1 \
    cuda-libraries-dev-12-1 \
    cuda-cudart-dev-12-1 \
    cuda-cupti-dev-12-1 \
    cuda-nvml-dev-12-1 \
    cuda-nvrtc-dev-12-1 \
    cuda-nvtx-12-1 \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 3. 创建符号链接，确保python和pip指向Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip

# 4. 配置CUDA环境变量
ENV PATH="/usr/local/cuda-12.1/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64"
ENV CUDA_HOME="/usr/local/cuda-12.1"

# 5. 验证CUDA工具链安装
RUN nvcc --version && \
    which nvcc

# 6. 配置pip使用清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 7. 安装PyTorch 2.4.0 + CUDA 12.1
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 8. 安装Vis4D
RUN pip install vis4d==1.0.0

# 9. 修复PyTorch C++库路径问题
RUN python3 -c "import torch; import os; torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib'); print(f'PyTorch库路径: {torch_lib}')" && \
    python3 -c "import torch; import os; torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib'); open('/etc/ld.so.conf.d/pytorch.conf', 'w').write(torch_lib + '\n')" && \
    ldconfig && \
    echo "已修复PyTorch C++库路径"

# 10. 【关键修复步骤】安装并编译vis4d_cuda_ops
# 在同一个RUN指令内设置所有编译环境变量，确保pip能继承
RUN FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    MAX_JOBS=4 \
    CMAKE_CUDA_ARCHITECTURES="86" \
    pip install git+https://github.com/SysCV/vis4d_cuda_ops.git --no-build-isolation --no-cache-dir

# 11. 安装3D-MOOD项目
WORKDIR /workspace
COPY . /workspace/3D-MOOD
RUN cd 3D-MOOD && pip install -v -e .

# 12. 验证安装（增强验证，确保CUDA功能真的可用）
RUN python -c "import torch; print('PyTorch版本:', torch.__version__)" && \
    python -c "import torch; print('CUDA版本（构建时）:', torch.version.cuda)" && \
    python -c "import vis4d; print('Vis4D版本:', vis4d.__version__)" && \
    python -c "import vis4d_cuda_ops; print('✅ 构建成功: vis4d_cuda_ops 模块已编译并安装')" && \
    python -c "import torch; print('提示: CUDA运行时测试将在容器启动(--gpus all)后进行。')"

# 13. 设置工作目录和默认命令
WORKDIR /workspace/3D-MOOD
CMD ["/bin/bash"]