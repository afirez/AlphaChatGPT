# 有了 Docker，可以使用任意老版本的 PyTorch 和 CUDA，而不会影响其他的项目
# FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
# docker run --rm --gpus all pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel nvidia-smi

# Dockerfile 中 APT 包的安装比较复杂，建议复制下面的格式
# rm /etc/apt/sources.list.d/cuda.list 如果镜像中没有 CUDA，则删掉这一行
# 如果不指定 DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai 和 -y 参数，构建镜像会因为 apt 等待用户输入而卡死
# RUN rm /etc/apt/sources.list.d/cuda.list \
#     && sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
#     && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
#     && apt-get update \
#     && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git gdb vim curl wget tmux zip cmake ffmpeg libsm6 libxext6 \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y git curl wget zip\
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda3
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && sh Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3 \
#     && ~/miniconda3/bin/conda init \
#     && source ~/.bashrc 

# environment.yml 需要放在 Dockerfile 同一目录下
# COPY environment.yml /tmp/environment.yml
# RUN conda env create -f /tmp/environment.yml
# RUN conda init bash

# RUN conda create -p ./envs/py311_unsloth python=3.11 pytorch-cuda=12.1 pytorch=2.4.0 cudatoolkit -c pytorch -c nvidia -y
# RUN conda init bash
# RUN conda activate ./py311_unsloth

# 除了 Conda，也可以直接用 pip 安装 Python 包，Docker 已经提供了环境隔离
# RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com h5py einops tqdm matplotlib tensorboard torch-tb-profiler ninja scipy

# 安装 PyTorch 2.4 与 CUDA 12.1
# RUN conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# RUN conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# RUN pip uninstall torch torchvision torchaudio
# Run pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# Run pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 安装 trl
RUN pip install trl[diffusers]
# RUN pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
# RUN pip install git+https://github.com/huggingface/trl.git

# 安装 vllm
RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
# RUN pip install vllm==0.6.4.post1 --extra-index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install vllm==0.6.4 --extra-index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install vLLM with CUDA 12.1
# RUN export VLLM_VERSION=0.7.2 \
#     && export PYTHON_VERSION=311 \
#     && pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu121-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121
# RUN pip show vllm
# RUN pip install --upgrade pillow

# 安装 Unsloth
# RUN pip install unsloth 
# RUN pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install "unsloth[cu121-ampere-torch251] @ git+https://github.com/unslothai/unsloth.git"

# 安装 Jupyter
RUN pip install ipykernel jupyter

# 设置工作目录
WORKDIR /workspace

# 暴露 Jupyter 端口
EXPOSE 8888

# 启动 Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]