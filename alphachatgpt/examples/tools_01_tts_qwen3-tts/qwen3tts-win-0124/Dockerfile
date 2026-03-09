# 使用 Python 3.10 基础镜像
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
# docker run --rm --gpus all pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel nvidia-smi

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --upgrade pip  # enable PEP 660 support

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
	sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 设置环境变量，防止 Python 产生 pyc 文件及启用缓存
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HUGGINGFACE_HUB_CACHE=/app/tmp/my_hf_cache

# 安装系统依赖（音频处理所需的库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# ENV PATH="/root/.cargo/bin:${PATH}"

# 将 uv 安装到隔离环境中
# RUN pipx install uv

RUN pip install uv

# 设置工作目录
WORKDIR /app

# 复制项目配置文件
COPY pyproject.toml uv.lock ./

# 使用 uv 安装依赖
RUN uv sync

# 复制其余项目文件（可选，如果需要本地的 runtime 脚本）
# COPY . .

# 暴露服务端口
EXPOSE 8000

# 默认启动命令 (以 1.7B 模型为例)
# CMD ["uv", "run", "qwen-tts-demo", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "--ip", "0.0.0.0", "--port", "8000", "--no-flash-attn", "--device", "cpu", "--dtype", "float32"]
CMD ["uv", "run", "qwen-tts-demo", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "--ip", "0.0.0.0", "--port", "8000", "--no-flash-attn"]