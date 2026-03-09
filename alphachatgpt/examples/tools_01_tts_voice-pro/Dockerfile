# FROM nvcr.io/nvidia/pytorch:23.10-py3
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
# docker run --rm --gpus all pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel nvidia-smi

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --upgrade pip  # enable PEP 660 support

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
	sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
    
WORKDIR /app

RUN pip3 install transformers==4.35.0 huggingface-hub==0.17.3 hf_transfer==0.1.4

RUN apt-get update && apt-get install cmake -y && apt-get autoremove -y

# HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --resume-download --local-dir-use-symlinks False ${模型在huggingface上的名字} ${模型文件名}.gguf --local-dir ${模型存储路径}
ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_ENABLE_HF_TRANSFER=1

# 拷贝项目
COPY voice-pro /app/voice-pro
ADD requirements-voice-gpu-2.txt /app/voice-pro/requirements-voice-gpu-2.txt

# 安装 voice 的依赖（GPU 版为例）
RUN pip3 install -r /app/voice-pro/requirements-voice-gpu-2.txt

# 启动 voice
WORKDIR /app/voice-pro
CMD ["python", "start-voice.py"]