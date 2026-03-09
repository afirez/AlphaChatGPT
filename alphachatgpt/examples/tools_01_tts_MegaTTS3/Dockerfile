# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --upgrade pip  # enable PEP 660 support

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
	sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    ffmpeg \
    && apt-get clean

# COPY ./MegaTTS3/requirements.txt /app/
COPY ./requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY ./MegaTTS3/ /app/

# ENV HF_ENDPOINT=https://hf-mirror.com
# # ENV HUGGINGFACE_HUB_CACHE=/app/tmp/my_hf_cache
# RUN huggingface-cli download ByteDance/MegaTTS3 --local-dir ./checkpoints --local-dir-use-symlinks False

CMD ["python", "-m", "tts.gradio_api"]
