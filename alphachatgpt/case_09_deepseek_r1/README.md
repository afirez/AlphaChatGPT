## Deepseek

```bash
git submodule add https://github.com/afirez/TinyZero.git
```

- 下载 huggingface 数据集

https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4

```bash
git lfs clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
# or
git lfs clone https://hf-mirror.com/datasets/Jiayi-Pan/Countdown-Tasks-3to4
```

```bash 
pip install -U huggingface_hub

# 设置环境变量以使用镜像站：
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME = "usr/local/"
# 对于 Windows Powershell，使用：
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\\cache"

huggingface-cli download --resume-download Jiayi-Pan/Countdown-Tasks-3to4

```