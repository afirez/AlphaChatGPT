## voice-pro

```bash
git clone https://github.com/OpenBMB/VoxCPM.git
```

VoxCPM 是一个无离散音频分词器（Tokenizer-Free）的语音合成系统，通过端到端的扩散自回归架构直接生成连续语音表征，绕过对音频的离散编码步骤，实现高度自然且富有表现力的语音合成。

VoxCPM2 是最新的版本 — 基于 MiniCPM-4 基座构建，总计 20亿 参数，在超过 200万小时 的多语种音频数据上训练，支持 30种全球语言+9种中文方言、音色设计、可控声音克隆，原生输出 48kHz 高质量音频。

✨ 核心特性
🌍 30种语言语音合成 — 直接输入原始文本即可合成（支持语言详见下文），无需额外语言标签
🎨 音色设计 — 用自然语言描述（性别、年龄、音色、情绪、语速……）凭空创建全新音色，无需参考音频
🎛️ 可控声音克隆 — 从参考音频片段克隆任意声音，可叠加风格指令控制情绪、语速和表现力，同时保持原始音色
🎙️ 极致克隆 — 提供参考音频及其文本内容，模型接着参考音频进行无缝续写，从而精准还原声音细节特征（与 VoxCPM1.5 一致）
🔊 48kHz 高质量音频 — 输入 16kHz 参考音频，通过 AudioVAE V2 的非对称编解码设计直接输出 48kHz 高质量音频，内置超分能力
🧠 语境感知合成 — 根据文本内容自动推断合适的韵律和表现力
⚡ 实时流式合成 — 在 NVIDIA RTX 4090 上 RTF 低至 ~0.3，通过 Nano-vLLM 或 vLLM-Omni（官方 vLLM 全模态服务，原生支持 VoxCPM2，提供 PagedAttention 与 OpenAI 兼容 API）加速后可达 ~0.13
📜 完全开源，商用就绪 — 权重和代码基于 Apache-2.0 协议发布，免费商用
🌍 支持的语言（30种）

阿拉伯语、缅甸语、中文、丹麦语、荷兰语、英语、芬兰语、法语、德语、希腊语、希伯来语、印地语、印尼语、意大利语、日语、高棉语、韩语、老挝语、马来语、挪威语、波兰语、葡萄牙语、俄语、西班牙语、斯瓦希里语、瑞典语、菲律宾语、泰语、土耳其语、越南语
中国方言：四川话、粤语、吴语、东北话、河南话、陕西话、山东话、天津话、闽南话

最新动态
[2026.04] 🔥 发布 VoxCPM2 — 20亿参数，30种语言，音色设计与可控声音克隆，48kHz 音频输出！模型权重 | 使用文档 | 在线体验 | 官网体验 (适用国内访问)
[2025.12] 🎉 开源 VoxCPM1.5 模型权重，支持 SFT 和 LoRA 微调。(🏆 GitHub Trending #1)
[2025.09] 🔥 发布 VoxCPM 技术报告。
[2025.09] 🎉 开源 VoxCPM-0.5B 模型权重 (🏆 HuggingFace Trending #1)


## Install 

1. 准备包
```bash
git clone https://github.com/OpenBMB/VoxCPM.git

cd VoxCPM

python -m venv venv

call venv\scripts\activate

pip install -e .
```

如果需要显卡推理，需要卸载默认安装的torch和torchaudio

```bash
pip list

pip uninstall torch,torchaudio -y

pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url [https://download.pytorch.org/whl/cu12](https://download.pytorch.org/whl/cu121)8
```

运行实时语音程序

```bash
python app.py 
```

# 浏览器访问地址: `http://localhost:8808`


``` bash
# build 容器
docker-compose -f ./docker-compose-build.yaml build

# 下载模型
python download.py

# docker-compose -f ./docker-compose.yaml up


# 检查 GPU 
docker exec -it <container> nvidia-smi 
docker exec -it <container> python -c "import torch; print(torch.cuda.is_available())"
```

2.  安装和运行

🚀 configure.bat
安装git、ffmpeg、CUDA（使用NVIDIA GPU时）
首次运行一次；需要网络，可能需要1小时以上
不要关闭命令窗口

🚀 start.bat
运行Voice-Pro网页界面
首次运行时安装依赖（可能需要1小时以上）
如果出现问题，删除installer_files后重新运行

```python
python start-abus.py voice
```


3.  更新
🚀 update.bat：更新Python环境（比重新安装更快）

4.  卸载
运行uninstall.bat或删除文件夹（便携式安装）