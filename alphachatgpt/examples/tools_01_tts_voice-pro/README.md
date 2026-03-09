## voice-pro

```bash
git clone https://github.com/abus-aikorea/voice-pro.git
```

Voice-Pro是一款革新多媒体内容制作的先进网页应用。它将YouTube视频下载、音频分离、语音识别、翻译和文本转语音(TTS)集成到一个强大的工具中，为创作者、研究人员和多语言专家提供理想的解决方案。

🔊 顶级语音识别: Whisper, Faster-Whisper, Whisper-Timestamped, WhisperX
🎤 零样本语音克隆: F5-TTS, E2-TTS, CosyVoice
📢 多语言文本转语音: Edge-TTS, kokoro (付费版包括 Azure TTS)
🎥 YouTube处理与音频提取: yt-dlp
🌍 超过100种语言的即时翻译: Deep-Translator (付费版包括 Azure Translator)
作为ElevenLabs的强大替代方案，Voice-Pro为播客主持人、开发者和创作者提供高级语音解决方案。

## Install 

1. 准备包
```bash
git clone https://github.com/abus-aikorea/voice-pro.git
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