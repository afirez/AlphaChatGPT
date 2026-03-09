@echo off
chcp 65001 > nul

title Qwen3-TTS 声音设计 1.7B 模型-by pyvideotrans.^com
set HF_ENDPOINT=https://hf-mirror.com

echo.
echo 【当前启动的是:声音设计 1.7B 模型 Qwen3-TTS-12Hz-1.7B-VoiceDesign 可自行使用文字描述创建新音色】
echo.
echo 启动成功后，请在浏览器中打开: http://127.0.0.1:8000
echo 第一次启动后需要下载模型，请耐心等待...
echo.
echo 	*******************************
echo 	如果配置环境和下载模型中出错，请尝试科学上网，然后右键本bat文件-编辑-删掉下面这行内容
echo.
echo 		set HF_ENDPOINT=https://hf-mirror.com
echo.
echo 	如果你有英伟达显卡并配置了CUDA环境，想加快语音合成速度，也请在本bat文件，删掉 
echo.
echo 		--device cpu --dtype float32
echo.
echo 	然后保存关闭重新运行
echo 	*******************************
echo.
echo 	运行中可能出现一些"Warning:"或"SoX could not"信息，忽略即可, 当显示如下信息时即为启动成功：
echo.
echo 	^* To create a public link, set `share=True` in `launch()`.

runtime\uv.exe run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --ip 0.0.0.0 --port 8000  --no-flash-attn --device cpu --dtype float32

pause