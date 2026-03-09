@echo off
chcp 65001 > nul

title Qwen3-TTS 语音克隆 0.6B 模型-by pyvideotrans.^com

set HF_ENDPOINT=https://hf-mirror.com

echo.
echo 【当前启动的是：语音克隆 0.6B 模型 Qwen3-TTS-12Hz-0.6B-Base，可基于参考音频进行语音克隆】
echo.
echo 	参考音频时长建议 3-10s
echo.
echo 启动成功后，请在浏览器中打开: http://127.0.0.1:8000
echo 第一次启动后需要下载模型，请耐心等待...
echo.
echo 	*******************************
echo 	如果你在 pyVideoTrans 中使用，请将该地址填写在菜单-TTS设置-Qwen3 TTS(本地)的WebUI URL中
echo.	
echo 	在该设置中测试时，请删掉填写的参考音频，自定义音色模型不可使用参考音频测试，否则会出错
echo 	*******************************
echo.
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


uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-0.6B-Base --ip 0.0.0.0 --port 8000  --no-flash-attn --device cpu --dtype float32

pause