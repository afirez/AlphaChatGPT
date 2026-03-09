#!/bin/bash

# 设置终端编码为UTF-8（适配Linux中文显示）
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 设置终端窗口标题（Linux终端兼容写法）
echo -e "\033]0;Qwen3-TTS 语音克隆 1.7B 模型-by pyvideotrans.com\007"

# 设置环境变量（对应原bat的set命令）
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=./tmp/my_hf_cache

# 输出提示信息（适配Linux echo语法，替换原bat的echo.为echo ""）
echo ""
echo "【当前启动的是：语音克隆 1.7B 模型 Qwen3-TTS-12Hz-1.7B-Base，可基于参考音频进行语音克隆】"
echo ""
echo "	参考音频时长建议 3-10s"
echo ""
echo "启动成功后，请在浏览器中打开: http://127.0.0.1:8000"
echo "第一次启动后需要下载模型，请耐心等待..."
echo ""
echo "	*******************************"
echo "	如果你在 pyVideoTrans 中使用，请将该地址填写在菜单-TTS设置-Qwen3 TTS(本地)的WebUI URL中"
echo ""	
echo "	在该设置中测试时，请删掉填写的参考音频，自定义音色模型不可使用参考音频测试，否则会出错"
echo "	*******************************"
echo ""
echo "	如果配置环境和下载模型中出错，请尝试科学上网，然后编辑本sh文件-删掉下面这行内容"
echo ""
echo "		export HF_ENDPOINT=https://hf-mirror.com"
echo ""
echo "	如果你有英伟达显卡并配置了CUDA环境，想加快语音合成速度，也请在本sh文件，删掉 "
echo ""
echo "		--device cpu --dtype float32"
echo ""
echo "	然后保存关闭重新运行"
echo "	*******************************"
echo ""
echo "	运行中可能出现一些\"Warning:\"或\"SoX could not\"信息，忽略即可, 当显示如下信息时即为启动成功："
echo ""
echo "	* To create a public link, set \`share=True\` in \`launch()\`."

# 执行核心命令（Linux下uv命令路径需确保可访问，若uv在虚拟环境需先激活）
# uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000  --no-flash-attn --device cpu --dtype float32
uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000  --no-flash-attn

# 替换原bat的pause（Linux下等待用户输入）
read -p "按任意键继续..."