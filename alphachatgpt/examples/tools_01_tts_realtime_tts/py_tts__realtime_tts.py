from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

"""
ComfyUI、SD-WEBUI、sdxl turbo、SVD、Whisper、RVC、TokenFlow
"""

# 打开文件
with open('./data.txt', 'r', encoding="utf-8") as file:
    # 读取文件内容
    file_content = file.read()

# 处理文件内容
# print(file_content)

engine = SystemEngine(voice="Huihui", print_installed_voices=True) # replace with your TTS engine
# engine = SystemEngine(print_installed_voices=True) # replace with your TTS engine
stream = TextToAudioStream(engine)
# stream.feed("Hello world! How are you today?")
# stream.play()
# stream.feed("你好啊")
stream.feed(file_content)
# stream.play_async()

stream.play(output_wavfile="./output.wav")