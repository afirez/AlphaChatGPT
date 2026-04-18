
"""
如果你希望先从 ModelScope 下载模型到本地（适用于国内网络访问），可以使用：
pip install modelscope
"""

from modelscope import snapshot_download
snapshot_download("OpenBMB/VoxCPM2", local_dir='./tmp/pretrained_models/OpenBMB/VoxCPM2') # 指定模型保存的本地路径
snapshot_download("iic/SenseVoiceSmall", local_dir='./tmp/pretrained_models/iic/SenseVoiceSmall') # 指定模型保存的本地路径


# from voxcpm import VoxCPM
# import soundfile as sf
# model = VoxCPM.from_pretrained('./pretrained_models/VoxCPM2', load_denoiser=False)

# wav = model.generate(
#     text="VoxCPM2 是目前推荐使用的多语言语音合成版本。",
#     cfg_value=2.0,
#     inference_timesteps=10,
# )
# sf.write("demo.wav", wav, model.tts_model.sample_rate)