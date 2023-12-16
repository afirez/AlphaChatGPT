# RealtimeTTS
https://github.com/KoljaB/RealtimeTTS

支持的引擎：
- OpenAI TTS, 
- Elevenlabs, 
- Azure Speech Services, 
- Coqui TTS
- System TTS

## Install

window 需安装 vs_BuildTools.exe vs build 工具

```
pip install RealtimeTTS
```

### CoquiEngine

Delivers high quality, local, neural TTS with voice-cloning.

Downloads a neural TTS model first. In most cases it be fast enought for Realtime using GPU synthesis. Needs around 4-5 GB VRAM.

to clone a voice submit the filename of a wave file containing the source voice as cloning_reference_wav to the CoquiEngine constructor
in my experience voice cloning works best with a 22050 Hz mono 16bit WAV file containing a short (~10-30 sec) sample
On most systems GPU support will be needed to run fast enough for realtime, otherwise you will experience stuttering.