# TTS

## emoti-voice & docker
网易易魔声 TTS

```bash
# 创建命令：
docker run -d --name emoti-voice --gpus all --network host syq163/emoti-voice:latest
# 停止命令：
docker stop emoti-voice
# 再次启动命令：
docker start emoti-voice
```