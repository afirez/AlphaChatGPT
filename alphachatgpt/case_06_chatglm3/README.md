# ChatGLM3

## chatglm3-6b-32k & docker

```bash
# 创建命令：
# 回车后会自动下载镜像并在自己的电脑上运行起来
docker run -d --name chatglm3 --gpus all --network host bucess/chatglm3:1 
# or
docker run -d --name chatglm3 --gpus all --network bridge -p 8501:8501 bucess/chatglm3:1 

# 停止命令： 
docker stop chatglm3

# 再次启动命令：
docker start chatglm3
```
