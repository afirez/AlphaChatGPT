# Qwen3-TTS

https://hf-mirror.com/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/main

### 构建与运行

在命令行中执行以下命令：

1.  **构建镜像**：
    ```bash
    # docker-compose build
    docker-compose build -f docker-cpmpose-build.yaml
    ```
2.  **启动容器**：
    ```bash
    docker-compose up
    ```

### 注意事项：

*   **GPU 支持**：如果你想在 Docker 中使用 NVIDIA 显卡加速，基础镜像需要更换为 `nvidia/cuda`（例如 `nvidia/cuda:12.1.0-base-ubuntu22.04`），并且需要安装 Python 环境以及配置 `nvidia-container-runtime`。同时，启动命令中的 `--device cpu` 需改为 `--device cuda`。
*   **模型选择**：你可以通过修改 `docker-compose.yml` 中的 `command` 字段来切换不同的模型（如 `0.6B-Base` 或 `VoiceDesign`）。
*   **网络**：容器启动后，你可以通过 `http://localhost:8000` 访问 WebUI。

这种方式可以将复杂的依赖环境完全封装在镜像中，保持你 Windows 宿主机的洁净。