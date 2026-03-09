from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import logging
from typing import List, Optional

"""
fastapi==0.104.1
uvicorn==0.24.0.post1
llama-cpp-python==0.2.78
pydantic==2.5.2
requests==2.31.0
"""

"""
# 卸载可能存在的旧版本
pip uninstall -y llama-cpp-python

# 启用CUDA支持安装（关键步骤）
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

CMAKE_ARGS="-DLLAMA_CUDA=on -DLLAMA_CUBLAS=on" pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

"""

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="Qwen3-30B-A3B-Instruct Q4 API服务")

# 模型加载配置
MODEL_PATH = "./models/tmp/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"  # 替换为你的模型路径
CONTEXT_WINDOW = 4096  # Qwen3的上下文窗口大小
N_THREADS = 8  # 根据CPU核心数调整
N_GPU_LAYERS = 20  # -1表示加载所有可能的层到GPU

# 加载模型
try:
    logger.info(f"开始加载模型: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=CONTEXT_WINDOW,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        model_type="qwen3moe",  # 明确指定模型类型
        verbose=False
    )
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise

# 请求和响应数据模型
class ChatRequest(BaseModel):
    messages: List[dict]
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    model: str = "Qwen3-30B-A3B-Instruct-Q4"
    tokens_used: int

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# 聊天接口
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 构建提示词（遵循Qwen的对话格式）
        prompt = ""
        for msg in request.messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<{role}>{content}</{role}>"
        prompt += "<assistant>"  # 指示模型开始生成回答
        
        # 生成响应
        logger.info(f"处理请求: {request.messages[-1]['content'][:50]}...")
        output = llm(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["</assistant>"],  # 停止标记
            echo=False
        )
        
        # 解析结果
        response_text = output["choices"][0]["text"].strip()
        tokens_used = output["usage"]["total_tokens"]
        
        logger.info(f"生成响应完成，使用tokens: {tokens_used}")
        return {
            "response": response_text,
            "tokens_used": tokens_used
        }
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

# 主函数
if __name__ == "__main__":
    # 启动服务，默认端口8000
    uvicorn.run(
        app="py_service_qwen3_30B_A3B:app",
        host="0.0.0.0",  # 允许外部访问
        port=8000,
        workers=1  # 单worker避免模型重复加载
    )
