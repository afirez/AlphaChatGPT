import os
import time
from typing import List, Optional, Dict, Any, Generator
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from llama_cpp import Llama #, LlamaGenerationResponse
from llama_cpp.llama_types import CreateChatCompletionResponse
# from dotenv import load_dotenv

# 加载环境变量配置
# load_dotenv()
# MODEL_PATH = os.getenv("MODEL_PATH", "./models/tmp/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")
# N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", 20))  # 根据GPU显存调整
# N_THREADS = int(os.getenv("N_THREADS", 8))
# N_CTX = int(os.getenv("N_CTX", 4096))  # 上下文长度

MODEL_PATH = "./models/tmp/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"  # 替换为你的模型路径
CONTEXT_WINDOW = 4096  # Qwen3的上下文窗口大小
N_CTX = CONTEXT_WINDOW
N_THREADS = 8  # 根据CPU核心数调整
N_GPU_LAYERS = 20  # -1表示加载所有可能的层到GPU

# 初始化FastAPI应用（自动生成OpenAPI规范）
app = FastAPI(
    title="Qwen3-30B OpenAPI Service",
    description="兼容OpenAPI规范的Qwen3-30B-A3B-Instruct模型服务",
    version="1.0.0",
    docs_url="/docs",  # OpenAPI交互式文档
    redoc_url="/redoc"  # 另一种风格的文档
)

# 全局模型实例
llm: Optional[Llama] = None

# ------------------------------
# 数据模型（符合OpenAPI规范）
# ------------------------------
class Message(BaseModel):
    role: str = Field(..., description="消息角色，可选值：system, user, assistant")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., description="对话历史消息列表")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数，控制随机性")
    max_tokens: Optional[int] = Field(None, description="最大生成token数")
    stream: bool = Field(False, description="是否启用流式响应")

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = Field(None, description="结束原因：stop, length等")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="响应ID")
    object: str = Field("chat.completion", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="使用的模型")
    choices: List[Choice]
    usage: Dict[str, int] = Field(..., description="token使用统计")

class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, str] = Field(..., description="增量内容")
    finish_reason: Optional[str]

class StreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]

# ------------------------------
# 工具函数
# ------------------------------
def format_prompt(messages: List[Message]) -> str:
    """将消息列表格式化为Qwen模型要求的prompt"""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<<system>>{msg.content}<</system>>\n"
        elif msg.role == "user":
            prompt += f"<<user>>{msg.content}<</user>>\n"
        elif msg.role == "assistant":
            prompt += f"<<assistant>>{msg.content}<</assistant>>\n"
    # 最后添加助手前缀，提示模型生成
    prompt += "<<assistant>>"
    return prompt

# ------------------------------
# 启动时加载模型
# ------------------------------
@app.on_event("startup")
def load_model():
    global llm
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        
        print(f"开始加载模型: {MODEL_PATH}")
        start_time = time.time()
        llm = Llama(
            model_path=MODEL_PATH,
            model_type="qwen",  # 强制指定模型类型
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False
        )
        print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise  # 启动失败

# ------------------------------
# API接口（兼容OpenAPI）
# ------------------------------
@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        500: {"description": "服务器内部错误"},
        400: {"description": "无效请求参数"}
    },
    description="聊天补全接口，兼容OpenAI风格"
)
def chat_completions(request: ChatCompletionRequest):
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型未加载，请稍后重试"
        )
    
    # 验证模型名称（可选，可根据实际支持的模型列表验证）
    if False and request.model not in ["qwen3-30b-a3b-instruct-q4"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的模型: {request.model}，支持的模型: qwen3-30b-a3b-instruct-q4"
        )
    
    # 格式化prompt
    prompt = format_prompt(request.messages)
    
    try:
        # 非流式响应
        if not request.stream:
            
            response: CreateChatCompletionResponse = llm.create_completion(
                prompt=prompt,
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature,
                stop=["<</assistant>>"],  # 停止符
                echo=False
            )
            if True:
                return response

            # 构造响应
            completion_id = f"chatcmpl-{int(time.time() * 1000)}"
            created = int(time.time())
            content = response["choices"][0]["text"].strip()
            
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=content),
                        finish_reason=response["choices"][0]["finish_reason"]
                    )
                ],
                usage=response["usage"]
            )
        
        # 流式响应
        else:
            def stream_generator() -> Generator[str, None, None]:
                completion_id = f"chatcmpl-{int(time.time() * 1000)}"
                created = int(time.time())
                first_chunk = True
                
                for chunk in llm.create_completion(
                    prompt=prompt,
                    max_tokens=request.max_tokens or 1024,
                    temperature=request.temperature,
                    stop=["<</assistant>>"],
                    stream=True,
                    echo=False
                ):
                    # 构造流式响应
                    delta = {"content": chunk["choices"][0]["text"]} if first_chunk else {"content": chunk["choices"][0]["text"]}
                    first_chunk = False
                    
                    stream_resp = StreamResponse(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=delta,
                                finish_reason=chunk["choices"][0]["finish_reason"]
                            )
                        ]
                    )
                    yield f"data: {stream_resp.model_dump_json()}\n\n"
                    
                    # 如果结束，发送终止符
                    if chunk["choices"][0]["finish_reason"] is not None:
                        yield "data: [DONE]\n\n"
        
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成响应失败: {str(e)}"
        )

# 健康检查接口
@app.get("/health", description="服务健康检查")
def health_check():
    return {
        "status": "healthy" if llm is not None else "unhealthy",
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "timestamp": int(time.time())
    }
    

# 主函数
if __name__ == "__main__":
    import uvicorn
    # 启动服务，默认端口8000
    uvicorn.run(
        app="py_service_qwen3_30B_A3B_v2:app",
        host="0.0.0.0",  # 允许外部访问
        port=8000,
        workers=1  # 单worker避免模型重复加载
    )
