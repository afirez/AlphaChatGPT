import requests
import json

# API服务地址
API_URL = "http://localhost:8000/chat"

# 测试对话
def test_chat():
    # 构建请求数据
    payload = {
        "messages": [
            {"role": "system", "content": "你是一个 helpful 的人工智能助手。请用简洁明了的语言回答问题。"},
            {"role": "user", "content": "什么是量子计算？用简单的语言解释一下。"}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        # 发送请求
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            print(f"模型响应: {result['response']}")
            print(f"使用tokens: {result['tokens_used']}")
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"请求发生错误: {str(e)}")

# 测试健康检查
def test_health():
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("服务健康状态: ", response.json())
        else:
            print(f"健康检查失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"健康检查发生错误: {str(e)}")

if __name__ == "__main__":
    test_health()
    print("\n--- 测试对话 ---")
    test_chat()
