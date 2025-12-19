import requests
import json

def chat_with_llama():
    print("=== 与 Llama 3.1 对话 ===")
    print("输入 '/quit' 退出对话")
    print("-" * 30)
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ['/quit', '/exit', '退出']:
            print("对话结束！")
            break
            
        response = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json={
                'model': 'llama3.1',
                'prompt': user_input,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            answer = response.json()['response']
            print(f"Llama: {answer}")
        else:
            print("错误：无法连接到 Ollama 服务")

if __name__ == "__main__":
    chat_with_llama()