import nest_asyncio
from pyngrok import ngrok
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import threading
import time
import sys

# Cho phép nested event loops
nest_asyncio.apply()

# Khởi tạo FastAPI app
app = FastAPI(title="Math Solver API")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (chạy một lần)
model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
device = "cuda"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded successfully!")

# Định nghĩa request model
class GenerationRequest(BaseModel):
    prompt: str
    reasoning_method: str = "CoT"  # "CoT" hoặc "TIR"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system_message: str = None

@app.get("/")
async def root():
    return {"message": "Math Solver API is running!", "model": model_name}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": model_name}

@app.post("/generate")
async def generate_response(request: GenerationRequest):
    try:
        # Chuẩn bị system message dựa trên phương pháp
        if request.system_message:
            system_content = request.system_message
        elif request.reasoning_method == "CoT":
            system_content = "Please reason step by step, and put your final answer within \\boxed{}."
        else:  # TIR
            system_content = "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."

        # Tạo messages
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": request.prompt}
        ]

        # Tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {
            "response": response,
            "status": "success",
            "model": model_name,
            "parameters": {
                "reasoning_method": request.reasoning_method,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# cell 3: Hàm khởi chạy server trong thread riêng
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Khởi chạy server trong một thread riêng
import threading

# Tạo và khởi chạy server thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Cho server thời gian để khởi động
time.sleep(3)

# cell 4: Thiết lập ngrok tunnel
NGROK_AUTH_TOKEN = ""  # Thay bằng token của bạn

# Thiết lập ngrok auth token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Mở tunnel
public_url = ngrok.connect(8000)
print(f"🎯 Public URL: {public_url}")
print(f"🔗 Use this in Streamlit: {public_url.public_url}")
print("\n📊 Server Information:")
print(f"   Model: {model_name}")
print(f"   Local URL: http://localhost:8000")
print(f"   Public URL: {public_url.public_url}")
print(f"   Health check: {public_url.public_url}/health")
print(f"   API endpoint: {public_url.public_url}/generate")
print("\n⚠️  Keep this Colab tab open! Closing it will stop the server.")
print("📱 Now run Streamlit app on your local machine.")

# Giữ cho cell chạy
try:
    while True:
        time.sleep(10)
        # In trạng thái để biết server vẫn đang chạy
        print(f"[{time.strftime('%H:%M:%S')}] Server is running...")
except KeyboardInterrupt:
    print("\n👋 Stopping server...")
    sys.exit(0)
