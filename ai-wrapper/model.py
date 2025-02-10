from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, Response
import threading
import torch

app = FastAPI()

histories = {}
lock = threading.Lock()

class QwenModel:
    def __init__(self):
        self.ckpt_path = "Qwen/Qwen2.5-7B-Instruct"
        self.cpu_only = False
        self.model = None
        self.tokenizer = None

    def load_model_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt_path,
            resume_download=True,
        )

        if self.cpu_only:
            device_map = "cpu"
        else:
            device_map = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_path,
            torch_dtype="auto",
            device_map=device_map,
            resume_download=True,
        ).eval()
        self.model.generation_config.max_new_tokens = 2048

    def chat_stream(self, query, history):
        conversation = []
        for query_h, response_h in history:
            conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
        conversation.append({"role": "user", "content": query})
        
        input_text = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def prompt(self, query, history, id):
        partial_text = ""

        with lock:
            for new_text in self.chat_stream(query, history):
                partial_text += new_text
                yield new_text
            
            history.append((query, partial_text))
            histories[id] = history

model = QwenModel()
model.load_model_tokenizer()

@app.post("/chat")
async def chat_stream(request: dict):
    query = request.get("query")
    history = request.get("history", [])
    id = request.get("id")

    return StreamingResponse(model.prompt(query, history, id), media_type="text/plain")

@app.get("/chat/history")
async def chat_history(id: str):
    if id in histories:
        return JSONResponse(histories[id])
    return JSONResponse({"message": "History not found"}, status_code=404)

@app.get("/chat/delete")
async def chat_delete(id: str):
    if id in histories:
        histories.pop(id)
    
    return Response()