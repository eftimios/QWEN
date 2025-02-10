from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, AutoModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, Response
import threading
import faiss
import numpy as np
import torch

app = FastAPI()

lock = threading.Lock()

class QwenModel:
    def __init__(self):
        self.ckpt_path = "Qwen/Qwen2.5-7B-Instruct"
        self.cpu_only = False
        self.model = None
        self.tokenizer = None
        self.index = None
        self.doc_embeddings = None
        self.histories = {}
        self.latest_docs = {}

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

        self.embedder = self.model.base_model

        self.model.generation_config.max_new_tokens = 2048

        self.create_faiss_index()

    def create_faiss_index(self):
        documents = ["Document 1 text", "Document 2 text"]
        doc_embeddings = []
        for doc in documents:
            doc_embeddings.append(self.get_embedding(doc))

        doc_embeddings_np = np.array(doc_embeddings)
        dimension = doc_embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(doc_embeddings_np)

    def get_embedding(self, text):
        """Get the embedding for a text using the model"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.embedder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            embeddings = embeddings.to(torch.float32)
        return embeddings.cpu().numpy().squeeze()

    def retrieve_documents(self, query, id):
        query_embedding = self.get_embedding(query)

        query_embedding_np = query_embedding.reshape(1, -1)
    
        k = 2
        docs, indices = self.index.search(query_embedding_np, k)

        print(docs)
        print(indices)
        
        retrieved_docs = [f"Document {i+1} text" for i in indices[0]]
        
        self.latest_docs[id] = retrieved_docs

        return retrieved_docs

    def chat_stream(self, query, history, id, use_rag=True):
        conversation = []
        for query_h, response_h in history:
            conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
        
        if use_rag:
            retrieved_docs = self.retrieve_documents(query, id)
            conversation.extend([{"role": "system", "content": doc} for doc in retrieved_docs])

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

    def prompt(self, query, history, id, use_rag=True):
        partial_text = ""

        with lock:
            for new_text in self.chat_stream(query, history, id, use_rag):
                partial_text += new_text
                yield new_text

            history.append((query, partial_text))
            self.histories[id] = history

model = QwenModel()
model.load_model_tokenizer()

@app.post("/chat")
async def chat_stream(request: dict):
    query = request.get("query")
    history = request.get("history", [])
    id = request.get("id")
    use_rag = request.get("rag", True)

    return StreamingResponse(model.prompt(query, history, id, use_rag), media_type="text/plain")

@app.get("/chat/history")
async def chat_history(id: str):
    if id in model.histories:
        return JSONResponse(model.histories[id])
    return JSONResponse({"message": "History not found"}, status_code=404)

@app.get("/chat/del-history")
async def chat_delete_history(id: str):
    if id in model.histories:
        model.histories.pop(id)
    
    return Response()

@app.get("/chat/rag-doc")
async def chat_rag_doc(id: str):
    if id in model.latest_docs:
        return JSONResponse(model.latest_docs[id])
    return JSONResponse({"message": "History not found"}, status_code=404)

@app.get("/chat/del-rag-doc")
async def chat_delete_rag_doc(id: str):
    if id in model.latest_docs:
        model.latest_docs.pop(id)
    
    return Response()
