from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, Response
import threading
import faiss
import numpy as np
import torch
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import requests

load_dotenv()

app = FastAPI()

lock = threading.Lock()

class QwenModel:
    def __init__(self):
        self.ckpt_path = "Qwen/Qwen2.5-7B-Instruct"
        self.cpu_only = False
        self.model = None
        self.tokenizer = None
        self.doc_embeddings = None
        self.histories = {}
        self.docs = {}
        self.links = {}
        self.doc_indices = {}
        self.google_api_key = os.getenv("YOUR_GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("YOUR_CUSTOM_SEARCH_ENGINE_ID")
        
    def get_full_text(self, url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                return soup.get_text()
            else:
                return None
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return None

    def get_documents_from_google(self, query, num_results=5):
        service = build("customsearch", "v1", developerKey=self.google_api_key)
        res = service.cse().list(q=query, cx=self.google_cse_id, num=num_results).execute()

        documents = []
        links = []

        if "items" in res:
            for item in res["items"]:
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                links.append(link)

                full_text = self.get_full_text(link)
                if full_text:
                    documents.append({
                        "snippet": snippet,
                        "full_text": full_text
                    })

        return documents, links

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

    def create_faiss_index(self, query, id):
        self.docs[id], self.links[id] = self.get_documents_from_google(query)
        doc_embeddings = []
        for doc in self.docs[id]:
            doc_embeddings.append(self.get_embedding(doc))

        doc_embeddings_np = np.array(doc_embeddings)
        dimension = doc_embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings_np)

        return index

    def get_embedding(self, doc):
        """Get the embedding for a text using the model"""
        inputs = self.tokenizer(f"Snippet: {doc.get("snippet")}\nFull-Text: {doc.get("full_text")}", 
            return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.embedder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            embeddings = embeddings.to(torch.float32)
        return embeddings.cpu().numpy().squeeze()

    def retrieve_documents(self, query, id):
        query_embedding = self.get_embedding(query)
        query_embedding_np = query_embedding.reshape(1, -1)
    
        k = 3
        index = self.create_faiss_index(query, id)
        _, indices = index.search(query_embedding_np, k)
        self.doc_indices[id] = indices.squeeze()
        
        retrieved_docs = [self.docs[id][i] for i in self.doc_indices[id]]

        return retrieved_docs

    def chat_stream(self, query, history, id, use_rag=True):
        conversation = []
        for query_h, response_h in history:
            conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
        
        if use_rag:
            retrieved_docs = self.retrieve_documents(query, id)
            conversation.extend([{"role": "system", "content": f"Snippet: {doc.get("snippet")}\nFull-Text: {doc.get("full_text")}"} for doc in retrieved_docs])

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
    if id in model.doc_indices:
        docs = [{
            "docs": model.docs[id][i],
            "links": model.links[id][i]
        } for i in model.doc_indices[id]]
        return JSONResponse(docs)
    return JSONResponse({"message": "History not found"}, status_code=404)

@app.get("/chat/del-rag-doc")
async def chat_delete_rag_doc(id: str):
    if id in model.doc_indices:
        model.doc_indices.pop(id)
        model.docs.pop(id)
    
    return Response()
