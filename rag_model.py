from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from typing import List

# Initialize FastAPI 
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup model paths and load AI models
model_path = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False, n_ctx=8192)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Dictionary to store FAISS indices and document chunks
department_indices = {}  # {"hr": FAISS index, "finance": FAISS index, ...}
department_chunks = {}  # {"hr": [chunks], "finance": [chunks], ...}

# Create folder for document storage
docs_folder = "uploaded_documents"
os.makedirs(docs_folder, exist_ok=True)

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return text

# Function to chunk text
def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Store chunks in FAISS
def store_chunks(chunks, department):
    global department_indices, department_chunks

    if department not in department_indices:
        department_indices[department] = faiss.IndexFlatL2(384)
        department_chunks[department] = []

    embeddings = embedder.encode(chunks)
    department_indices[department].add(np.array(embeddings, dtype=np.float32))
    department_chunks[department].extend(chunks)

# Retrieve relevant text for queries
def retrieve_relevant_chunks(query, department, top_k=3):
    if department not in department_indices:
        return []
    
    query_embedding = embedder.encode([query])
    _, indices = department_indices[department].search(np.array(query_embedding, dtype=np.float32), top_k)
    return [department_chunks[department][i] for i in indices[0]]

# Query AI model
def query_rag(query, department):
    relevant_chunks = retrieve_relevant_chunks(query, department, top_k=3)
    context = "\n".join(relevant_chunks)
    
    prompt = (
        f"Based on the following document text, answer the question in detail:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer: "
    )
    
    response = llm(prompt, max_tokens=400)
    return response["choices"][0]["text"].strip()

# API to upload PDFs
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    department = filename.split("_")[0].lower()
    file_path = os.path.join(docs_folder, filename)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    store_chunks(chunks, department)
    
    return {"message": f"File '{filename}' uploaded and indexed under '{department}'"}

# API to query documents
@app.get("/query/")
async def query_document(department: str, query: str):
    response = query_rag(query, department)
    return {"response": response}
