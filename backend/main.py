from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from typing import Union

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer

import os
from dotenv import load_dotenv
import pathlib
import uvicorn

# ===============================
# 🔷 Load environment variables
# ===============================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("❌ Please set GROQ_API_KEY in your .env file")

# ===============================
# 🔷 LangChain setup
# ===============================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatOpenAI(
    model_name="llama3-70b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", legacy=False)

class QueryRequest(BaseModel):
    query: str
@app.get("/")
def read():
    return "it runs"

# @app.post("/api/ask")
# async def  ask_question(req: QueryRequest):
#     """Answer user question using QA chain"""
#     result = qa_chain.invoke({"query": req.query})["result"]
#     return {"answer": result}


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)


@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    """Answer user question using QA chain, print token usage, return only result"""

    # 🔷 Count tokens in query
    query_ids = tokenizer.encode(req.query)
    query_token_count = len(query_ids)

    # 🔷 Retrieve context
    docs = retriever.get_relevant_documents(req.query)
    context_text = "\n\n".join(doc.page_content for doc in docs)
    context_token_count = len(tokenizer.encode(context_text))

    # 🔷 Run QA chain
    result = qa_chain.invoke({"query": req.query})["result"]

    # 🔷 Count tokens in response
    response_token_count = len(tokenizer.encode(result))

    total_tokens = query_token_count + context_token_count + response_token_count

    # 🔷 Print to terminal
    print(f"\n📊 Token Usage for Query: \"{req.query}\"")
    print(f"    🔷 Query tokens:    {query_token_count}")
    print(f"    🔷 Query Tokens: {tokenizer.convert_ids_to_tokens(query_ids)}")
    print(f"    🔷 Context tokens:  {context_token_count}")
    print(f"    🔷 Response tokens: {response_token_count}")
    print(f"    🔷 TOTAL tokens:    {total_tokens}\n")

    return {"answer": result}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)