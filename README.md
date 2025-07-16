# RAGagent

A simple Retrieval-Augmented Generation (RAG) agent built with LangChain, FAISS, HuggingFace embeddings, and Groq’s LLaMA3-70B model.

This agent indexes a document into a vector store and allows you to ask questions about it. Relevant chunks are retrieved using semantic similarity and passed to the LLM to generate answers.

## 📄 Features

✅ Loads and splits a Word document (`.docx`)  
✅ Embeds and saves chunks to FAISS vectorstore  
✅ Uses HuggingFace `all-MiniLM-L6-v2` for embeddings  
✅ Runs Groq LLaMA3-70B for answering questions  
✅ Interactive CLI with top relevant chunks shown

## 🚀 Run

1️⃣ Install dependencies:
```bash
pip install -r requirements.txt
