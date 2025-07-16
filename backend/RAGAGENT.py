import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import bs4


load_dotenv()  
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("❌ Please set GROQ_API_KEY in your .env file")

# 🔷 Load and parse web page
print("Loading document…")
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)
docs = loader.load()
print(f" Loaded {len(docs)} document(s)")

# 🔷 Split into chunks
print("Splitting into chunks…")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# 🔷 Embed chunks (locally, free)
print("Embedding chunks with HuggingFace…")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("faiss_index")
print("FAISS index saved in ./faiss_index")

# 🔷 Setup retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 🔷 Setup Groq LLM
print("Connecting to Groq LLM…")
llm = ChatOpenAI(
    model_name="llama3-70b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,
    temperature=0
)


rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

print("RAG agent ready! Type your question below.\n")


while True:
    query = input("Your question (or type 'exit'): ").strip()
    if query.lower() in {"exit", "quit"}:
        print(" Goodbye!")
        break

    answer = rag_chain.run(query)
    print(f"Answer: {answer}\n")
