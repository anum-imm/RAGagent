from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from uuid import uuid4
import os

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer

from db import SessionLocal, ChatSession, Conversation  

import uvicorn
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
# ===============================
# 🔷 Load environment
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

#it will replace chain and work node wise
workflow = StateGraph(state_schema=MessagesState)

def call_rag(state: MessagesState):
    # get last user input
    user_message = state["messages"][-1].content

    # retrieve documents
    retrieved_docs = retriever.get_relevant_documents(user_message)
    context = "\n".join(doc.page_content for doc in retrieved_docs)

    # build prompt
    system_prompt = (
        "You are a helpful RAG assistant. Use only the retrieved context to answer.\n\n"
        "If the user query is out of context DO NOT answer it "
        f"Context:\n{context}"
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}

workflow.add_node("rag", call_rag)
workflow.add_edge(START, "rag")

rag_app = workflow.compile(checkpointer=memory)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",
#    # memory=memory
# )

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", legacy=False)




# app = qa_chain.compile(checkpointer=memory)

# ===============================
# 🔷 FastAPI
# ===============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = None


@app.get("/")
def read():
    return "it runs"

@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    session_id = req.session_id or str(uuid4())

    db = SessionLocal()
    try:
        chat_session = db.query(ChatSession).filter_by(id=session_id).first()
        if not chat_session:
            chat_session = ChatSession(id=session_id, title="untitled")
            db.add(chat_session)
            db.commit()

        # call the graph
        graph_result = rag_app.invoke(
            {"messages": [{"role": "user", "content": req.query}]},
            config={"configurable": {"thread_id": session_id}},
        )

        result = graph_result["messages"][-1].content


        # calculate tokens
        query_tokens = len(tokenizer.encode(req.query))
        response_tokens = len(tokenizer.encode(result))
        total_tokens = query_tokens + response_tokens

        # save to DB
        convo = Conversation(
            session_id=session_id,
            user_message=req.query,
            bot_response=result,
            tokens_used=total_tokens
        )
        db.add(convo)

        chat_session.total_tokens += total_tokens
        db.commit()

        print(f"\n📊 Token Usage for Query: \"{req.query}\"")
        print(f"    🔷 Query tokens:    {query_tokens}")
        print(f"    🔷 Response tokens: {response_tokens}")
        print(f"    🔷 TOTAL tokens:    {total_tokens}\n")

    finally:
        db.close()
        print(result)
        return {
            "answer": result,
            "session_id": session_id
        }

# @app.post("/api/ask")
# async def ask_question(req: QueryRequest):
#     session_id = req.session_id or str(uuid4())

#     db = SessionLocal()
#     try:
#         chat_session = db.query(ChatSession).filter_by(id=session_id).first()
#         if not chat_session:
#             chat_session = ChatSession(id=session_id, title="untitled")
#             db.add(chat_session)
#             db.commit()

#         result = qa_chain.invoke({"query": req.query})["result"]
        
#         # calculate tokens for this turn
#         query_tokens = len(tokenizer.encode(req.query))
#         response_tokens = len(tokenizer.encode(result))
#         total_tokens = query_tokens + response_tokens

#         # save in conversation table
#         convo = Conversation(
#             session_id=session_id,
#             user_message=req.query,
#             bot_response=result,
#             tokens_used=total_tokens
#         )
#         db.add(convo)

#         chat_session.total_tokens += total_tokens
#         db.commit()

#         print(f"\n📊 Token Usage for Query: \"{req.query}\"")
#         print(f"    🔷 Query tokens:    {query_tokens}")
#         print(f"    🔷 Response tokens: {response_tokens}")
#         print(f"    🔷 TOTAL tokens:    {total_tokens}\n")
#        # print(f"Session ID: {session_id}")
  
#     finally:
#         db.close()
#         print(result)
#         return {
#             "answer": result,
#             "session_id": session_id
#         }
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
