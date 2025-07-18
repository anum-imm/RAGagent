from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from uuid import uuid4
import os

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer

from db import SessionLocal, ChatSession, Conversation  

import uvicorn
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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    memory=memory
)

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
            chat_session = ChatSession(id=session_id, title=req.query)
            db.add(chat_session)
            db.commit()

        result = app.invoke(
            MessagesState(),
            config={"configurable": {"thread_id": session_id}}
        )

        bot_response = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, SystemMessage):
                bot_response = msg.content
                break

        # calculate tokens for this turn
        query_tokens = len(tokenizer.encode(req.query))
        response_tokens = len(tokenizer.encode(bot_response))
        total_tokens = query_tokens + response_tokens

        # save in conversation table
        convo = Conversation(
            session_id=session_id,
            user_message=req.query,
            bot_response=bot_response,
            tokens_used=total_tokens
        )
        db.add(convo)

       
        chat_session.total_tokens += total_tokens
        db.commit()

        return {
            "answer": bot_response or "🤷 Sorry, no answer found.",
            "session_id": session_id
        }

    finally:
        db.close()


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    db = SessionLocal()
    try:
        conversations = (
            db.query(Conversation)
            .filter_by(session_id=session_id)
            .order_by(Conversation.created_at.asc())
            .all()
        )
        return [
            {
                "user_message": c.user_message,
                "bot_response": c.bot_response,
                "created_at": c.created_at.isoformat()
            }
            for c in conversations
        ]


    finally:
        db.close()


@app.get("/api/sessions")
def list_sessions():
    """
    📜 List all past chat sessions (id + timestamp).
    Used to populate session selector in frontend.
    """
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.started_at.desc()).all()

        return [
            {
                "session_id": s.id,
                "started_at": s.started_at.isoformat(),
                "total_tokens": s.total_tokens,
                "title": s.title or "Untitled"
            }
            for s in sessions
        ]
    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
