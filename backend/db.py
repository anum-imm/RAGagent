from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import uuid

# DB connection string: sqlite
engine = create_engine("sqlite:///RAGagent.db")

# session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 🔷 Session table: one per chat thread
class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)  # NEW
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    total_tokens = Column(Integer, default=0)

    conversations = relationship("Conversation", back_populates="session")

# 🔷 Conversation table: one per message-response pair
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    user_message = Column(String)
    bot_response = Column(String)
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("ChatSession", back_populates="conversations")

# 🔷 function to initialize tables
def init_db():
    Base.metadata.create_all(bind=engine)
