import os
import json
import pymysql
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASS = os.getenv("MYSQL_PASS", "root")
DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_PORT = os.getenv("MYSQL_PORT", "3306")
DB_NAME = os.getenv("MYSQL_DB", "namma_rag")

# 1. Auto-create database if not exists
def ensure_database_exists():
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, port=int(DB_PORT))
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        conn.commit()
        cursor.close()
        conn.close()
        print(f"MySQL Database '{DB_NAME}' verified!")
    except Exception as e:
        print(f"MySQL Database creation warning: {e}")

ensure_database_exists()

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. Database Models
class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), unique=True, index=True, nullable=False)
    page_count = Column(Integer, default=1)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    session_id = Column(String(100), primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("ChatMessageModel", back_populates="session", cascade="all, delete-orphan")

class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    telemetry_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSessionModel", back_populates="messages")

def init_db():
    Base.metadata.create_all(bind=engine)
    print("MySQL Tables initialized successfully!")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
