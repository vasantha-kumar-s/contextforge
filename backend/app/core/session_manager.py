import json
import uuid
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, ChatSessionModel, ChatMessageModel

class ChatSessionManager:
    """
    Manages chat sessions and message history using MySQL database.
    """

    def __init__(self):
        self._ensure_default_session()

    def _ensure_default_session(self):
        db: Session = SessionLocal()
        try:
            default = db.query(ChatSessionModel).filter(ChatSessionModel.session_id == "default_session").first()
            if not default:
                default_session = ChatSessionModel(
                    session_id="default_session",
                    title="Namma-RAG Workspace"
                )
                db.add(default_session)
                db.commit()
        except Exception as e:
            print(f"Error ensuring default session: {e}")
            db.rollback()
        finally:
            db.close()

    def create_session(self, title: Optional[str] = None) -> Dict[str, Any]:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        session_title = title or "New Session"
        db: Session = SessionLocal()
        try:
            s_model = ChatSessionModel(session_id=session_id, title=session_title)
            db.add(s_model)
            db.commit()
            db.refresh(s_model)
            return {
                "session_id": s_model.session_id,
                "title": s_model.title,
                "created_at": s_model.created_at.timestamp() if s_model.created_at else time.time(),
                "messages": []
            }
        finally:
            db.close()

    def list_sessions(self) -> List[Dict[str, Any]]:
        db: Session = SessionLocal()
        try:
            sessions = db.query(ChatSessionModel).order_by(ChatSessionModel.created_at.desc()).all()
            return [
                {
                    "session_id": s.session_id,
                    "title": s.title,
                    "created_at": s.created_at.timestamp() if s.created_at else time.time(),
                    "message_count": len(s.messages)
                }
                for s in sessions
            ]
        finally:
            db.close()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        db: Session = SessionLocal()
        try:
            s = db.query(ChatSessionModel).filter(ChatSessionModel.session_id == session_id).first()
            if not s:
                return None
            
            messages = []
            for m in s.messages:
                messages.append({
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "citations": json.loads(m.citations_json) if m.citations_json else [],
                    "telemetry": json.loads(m.telemetry_json) if m.telemetry_json else {},
                    "timestamp": m.created_at.timestamp() if m.created_at else time.time()
                })

            return {
                "session_id": s.session_id,
                "title": s.title,
                "created_at": s.created_at.timestamp() if s.created_at else time.time(),
                "messages": messages
            }
        finally:
            db.close()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        db: Session = SessionLocal()
        try:
            s = db.query(ChatSessionModel).filter(ChatSessionModel.session_id == session_id).first()
            if not s:
                s = ChatSessionModel(session_id=session_id, title=content[:30] if role == "user" else "New Session")
                db.add(s)
                db.commit()

            # Auto update title if first user message
            if role == "user" and len(s.messages) == 0:
                s.title = content[:30] + ("..." if len(content) > 30 else "")

            msg = ChatMessageModel(
                session_id=session_id,
                role=role,
                content=content,
                citations_json=json.dumps(citations) if citations else None,
                telemetry_json=json.dumps(telemetry) if telemetry else None
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)

            return {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "citations": citations or [],
                "telemetry": telemetry or {},
                "timestamp": msg.created_at.timestamp() if msg.created_at else time.time()
            }
        finally:
            db.close()

    def get_history(self, session_id: str, limit: int = 6) -> List[Dict[str, str]]:
        db: Session = SessionLocal()
        try:
            msgs = db.query(ChatMessageModel).filter(ChatMessageModel.session_id == session_id).order_by(ChatMessageModel.id.desc()).limit(limit).all()
            msgs.reverse()
            return [{"role": m.role, "content": m.content} for m in msgs]
        finally:
            db.close()

    def delete_session(self, session_id: str):
        db: Session = SessionLocal()
        try:
            s = db.query(ChatSessionModel).filter(ChatSessionModel.session_id == session_id).first()
            if s:
                db.delete(s)
                db.commit()
        finally:
            db.close()
