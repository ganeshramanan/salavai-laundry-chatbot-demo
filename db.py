"""
Database layer for the multi-tenant chatbot.

Uses SQLAlchemy + Postgres (Neon free tier). Falls back gracefully is NOT
provided here on purpose -- DATABASE_URL is required once this module is
wired in. See app.py for how these functions replace the old in-memory
dicts/lists (SESSIONS, franchise_leads, live_chat_leads, SITE_KNOWLEDGE_BASE).
"""

import os
import time
import uuid
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Text, Boolean, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Render/Neon sometimes hand out "postgres://" -- SQLAlchemy needs "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine) if engine else None
Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(8), primary_key=True)
    site_id = Column(String(50), nullable=False)
    mode = Column(String(20), default="bot")  # bot | live | franchise
    state = Column(JSON, default=dict)        # franchise flow step/data
    claimed_by = Column(String(100), nullable=True)
    telegram_msg_id = Column(Integer, nullable=True)
    created_at = Column(Float, default=time.time)

    messages = relationship("ChatMessage", backref="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(8), ForeignKey("chat_sessions.id"), nullable=False)
    sender = Column(String(20))  # user | bot | agent
    text = Column(Text)
    time = Column(Float, default=time.time)
    delivered = Column(Boolean, default=False)  # True once sent in a /poll response


class TelegramMsgMap(Base):
    __tablename__ = "telegram_msg_map"
    telegram_msg_id = Column(Integer, primary_key=True)
    session_id = Column(String(8), nullable=False)


class FranchiseLead(Base):
    __tablename__ = "franchise_leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String(50))
    site_name = Column(String(200))
    name = Column(String(200))
    city = Column(String(200))
    experience = Column(Text)
    budget = Column(String(200))
    time = Column(Float, default=time.time)


class LiveChatLead(Base):
    __tablename__ = "live_chat_leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String(50))
    site_name = Column(String(200))
    session_id = Column(String(8))
    message = Column(Text)
    time = Column(Float, default=time.time)


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String(50), nullable=True)  # NULL = core/shared entry
    question = Column(Text)
    answer = Column(Text)
    batch_id = Column(String(32), nullable=True)   # groups chunks from one upload
    filename = Column(String(300), nullable=True)
    uploaded_at = Column(Float, default=time.time)


class NotifyEmail(Base):
    __tablename__ = "notify_emails"
    email = Column(String(300), primary_key=True)


class Site(Base):
    """A tenant/website config -- replaces the old hardcoded SITES dict so new
    websites can be added via /admin without a code change + redeploy."""
    __tablename__ = "sites"
    site_id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    primary_color = Column(String(20), default="#a41e22")
    dark_color = Column(String(20), default="#14324f")
    greeting = Column(Text, default="👋 Hello! How can we assist you today?")
    quick_replies = Column(JSON, default=list)  # list[str]
    created_at = Column(Float, default=time.time)


def init_db():
    """Create all tables if they don't exist yet. Safe to call on every startup."""
    if not engine:
        raise RuntimeError("DATABASE_URL is not set -- cannot initialize DB.")
    Base.metadata.create_all(engine)


def get_db():
    """Returns a new SQLAlchemy session. Caller is responsible for closing it."""
    if not SessionLocal:
        raise RuntimeError("DATABASE_URL is not set -- cannot open a DB session.")
    return SessionLocal()
