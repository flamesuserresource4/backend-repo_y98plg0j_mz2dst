"""
Application Schemas (Pydantic Models)

Each Pydantic model name corresponds to a MongoDB collection with the lowercase name.
Example: class User -> collection "user"
"""
from __future__ import annotations
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Any
from datetime import datetime

# ============ Core/User/Auth ============
class User(BaseModel):
    email: EmailStr
    password_hash: str
    is_admin: bool = False
    created_at: Optional[datetime] = None

class AuthRegister(BaseModel):
    email: EmailStr
    password: str

class AuthLogin(BaseModel):
    email: EmailStr
    password: str

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

# ============ Document & Ingestion ============
class Document(BaseModel):
    user_id: str
    title: str
    filename: str
    file_path: str
    file_type: str
    size_bytes: int
    status: str = Field(description="UPLOADED|PROCESSING|READY|FAILED")
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class DocumentChunk(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    metadata_json: Optional[dict] = None
    created_at: Optional[datetime] = None

class ChunkEmbedding(BaseModel):
    chunk_id: str
    embedding_vector: List[float]
    model_name: str
    created_at: Optional[datetime] = None

class IngestionRun(BaseModel):
    doc_id: str
    user_id: str
    status: str
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    mlflow_run_id: Optional[str] = None

# ============ Chat ============
class ChatSession(BaseModel):
    user_id: str
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ChatMessage(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    role: str  # user|assistant
    content: str
    created_at: Optional[datetime] = None

class ChatMessageContext(BaseModel):
    message_id: str
    chunk_id: str
    relevance_score: float

# ============ Request Models ============
class RefreshTokenRequest(BaseModel):
    refresh_token: str

class UploadResponse(BaseModel):
    doc_id: str
    status: str

class DocumentListItem(BaseModel):
    id: str
    name: str
    size: int
    type: str
    upload_date: datetime
    status: str
    chunks: int = 0
    last_indexed_time: Optional[datetime] = None

class ChatQueryRequest(BaseModel):
    query: str
    agent: str = Field(default="qa", description="qa|summary|analysis|code")
    selected_document_ids: Optional[List[str]] = None

class SummarizeRequest(BaseModel):
    doc_ids: Optional[List[str]] = None
    text: Optional[str] = None

class CodeRequest(BaseModel):
    instructions: str
    schema_or_context: Optional[str] = None

# Backwards-compat for database helper type hinting
class AnyModel(BaseModel):
    model_config = {"extra": "allow"}
