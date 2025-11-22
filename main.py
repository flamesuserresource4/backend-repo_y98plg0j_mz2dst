import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import (
    User, AuthRegister, AuthLogin, TokenPair, RefreshTokenRequest,
    Document as DocumentModel, UploadResponse, DocumentListItem,
)

# ----------------------------------------------------
# Config & Settings
# ----------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Logging
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme (token in Authorization: Bearer <token>)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

app = FastAPI(title="AI RAG App Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict, expires_days: int = REFRESH_TOKEN_EXPIRE_DAYS):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=expires_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


class TokenPayload(BaseModel):
    sub: str
    exp: int
    type: Optional[str] = "access"


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_data = TokenPayload(**payload)
        if token_data.type != "access":
            raise credentials_exception
        user_id = token_data.sub
    except JWTError:
        raise credentials_exception
    return user_id


# ----------------------------------------------------
# Health
# ----------------------------------------------------
@app.get("/")
def root():
    return {"message": "Backend running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# ----------------------------------------------------
# Auth Endpoints
# ----------------------------------------------------
@app.post("/auth/register", response_model=TokenPair)
def register(payload: AuthRegister):
    existing = db["user"].find_one({"email": payload.email.lower()}) if db else None
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_doc = {
        "email": payload.email.lower(),
        "password_hash": get_password_hash(payload.password),
        "is_admin": False,
        "created_at": datetime.now(timezone.utc),
    }
    res = db["user"].insert_one(user_doc)
    user_id = str(res.inserted_id)

    access = create_access_token({"sub": user_id, "type": "access"})
    refresh = create_refresh_token({"sub": user_id, "type": "refresh"})
    return TokenPair(access_token=access, refresh_token=refresh)


@app.post("/auth/login", response_model=TokenPair)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db["user"].find_one({"email": form_data.username.lower()}) if db else None
    if not user or not verify_password(form_data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user_id = str(user["_id"])
    access = create_access_token({"sub": user_id, "type": "access"})
    refresh = create_refresh_token({"sub": user_id, "type": "refresh"})
    return TokenPair(access_token=access, refresh_token=refresh)


@app.post("/auth/refresh", response_model=TokenPair)
def refresh_tokens(payload: RefreshTokenRequest):
    try:
        data = jwt.decode(payload.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if data.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        user_id = data.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    access = create_access_token({"sub": user_id, "type": "access"})
    refresh = create_refresh_token({"sub": user_id, "type": "refresh"})
    return TokenPair(access_token=access, refresh_token=refresh)


# ----------------------------------------------------
# Documents
# ----------------------------------------------------
@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    token: str = Depends(oauth2_scheme),
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    user_id = get_current_user_id(token)

    filename = f"{uuid.uuid4()}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        size_bytes = os.path.getsize(save_path)
        file_type = (file.filename.split(".")[-1] or "").lower()
        doc = DocumentModel(
            user_id=user_id,
            title=title or file.filename,
            filename=file.filename,
            file_path=save_path,
            file_type=file_type,
            size_bytes=size_bytes,
            status="UPLOADED",
            tags=(tags.split(",") if tags else None),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        doc_id = create_document("document", doc)
        # Normally trigger background ingestion / airflow here
        return UploadResponse(doc_id=doc_id, status="UPLOADED")
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentListItem])
def list_documents(user_id: str = Depends(get_current_user_id)):
    docs = get_documents("document", {"user_id": user_id}) if db else []
    items: List[DocumentListItem] = []
    for d in docs:
        items.append(
            DocumentListItem(
                id=str(d.get("_id")),
                name=d.get("title") or d.get("filename"),
                size=d.get("size_bytes", 0),
                type=d.get("file_type", ""),
                upload_date=d.get("created_at") or datetime.now(timezone.utc),
                status=d.get("status", "UPLOADED"),
                chunks=d.get("chunks", 0) if isinstance(d.get("chunks"), int) else 0,
                last_indexed_time=d.get("last_indexed_time"),
            )
        )
    return items


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, user_id: str = Depends(get_current_user_id)):
    from bson import ObjectId

    try:
        d = db["document"].find_one({"_id": ObjectId(doc_id), "user_id": user_id})
        if not d:
            raise HTTPException(status_code=404, detail="Document not found")
        d["id"] = str(d.pop("_id"))
        return d
    except Exception:
        raise HTTPException(status_code=404, detail="Document not found")


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user_id: str = Depends(get_current_user_id)):
    from bson import ObjectId

    try:
        res = db["document"].update_one(
            {"_id": ObjectId(doc_id), "user_id": user_id},
            {"$set": {"status": "DELETED", "updated_at": datetime.now(timezone.utc)}},
        )
        if res.matched_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"id": doc_id, "status": "DELETED"}
    except Exception:
        raise HTTPException(status_code=404, detail="Document not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
