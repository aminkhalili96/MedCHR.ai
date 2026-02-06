from datetime import datetime
from uuid import UUID
from typing import Optional
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi import Request, HTTPException, Depends, status
from psycopg.rows import dict_row
from .db import get_conn, set_actor_context, set_tenant_context
from .ip_whitelist import check_tenant_ip_access, extract_client_ip
from .crypto import decrypt_value, encrypt_value

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    id: UUID
    email: EmailStr
    role: str
    tenant_id: UUID
    created_at: datetime
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str = "clinician"

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user_by_email(email: str, conn) -> Optional[dict]:
    row = conn.execute(
        "SELECT id, email, password_hash, role, tenant_id, created_at, mfa_enabled, mfa_secret, mfa_secret_encrypted FROM users WHERE email = %s",
        (email,)
    ).fetchone()
    return _hydrate_user_record(row)

def get_user_by_id(user_id: str, conn) -> Optional[dict]:
    row = conn.execute(
        "SELECT id, email, password_hash, role, tenant_id, created_at, mfa_enabled, mfa_secret, mfa_secret_encrypted FROM users WHERE id = %s",
        (user_id,)
    ).fetchone()
    return _hydrate_user_record(row)


def _hydrate_user_record(row: Optional[dict]) -> Optional[dict]:
    if not row:
        return None
    data = dict(row)
    encrypted_secret = data.get("mfa_secret_encrypted")
    if encrypted_secret:
        decrypted = decrypt_value(str(encrypted_secret))
        data["mfa_secret"] = decrypted or None
    return data


def encrypt_mfa_secret(secret: str) -> str:
    return encrypt_value(secret)

def authenticate_user(email: str, password: str):
    with get_conn() as conn:
        user = get_user_by_email(email, conn)
        if not user:
            return None
        if not verify_password(password, user["password_hash"]):
            return None
        return user

def generate_totp_secret() -> str:
    import pyotp
    return pyotp.random_base32()

def get_totp_uri(email: str, secret: str, issuer: str = "MedCHR.ai") -> str:
    import pyotp
    return pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name=issuer)

def verify_totp(secret: str, code: str) -> bool:
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.verify(code)

def get_current_user(request: Request) -> User:
    user_id = request.session.get("user_id")
    # Check for partial auth for MFA flow? No, this dependency implies fully managed session.
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    with get_conn() as conn:
        user_data = get_user_by_id(user_id, conn)
        if user_data:
            tenant_id = str(user_data["tenant_id"])
            client_ip = extract_client_ip(request)
            access = check_tenant_ip_access(tenant_id, client_ip, conn)
            if not access.get("allowed", False):
                request.session.clear()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied from this IP address",
                )
    
    if not user_data:
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    set_tenant_context(str(user_data["tenant_id"]))
    set_actor_context(str(user_data["id"]))
    return User(**user_data)

def require_admin(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return user
