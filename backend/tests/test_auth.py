from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime
from backend.app.auth import authenticate_user, get_password_hash, User

def test_authenticate_user_success():
    email = "test@example.com"
    password = "securepassword"
    pw_hash = get_password_hash(password)
    user_id = uuid4()
    tenant_id = uuid4()
    
    expected_user = {
        "id": user_id,
        "email": email,
        "password_hash": pw_hash,
        "role": "clinician",
        "tenant_id": tenant_id,
        "created_at": datetime.now()
    }
    
    with patch("backend.app.auth.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        # Mock fetchone to return our user
        mock_conn.execute.return_value.fetchone.return_value = expected_user
        
        user = authenticate_user(email, password)
        
        assert user is not None
        assert user["email"] == email
        assert user["id"] == user_id

def test_authenticate_user_wrong_password():
    email = "test@example.com"
    password = "securepassword"
    pw_hash = get_password_hash("differentpassword")
    
    user_db = {
        "id": uuid4(),
        "email": email,
        "password_hash": pw_hash,
        "role": "clinician",
        "tenant_id": uuid4(),
        "created_at": datetime.now()
    }
    
    with patch("backend.app.auth.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        mock_conn.execute.return_value.fetchone.return_value = user_db
        
        user = authenticate_user(email, "WRONGPASSWORD")
        
        assert user is None

def test_authenticate_user_not_found():
    with patch("backend.app.auth.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        mock_conn.execute.return_value.fetchone.return_value = None
        
        user = authenticate_user("nonexistent@example.com", "any")
        
        assert user is None
