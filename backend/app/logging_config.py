"""
Structured Logging Module for MedCHR.ai

Provides:
- JSON-formatted logs with correlation IDs
- Request context propagation
- Sensitive data redaction
- Log levels appropriate for production

Gap Reference: M01
"""

import logging
import json
import re
import sys
import uuid
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional

# Context variable for request correlation
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
tenant_id_var: ContextVar[Optional[str]] = ContextVar('tenant_id', default=None)


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """
    
    SENSITIVE_KEYS = {
        'password', 'token', 'api_key', 'secret', 'authorization',
        'ssn', 'dob', 'mrn', 'patient_id', 'email', 'phone'
    }
    MESSAGE_PATTERNS = (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
        re.compile(r"\b\+?\d[\d\-\s()]{8,}\d\b"),  # phone-like
    )
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": self._redact_message(record.getMessage()),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "tenant_id": tenant_id_var.get(),
        }
        
        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields (redacted)
        if hasattr(record, 'extra_data'):
            log_data["data"] = self._redact_sensitive(record.extra_data)
        
        return json.dumps(log_data, default=str)
    
    def _redact_sensitive(self, data: Any, depth: int = 0) -> Any:
        """Redact sensitive fields from log data."""
        if depth > 5:  # Prevent infinite recursion
            return "[DEPTH_LIMIT]"
        
        if isinstance(data, dict):
            return {
                k: "[REDACTED]" if k.lower() in self.SENSITIVE_KEYS 
                else self._redact_sensitive(v, depth + 1)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._redact_sensitive(item, depth + 1) for item in data[:10]]
        elif isinstance(data, str) and len(data) > 1000:
            return data[:500] + "... [TRUNCATED]"
        return data

    def _redact_message(self, message: str) -> str:
        redacted = message
        for pattern in self.MESSAGE_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    log_file: Optional[str] = None
):
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, use JSON format; otherwise, human-readable
        log_file: Optional file path for file logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_output:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        ))
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **extra
):
    """Log a message with extra context data."""
    record = logger.makeRecord(
        logger.name,
        level,
        "",
        0,
        message,
        (),
        None
    )
    record.extra_data = extra
    logger.handle(record)


# Convenience functions
def log_info(logger: logging.Logger, message: str, **extra):
    log_with_context(logger, logging.INFO, message, **extra)

def log_error(logger: logging.Logger, message: str, **extra):
    log_with_context(logger, logging.ERROR, message, **extra)

def log_warning(logger: logging.Logger, message: str, **extra):
    log_with_context(logger, logging.WARNING, message, **extra)

def log_debug(logger: logging.Logger, message: str, **extra):
    log_with_context(logger, logging.DEBUG, message, **extra)


# Request context helpers
def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None
):
    """Set context variables for the current request."""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if tenant_id:
        tenant_id_var.set(tenant_id)

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:8]

def clear_request_context():
    """Clear context variables after request."""
    request_id_var.set(None)
    user_id_var.set(None)
    tenant_id_var.set(None)
