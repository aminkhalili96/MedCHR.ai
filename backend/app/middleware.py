"""
Middleware for MedCHR.ai FastAPI application.
Handles security headers, CSRF protection, and other HTTP middleware.
"""

import secrets
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store"
        
        # Content Security Policy
        csp_nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = csp_nonce

        # Swagger/ReDoc use inline scripts and CDN CSS — relax CSP for docs paths.
        _docs_paths = ("/docs", "/redoc", "/openapi.json")
        if request.url.path in _docs_paths:
            csp = (
                "default-src 'self'; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "connect-src 'self'; "
                "object-src 'none'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        else:
            csp = (
                "default-src 'self'; "
                "img-src 'self' data:; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                f"script-src 'self' 'nonce-{csp_nonce}' https://cdn.jsdelivr.net; "
                "connect-src 'self'; "
                "object-src 'none'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )

        response.headers["Content-Security-Policy"] = csp
        
        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # CSRF protection logic would go here
        # For now, just pass through
        return await call_next(request)


def add_security_headers(app):
    """Add security headers middleware to the FastAPI app."""
    app.add_middleware(SecurityHeadersMiddleware)


def add_csrf_protection(app):
    """Add CSRF protection middleware to the FastAPI app."""
    app.add_middleware(CSRFMiddleware)