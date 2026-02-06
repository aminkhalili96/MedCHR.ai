"""
IP Whitelist Enforcement Module

Enforces IP-based access control for tenants.

Gap Reference: S04
"""

import ipaddress
from typing import List, Optional


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_candidate_ip(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    if parse_ip(candidate):
        return candidate
    # Handle "ip:port" for IPv4 sources in forwarding headers.
    if "." in candidate and candidate.count(":") == 1:
        maybe_ip = candidate.rsplit(":", 1)[0]
        if parse_ip(maybe_ip):
            return maybe_ip
    return candidate


def _is_trusted_proxy(remote_ip: str, trusted_entries: list[str]) -> bool:
    remote = parse_ip(remote_ip)
    if not remote:
        return False
    for entry in trusted_entries:
        trusted_ip = parse_ip(entry)
        if trusted_ip and trusted_ip == remote:
            return True
        network = parse_cidr(entry)
        if network and remote in network:
            return True
    return False


def extract_client_ip(request) -> str:
    """
    Extract client IP safely.

    Proxy-provided headers are only trusted when:
    - TRUST_PROXY_HEADERS=true, and
    - source socket IP is in TRUSTED_PROXY_IPS.
    """
    from .config import get_settings

    client = getattr(request, "client", None)
    remote_ip = getattr(client, "host", "0.0.0.0")
    settings = get_settings()
    if not settings.trust_proxy_headers:
        return remote_ip
    trusted_entries = _split_csv(settings.trusted_proxy_ips)
    if not trusted_entries or not _is_trusted_proxy(remote_ip, trusted_entries):
        return remote_ip

    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        for item in forwarded.split(","):
            candidate = _normalize_candidate_ip(item)
            parsed = parse_ip(candidate)
            if parsed:
                return str(parsed)
    real_ip = _normalize_candidate_ip(request.headers.get("x-real-ip", ""))
    parsed_real = parse_ip(real_ip)
    if parsed_real:
        return str(parsed_real)
    return remote_ip


def parse_ip(ip_string: str) -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """
    Parse an IP address string.
    """
    try:
        return ipaddress.ip_address(ip_string)
    except ValueError:
        return None


def parse_cidr(cidr_string: str) -> Optional[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """
    Parse a CIDR block string.
    """
    try:
        return ipaddress.ip_network(cidr_string, strict=False)
    except ValueError:
        return None


def is_ip_allowed(
    client_ip: str,
    whitelist: List[str],
    allow_empty: bool = True
) -> bool:
    """
    Check if client IP is in the whitelist.
    
    Args:
        client_ip: Client's IP address
        whitelist: List of IPs or CIDR blocks
        allow_empty: If True, empty whitelist allows all
        
    Returns:
        True if allowed, False otherwise
    """
    if not whitelist:
        return allow_empty
    
    ip = parse_ip(client_ip)
    if not ip:
        return False
    
    for entry in whitelist:
        # Check if it's a single IP
        allowed_ip = parse_ip(entry)
        if allowed_ip and ip == allowed_ip:
            return True
        
        # Check if it's a CIDR block
        network = parse_cidr(entry)
        if network and ip in network:
            return True
    
    return False


def get_tenant_whitelist(tenant_id: str, conn = None) -> List[str]:
    """
    Get IP whitelist for a tenant.
    """
    from .db import get_conn
    import json
    
    if conn is None:
        conn = get_conn()
    
    row = conn.execute(
        "SELECT ip_whitelist FROM tenants WHERE id = %s",
        (tenant_id,)
    ).fetchone()
    
    if not row or not row["ip_whitelist"]:
        return []
    
    whitelist = row["ip_whitelist"]
    if isinstance(whitelist, str):
        whitelist = json.loads(whitelist)
    
    return whitelist


def update_tenant_whitelist(
    tenant_id: str,
    whitelist: List[str],
    conn = None
):
    """
    Update IP whitelist for a tenant.
    """
    from .db import get_conn
    import json
    
    # Validate all entries
    for entry in whitelist:
        if not parse_ip(entry) and not parse_cidr(entry):
            raise ValueError(f"Invalid IP or CIDR: {entry}")
    
    if conn is None:
        conn = get_conn()
    
    conn.execute(
        "UPDATE tenants SET ip_whitelist = %s WHERE id = %s",
        (json.dumps(whitelist), tenant_id)
    )
    conn.commit()


def check_tenant_ip_access(
    tenant_id: str,
    client_ip: str,
    conn = None
) -> dict:
    """
    Check if client IP is allowed for tenant.
    
    Returns dict with allowed status and details.
    """
    whitelist = get_tenant_whitelist(tenant_id, conn)
    allowed = is_ip_allowed(client_ip, whitelist)
    
    return {
        "allowed": allowed,
        "client_ip": client_ip,
        "whitelist_enabled": len(whitelist) > 0,
        "whitelist_count": len(whitelist)
    }


class IPWhitelistMiddleware:
    """
    ASGI middleware for IP whitelist enforcement.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Do not trust forwarded headers at raw ASGI layer.
        # Route-level checks use `extract_client_ip`, which applies proxy trust boundaries.
        client = scope.get("client") or ("0.0.0.0", 0)
        _client_ip = client[0]
        
        # Get tenant from session or skip check
        # This would need to integrate with auth middleware
        # For now, we'll check at the route level
        
        return await self.app(scope, receive, send)


def require_tenant_ip(func):
    """
    Decorator to require IP whitelist check for a route.
    """
    from functools import wraps
    from fastapi import Request, HTTPException
    
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        # Get tenant from session
        tenant_id = request.session.get("tenant_id")
        if not tenant_id:
            return await func(request, *args, **kwargs)
        
        # Get client IP using trusted-proxy extraction rules.
        client_ip = extract_client_ip(request)
        
        # Check whitelist
        from .db import get_conn
        with get_conn() as conn:
            result = check_tenant_ip_access(tenant_id, client_ip, conn)
        
        if not result["allowed"]:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: IP {client_ip} not in allowed list"
            )
        
        return await func(request, *args, **kwargs)
    
    return wrapper
