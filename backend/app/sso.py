"""
SSO Authentication Module

Provides SAML 2.0 and OIDC integration for enterprise IdP.

Gap Reference: S02
"""

import secrets

from authlib.integrations.starlette_client import OAuth
from starlette.requests import Request
from starlette.responses import RedirectResponse

from .config import get_settings
from .db import get_conn


# OAuth client instance
oauth = OAuth()


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _is_verified_email(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"true", "1", "yes", "y"}
    return False


def _enforce_sso_policy(*, provider: str, user_info: dict) -> None:
    settings = get_settings()
    allowed_providers = _split_csv(settings.sso_allowed_providers)
    if allowed_providers and provider.lower() not in allowed_providers:
        raise ValueError("SSO provider is not allowed")

    email = (user_info.get("email") or "").strip().lower()
    if not email:
        raise ValueError("SSO response missing email claim")
    if settings.sso_require_verified_email and not _is_verified_email(user_info.get("email_verified")):
        raise ValueError("SSO email is not verified")

    allowed_domains = _split_csv(settings.sso_allowed_domains)
    if allowed_domains:
        domain = email.split("@")[1] if "@" in email else ""
        if domain not in allowed_domains:
            raise ValueError("Email domain is not allowed for SSO")


def configure_sso():
    """
    Configure SSO providers based on environment settings.
    Call this during application startup.
    """
    settings = get_settings()
    
    # Configure OIDC provider if enabled
    if settings.oidc_enabled:
        oauth.register(
            name="oidc",
            client_id=settings.oidc_client_id,
            client_secret=settings.oidc_client_secret,
            server_metadata_url=settings.oidc_discovery_url,
            client_kwargs={
                "scope": "openid email profile"
            }
        )
    
    # Configure Azure AD if enabled
    if settings.azure_ad_enabled:
        oauth.register(
            name="azure",
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
            server_metadata_url=f"https://login.microsoftonline.com/{settings.azure_tenant_id}/v2.0/.well-known/openid-configuration",
            client_kwargs={
                "scope": "openid email profile"
            }
        )
    
    # Configure Google Workspace if enabled
    if settings.google_workspace_enabled:
        oauth.register(
            name="google",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={
                "scope": "openid email profile"
            }
        )


async def initiate_sso_login(request: Request, provider: str = "oidc") -> RedirectResponse:
    """
    Initiate SSO login flow by redirecting to IdP.
    """
    settings = get_settings()
    
    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    
    # Get the OAuth client
    client = oauth.create_client(provider)
    if not client:
        raise ValueError(f"SSO provider not configured: {provider}")
    
    # Build callback URL
    callback_url = f"{settings.app_base_url}/auth/callback/{provider}"
    
    # Redirect to IdP
    return await client.authorize_redirect(request, callback_url, state=state)


async def handle_sso_callback(request: Request, provider: str = "oidc") -> dict:
    """
    Handle callback from IdP after user authentication.
    Returns user info dict or raises exception.
    """
    # Verify state
    state = request.query_params.get("state")
    stored_state = request.session.get("oauth_state")
    
    if not state or state != stored_state:
        raise ValueError("Invalid OAuth state - possible CSRF attack")
    
    # Clear state
    request.session.pop("oauth_state", None)
    
    # Get the OAuth client
    client = oauth.create_client(provider)
    if not client:
        raise ValueError(f"SSO provider not configured: {provider}")
    
    # Exchange code for token
    token = await client.authorize_access_token(request)
    
    # Get user info
    user_info = token.get("userinfo")
    if not user_info:
        user_info = await client.userinfo(token=token)

    _enforce_sso_policy(provider=provider, user_info=user_info)

    return {
        "email": (user_info.get("email") or "").strip().lower(),
        "name": user_info.get("name"),
        "sub": user_info.get("sub"),
        "provider": provider,
        "email_verified": user_info.get("email_verified"),
        "token": token
    }


def provision_sso_user(user_info: dict, tenant_id: str = None) -> dict:
    """
    Provision or update a user from SSO.
    Creates user if doesn't exist, updates last login if exists.
    """
    with get_conn() as conn:
        email = (user_info.get("email") or "").strip().lower()
        if not email:
            raise ValueError("Cannot provision SSO user without email")

        # Check if user exists
        existing = conn.execute(
            "SELECT id, tenant_id, role FROM users WHERE email = %s",
            (email,)
        ).fetchone()
        
        if existing:
            # Update last login
            conn.execute("""
                UPDATE users 
                SET last_login = NOW(), sso_provider = %s, sso_sub = %s
                WHERE id = %s
            """, (user_info["provider"], user_info["sub"], existing["id"]))
            conn.commit()
            
            return {
                "user_id": str(existing["id"]),
                "email": email,
                "tenant_id": str(existing["tenant_id"]),
                "role": existing["role"],
                "is_new": False
            }
        
        # Create new user (JIT provisioning)
        # Determine tenant from email domain or use default
        if not tenant_id:
            domain = email.split("@")[1] if "@" in email else None
            if domain:
                tenant_row = conn.execute(
                    "SELECT id, sso_enabled FROM tenants WHERE domain = %s",
                    (domain,)
                ).fetchone()
                if tenant_row and tenant_row.get("sso_enabled", True):
                    tenant_id = str(tenant_row["id"])
                elif tenant_row:
                    raise ValueError("SSO is disabled for tenant")
        
        if not tenant_id:
            raise ValueError("Cannot determine tenant for SSO user")
        tenant_check = conn.execute(
            "SELECT id, sso_enabled FROM tenants WHERE id = %s",
            (tenant_id,),
        ).fetchone()
        if not tenant_check:
            raise ValueError("Unknown tenant for SSO user")
        if not tenant_check.get("sso_enabled", True):
            raise ValueError("SSO is disabled for tenant")
        
        # Create user
        result = conn.execute("""
            INSERT INTO users (
                email, tenant_id, role, 
                sso_provider, sso_sub, 
                created_at, last_login
            )
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            RETURNING id
        """, (
            email, tenant_id, "clinician",
            user_info["provider"], user_info["sub"]
        )).fetchone()
        conn.commit()
        
        return {
            "user_id": str(result["id"]),
            "email": email,
            "tenant_id": tenant_id,
            "role": "clinician",
            "is_new": True
        }


# SAML 2.0 Support (requires python3-saml)
class SAMLConfig:
    """SAML 2.0 configuration helper."""
    
    @staticmethod
    def get_settings(tenant_id: str = None) -> dict:
        """Get SAML settings for a tenant."""
        settings = get_settings()
        
        return {
            "strict": True,
            "debug": settings.app_env != "prod",
            "sp": {
                "entityId": f"{settings.app_base_url}/saml/metadata",
                "assertionConsumerService": {
                    "url": f"{settings.app_base_url}/saml/acs",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                },
                "singleLogoutService": {
                    "url": f"{settings.app_base_url}/saml/slo",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
                },
                "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
            },
            "idp": {
                # These would be loaded from tenant configuration
                "entityId": settings.saml_idp_entity_id,
                "singleSignOnService": {
                    "url": settings.saml_idp_sso_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
                },
                "x509cert": settings.saml_idp_cert
            }
        }
