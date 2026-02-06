"""
Observability Module

Provides metrics collection, distributed tracing, and health checks.

Gap Reference: M02, M03, M04, M05
"""

import time
from typing import Optional, Callable
from contextlib import contextmanager
from functools import wraps
import json

from starlette.requests import Request
from starlette.responses import Response


# ============================================================================
# Metrics Collection (Prometheus-compatible)
# ============================================================================

class MetricsCollector:
    """
    Simple in-memory metrics collector.
    In production, use prometheus_client library.
    """
    
    def __init__(self):
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
    
    def inc_counter(self, name: str, labels: dict = None, value: float = 1):
        """Increment a counter."""
        key = self._make_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, labels: dict = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self.gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: dict = None):
        """Observe a histogram value."""
        key = self._make_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = {"count": 0, "sum": 0, "buckets": {}}
        self.histograms[key]["count"] += 1
        self.histograms[key]["sum"] += value
        
        # Standard buckets
        for bucket in [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]:
            bucket_key = f"le_{bucket}"
            if bucket_key not in self.histograms[key]["buckets"]:
                self.histograms[key]["buckets"][bucket_key] = 0
            if value <= bucket:
                self.histograms[key]["buckets"][bucket_key] += 1
    
    def _make_key(self, name: str, labels: dict = None) -> str:
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for key, value in self.counters.items():
            lines.append(f"{key} {value}")
        
        for key, value in self.gauges.items():
            lines.append(f"{key} {value}")
        
        for key, data in self.histograms.items():
            base_name = key.split("{")[0] if "{" in key else key
            for bucket, count in data["buckets"].items():
                lines.append(f"{base_name}_bucket{{{bucket}}} {count}")
            lines.append(f"{base_name}_count {data['count']}")
            lines.append(f"{base_name}_sum {data['sum']}")
        
        return "\n".join(lines)


# Global metrics collector
metrics = MetricsCollector()


# Standard application metrics
def record_request(method: str, path: str, status: int, duration: float):
    """Record HTTP request metrics."""
    labels = {"method": method, "path": path, "status": str(status)}
    metrics.inc_counter("http_requests_total", labels)
    metrics.observe_histogram("http_request_duration_seconds", duration, labels)


def record_extraction(document_type: str, success: bool, duration: float):
    """Record document extraction metrics."""
    labels = {"document_type": document_type, "success": str(success)}
    metrics.inc_counter("extractions_total", labels)
    metrics.observe_histogram("extraction_duration_seconds", duration, labels)


def record_llm_request(model: str, tokens: int, duration: float):
    """Record LLM API call metrics."""
    labels = {"model": model}
    metrics.inc_counter("llm_requests_total", labels)
    metrics.inc_counter("llm_tokens_total", labels, value=tokens)
    metrics.observe_histogram("llm_request_duration_seconds", duration, labels)


# ============================================================================
# Distributed Tracing (OpenTelemetry-compatible)
# ============================================================================

class Span:
    """Simple span for tracing."""
    
    def __init__(self, name: str, trace_id: str, parent_id: str = None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = self._generate_id()
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time = None
        self.status = "OK"
        self.attributes = {}
        self.events = []
    
    def _generate_id(self) -> str:
        import secrets
        return secrets.token_hex(8)
    
    def set_attribute(self, key: str, value):
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: dict = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def end(self, status: str = None):
        self.end_time = time.time()
        if status:
            self.status = status
    
    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events
        }


class Tracer:
    """Simple tracer for distributed tracing."""
    
    def __init__(self):
        self.spans = []
    
    @contextmanager
    def start_span(self, name: str, trace_id: str = None, parent_id: str = None):
        """Start a new span."""
        import secrets
        if not trace_id:
            trace_id = secrets.token_hex(16)
        
        span = Span(name, trace_id, parent_id)
        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.end("ERROR")
            raise
        finally:
            span.end()
            self.spans.append(span)
            # In production, export to tracing backend
    
    def trace(self, name: str):
        """Decorator for tracing functions."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.start_span(name) as span:
                    result = func(*args, **kwargs)
                    return result
            return wrapper
        return decorator


# Global tracer
tracer = Tracer()


# ============================================================================
# Health Checks
# ============================================================================

def check_database_health(conn) -> dict:
    """Check database connectivity."""
    try:
        start = time.time()
        conn.execute("SELECT 1").fetchone()
        duration = time.time() - start
        return {
            "status": "healthy",
            "latency_ms": round(duration * 1000, 2)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_openai_health() -> dict:
    """Check OpenAI API availability."""
    from .config import get_settings
    from .llm_gateway import get_openai_client

    settings = get_settings()
    
    if not settings.openai_api_key:
        return {"status": "not_configured"}
    
    try:
        client = get_openai_client()
        start = time.time()
        client.models.list()
        duration = time.time() - start
        return {
            "status": "healthy",
            "latency_ms": round(duration * 1000, 2)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_storage_health() -> dict:
    """Check storage backend availability."""
    from .config import get_settings
    settings = get_settings()
    
    try:
        from .storage import list_bucket
        start = time.time()
        list_bucket(settings.storage_bucket, limit=1)
        duration = time.time() - start
        return {
            "status": "healthy",
            "latency_ms": round(duration * 1000, 2)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def get_health_status(conn) -> dict:
    """Get comprehensive health status."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {
            "database": check_database_health(conn),
            "openai": check_openai_health(),
            # "storage": check_storage_health(),
        },
        "version": "1.0.0"
    }


# ============================================================================
# Alerting
# ============================================================================

class AlertManager:
    """Simple alert manager."""
    
    def __init__(self):
        self.alerts = []
        self.webhooks = []
    
    def add_webhook(self, url: str, events: list = None):
        """Add a webhook for alerts."""
        self.webhooks.append({
            "url": url,
            "events": events or ["error", "critical"]
        })
    
    def send_alert(self, level: str, title: str, message: str, context: dict = None):
        """Send an alert to configured destinations."""
        alert = {
            "level": level,
            "title": title,
            "message": message,
            "context": context or {},
            "timestamp": time.time()
        }
        self.alerts.append(alert)
        
        # Send to webhooks (async in production)
        for webhook in self.webhooks:
            if level in webhook["events"]:
                self._send_webhook(webhook["url"], alert)
    
    def _send_webhook(self, url: str, alert: dict):
        """Send alert to webhook (placeholder)."""
        # In production, use httpx or aiohttp
        print(f"[ALERT] Would send to {url}: {alert['title']}")


# Global alert manager
alerts = AlertManager()
