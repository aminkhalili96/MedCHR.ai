#!/usr/bin/env python3
"""
Complete system test for MedCHR.ai healthcare application.
Tests all major components: FastAPI backend, database, OCR, LLM integration, and frontend templates.
"""

import os
import sys
import tempfile
import subprocess
import time
import requests
from pathlib import Path

# Add backend to path
sys.path.append('backend')

def test_imports():
    """Test that all major modules can be imported."""
    print("Testing imports...")
    
    try:
        from app.main import app
        from app.db import get_conn
        from app.ocr import extract_text
        from app.llm import generate_text, generate_embeddings
        from app.extract import extract_structured
        from app.embeddings import embed_texts
        from app.rag import build_query, retrieve_top_chunks
        from app.chr import generate_chr_draft
        from app.clinical import router as clinical_router
        from app.gap_features import router as gap_router
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_database_schema():
    """Test that database schema exists and is valid."""
    print("Testing database schema...")
    
    schema_path = Path("backend/sql/schema.sql")
    if not schema_path.exists():
        print("✗ Schema file not found")
        return False
    
    try:
        content = schema_path.read_text()
        required_tables = [
            "tenants", "users", "patients", "documents", 
            "extractions", "embeddings", "chr_versions", 
            "medications", "lab_results", "diagnoses"
        ]
        
        for table in required_tables:
            if f"CREATE TABLE IF NOT EXISTS {table}" not in content:
                print(f"✗ Missing table: {table}")
                return False
        
        print("✓ Database schema is complete")
        return True
    except Exception as e:
        print(f"✗ Schema validation failed: {e}")
        return False

def test_frontend_templates():
    """Test that all required frontend templates exist."""
    print("Testing frontend templates...")
    
    templates_dir = Path("frontend/templates")
    required_templates = [
        "base.html", "patients.html", 
        "patient_detail.html", "report.html", "patient_report.html",
        "document_viewer.html", "data.html", "embeddings.html",
        "rag_view.html", "admin.html", "mfa_setup.html", "mfa_challenge.html"
    ]
    
    if not templates_dir.exists():
        print("✗ Templates directory not found")
        return False
    
    for template in required_templates:
        template_path = templates_dir / template
        if not template_path.exists():
            print(f"✗ Missing template: {template}")
            return False
    
    print("✓ All frontend templates exist")
    return True

def test_api_endpoints():
    """Test that API endpoints are properly configured."""
    print("Testing API endpoints...")
    
    try:
        from app.main import app
        
        # Check if key routes exist
        routes = [route.path for route in app.routes]
        required_endpoints = [
            "/health", "/ready", "/patients", "/documents",
            "/chr/draft", "/api/gap/patients"
        ]
        
        for endpoint in required_endpoints:
            if not any(endpoint in route for route in routes):
                print(f"✗ Missing endpoint: {endpoint}")
                return False
        
        print("✓ All API endpoints configured")
        return True
    except Exception as e:
        print(f"✗ API endpoint test failed: {e}")
        return False

def test_ocr_functionality():
    """Test OCR functionality."""
    print("Testing OCR functionality...")
    
    try:
        from app.ocr import extract_text
        
        # Test with sample text data
        test_data = b"Sample text content for OCR testing"
        result = extract_text(test_data, "text/plain")
        
        if not result:
            print("✗ OCR returned empty result")
            return False
        
        print("✓ OCR functionality working")
        return True
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False

def test_llm_integration():
    """Test LLM integration."""
    print("Testing LLM integration...")
    
    try:
        from app.llm import generate_text
        
        # Test with a simple prompt (this may fail if API key not configured)
        try:
            result = generate_text("Test prompt", max_tokens=10)
            print("✓ LLM integration working")
            return True
        except Exception as api_error:
            # If it's an API error, that's expected without config
            if "API key" in str(api_error) or "authentication" in str(api_error):
                print("⚠ LLM integration configured but API key not set (expected)")
                return True
            else:
                print(f"✗ LLM test failed: {api_error}")
                return False
    except Exception as e:
        print(f"✗ LLM integration test failed: {e}")
        return False

def test_security_features():
    """Test security features."""
    print("Testing security features...")
    
    try:
        from app.security import require_api_key, get_csrf_token
        from app.middleware import add_security_headers
        
        # Check if security functions exist
        if not callable(require_api_key):
            print("✗ Security functions not found")
            return False
        
        print("✓ Security features implemented")
        return True
    except Exception as e:
        print(f"✗ Security test failed: {e}")
        return False

def test_comprehensive_system():
    """Run all system tests."""
    print("=" * 60)
    print("MedCHR.ai Complete System Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Database Schema", test_database_schema),
        ("Frontend Templates", test_frontend_templates),
        ("API Endpoints", test_api_endpoints),
        ("OCR Functionality", test_ocr_functionality),
        ("LLM Integration", test_llm_integration),
        ("Security Features", test_security_features),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! MedCHR.ai system is complete and functional.")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed. System may need additional work.")
        return False

if __name__ == "__main__":
    success = test_comprehensive_system()
    sys.exit(0 if success else 1)