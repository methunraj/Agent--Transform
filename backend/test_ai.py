#!/usr/bin/env python3
"""
Test to verify AI processing is working vs direct conversion
"""

import requests
import json

def test_ai_vs_direct():
    base_url = "http://localhost:8000"
    
    # Test data with financial info that should trigger AI processing
    financial_data = {
        "company": "TechCorp Inc",
        "financial_year": "2024",
        "revenue": 5000000,
        "expenses": 3000000,
        "profit": 2000000,
        "currency": "EUR",  # Non-USD currency to test conversion
        "regions": [
            {"name": "Europe", "revenue": 3000000, "currency": "EUR"},
            {"name": "North America", "revenue": 2000000, "currency": "USD"}
        ]
    }
    
    print("🧪 Testing AI Processing vs Direct Conversion")
    print("=" * 50)
    
    # Test 1: AI Processing
    print("\n1. Testing AI-only processing...")
    ai_request = {
        "json_data": json.dumps(financial_data, indent=2),
        "file_name": "ai_test",
        "description": "Financial data with multiple currencies for AI processing",
        "processing_mode": "ai_only",
        "user_id": "test_ai_user",
        "session_id": "test_ai_session"
    }
    
    try:
        response = requests.post(f"{base_url}/process", json=ai_request, timeout=120)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ AI processing successful!")
                print(f"   Processing method: {result.get('processing_method')}")
                print(f"   File ID: {result.get('file_id')}")
                if result.get('ai_analysis'):
                    print(f"   AI analysis length: {len(result['ai_analysis'])} characters")
                    print(f"   AI analysis preview: {result['ai_analysis'][:200]}...")
                else:
                    print("   ⚠️ No AI analysis returned")
            else:
                print(f"❌ AI processing failed: {result.get('error')}")
        else:
            print(f"❌ AI processing HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.Timeout:
        print("❌ AI processing timed out (2 minutes)")
    except Exception as e:
        print(f"❌ AI processing error: {e}")
    
    # Test 2: Direct Processing
    print("\n2. Testing direct-only processing...")
    direct_request = {
        "json_data": json.dumps(financial_data, indent=2),
        "file_name": "direct_test",
        "description": "Same data for direct processing comparison",
        "processing_mode": "direct_only"
    }
    
    try:
        response = requests.post(f"{base_url}/process", json=direct_request, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Direct processing successful!")
                print(f"   Processing method: {result.get('processing_method')}")
                print(f"   File ID: {result.get('file_id')}")
                if result.get('ai_analysis'):
                    print(f"   Unexpected: Direct mode has AI analysis")
                else:
                    print("   ✓ No AI analysis (as expected for direct mode)")
            else:
                print(f"❌ Direct processing failed: {result.get('error')}")
        else:
            print(f"❌ Direct processing HTTP error: {response.status_code}")
    except Exception as e:
        print(f"❌ Direct processing error: {e}")
    
    # Test 3: Auto Mode (should choose AI for this data)
    print("\n3. Testing auto processing...")
    auto_request = {
        "json_data": json.dumps(financial_data, indent=2),
        "file_name": "auto_test",
        "description": "Financial data to test auto mode selection",
        "processing_mode": "auto"
    }
    
    try:
        response = requests.post(f"{base_url}/process", json=auto_request, timeout=120)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Auto processing successful!")
                print(f"   Processing method: {result.get('processing_method')}")
                print(f"   File ID: {result.get('file_id')}")
                if result.get('ai_analysis'):
                    print("   ✓ Auto mode chose AI processing (good!)")
                    print(f"   AI analysis length: {len(result['ai_analysis'])} characters")
                else:
                    print("   Auto mode chose direct processing")
            else:
                print(f"❌ Auto processing failed: {result.get('error')}")
        else:
            print(f"❌ Auto processing HTTP error: {response.status_code}")
    except Exception as e:
        print(f"❌ Auto processing error: {e}")

if __name__ == "__main__":
    test_ai_vs_direct()