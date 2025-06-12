#!/usr/bin/env python3
"""
Quick test to verify the fixes work
Run this after starting the server to test basic functionality
"""

import requests
import json

def test_server_quick():
    base_url = "http://localhost:8000"
    
    try:
        # Test 1: Root endpoint
        print("🧪 Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
        
        # Test 2: Metrics endpoint
        print("🧪 Testing metrics endpoint...")
        response = requests.get(f"{base_url}/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Metrics endpoint working")
            metrics = response.json()
            print(f"   Total requests: {metrics.get('total_requests', 0)}")
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
        
        # Test 3: Simple direct conversion
        print("🧪 Testing direct conversion...")
        test_data = {
            "json_data": '{"company": "Test Corp", "revenue": 1000000, "currency": "USD"}',
            "file_name": "quick_test",
            "description": "Quick functionality test",
            "processing_mode": "direct_only"
        }
        
        response = requests.post(f"{base_url}/process", json=test_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Direct conversion working")
                print(f"   File ID: {result.get('file_id')}")
                print(f"   Processing method: {result.get('processing_method')}")
                return True
            else:
                print(f"❌ Direct conversion failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Direct conversion request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Quick Backend Test")
    print("=" * 30)
    
    success = test_server_quick()
    
    if success:
        print("\n🎉 All tests passed! The fixes are working.")
    else:
        print("\n💥 Some tests failed. Check the server logs.")