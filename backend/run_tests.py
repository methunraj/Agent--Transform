#!/usr/bin/env python3
"""
Simple test runner for IntelliExtract backend
Run this to execute different types of tests
"""

import subprocess
import sys
import os
import time
import requests

def check_server_running(url="http://localhost:8000", timeout=5):
    """Check if the backend server is running"""
    try:
        response = requests.get(f"{url}/", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    print("🧪 Running Comprehensive Test Suite...")
    print("=" * 50)
    
    if not check_server_running():
        print("❌ Backend server is not running!")
        print("Please start the server first:")
        print("   uvicorn app.main:app --reload")
        return False
    
    try:
        result = subprocess.run([sys.executable, "test.py"], cwd=os.path.dirname(__file__))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print("💨 Running Quick Smoke Test...")
    print("=" * 50)
    
    if not check_server_running():
        print("❌ Backend server is not running!")
        return False
    
    try:
        # Test basic endpoints
        base_url = "http://localhost:8000"
        
        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print("❌ Root endpoint failed")
            return False
        
        # Test metrics endpoint
        response = requests.get(f"{base_url}/metrics", timeout=10)
        if response.status_code == 200:
            print("✅ Metrics endpoint working")
        else:
            print("❌ Metrics endpoint failed")
            return False
        
        # Test simple processing
        test_data = {
            "json_data": '{"company": "Test Corp", "revenue": 1000000, "currency": "USD"}',
            "file_name": "smoke_test",
            "description": "Quick smoke test",
            "processing_mode": "direct_only"
        }
        
        response = requests.post(f"{base_url}/process", json=test_data, timeout=30)
        if response.status_code == 200 and response.json().get("success"):
            print("✅ Basic processing working")
            return True
        else:
            print("❌ Basic processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 IntelliExtract Backend Test Runner")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        print("Available test types:")
        print("  smoke    - Quick smoke test (fast)")
        print("  full     - Comprehensive test suite (slow)")
        print("  check    - Just check if server is running")
        print()
        test_type = input("Enter test type (smoke/full/check): ").lower().strip()
    
    if test_type == "check":
        if check_server_running():
            print("✅ Backend server is running and responding")
            sys.exit(0)
        else:
            print("❌ Backend server is not running or not responding")
            sys.exit(1)
    
    elif test_type == "smoke":
        success = run_quick_smoke_test()
        sys.exit(0 if success else 1)
    
    elif test_type == "full":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    
    else:
        print(f"❌ Unknown test type: {test_type}")
        print("Use: smoke, full, or check")
        sys.exit(1)

if __name__ == "__main__":
    main()