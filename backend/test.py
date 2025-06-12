#!/usr/bin/env python3
"""
Comprehensive Test Suite for IntelliExtract Agno AI Backend
Tests various JSON data structures and processing modes
"""

import requests
import json
import time
import uuid
from typing import Dict, Any, List
from datetime import datetime, timedelta
import random

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = f"test_user_{uuid.uuid4().hex[:8]}"

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_test_header(test_name: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}TEST: {test_name}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_success(message: str):
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_error(message: str):
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_warning(message: str):
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_info(message: str):
    print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")

# Test Data Structures

# 1. VALID JSON TEST CASES
VALID_JSON_TESTS = {
    "simple_financial_dict": {
        "data": {
            "company": "TechCorp Inc",
            "revenue": 1500000,
            "expenses": 800000,
            "profit": 700000,
            "currency": "USD",
            "quarter": "Q4 2024"
        },
        "description": "Simple financial dictionary with basic metrics"
    },
    
    "financial_array": {
        "data": [
            {"date": "2024-01-01", "revenue": 100000, "currency": "EUR"},
            {"date": "2024-02-01", "revenue": 110000, "currency": "EUR"},
            {"date": "2024-03-01", "revenue": 120000, "currency": "EUR"}
        ],
        "description": "Array of monthly financial data"
    },
    
    "complex_financial_structure": {
        "data": {
            "company_info": {
                "name": "GlobalTech Solutions",
                "country": "Germany",
                "currency": "EUR",
                "employees": 250
            },
            "financial_data": {
                "income_statement": [
                    {"item": "Revenue", "q1": 2500000, "q2": 2750000, "q3": 3000000, "q4": 3250000},
                    {"item": "Cost of Goods Sold", "q1": 1000000, "q2": 1100000, "q3": 1200000, "q4": 1300000},
                    {"item": "Gross Profit", "q1": 1500000, "q2": 1650000, "q3": 1800000, "q4": 1950000}
                ],
                "balance_sheet": {
                    "assets": {
                        "current_assets": {"cash": 500000, "inventory": 300000, "accounts_receivable": 200000},
                        "fixed_assets": {"equipment": 1000000, "real_estate": 2000000}
                    },
                    "liabilities": {
                        "current_liabilities": {"accounts_payable": 150000, "short_term_debt": 100000},
                        "long_term_liabilities": {"long_term_debt": 800000}
                    }
                },
                "cash_flow": [
                    {"month": "Jan", "operating": 200000, "investing": -50000, "financing": -25000},
                    {"month": "Feb", "operating": 220000, "investing": -30000, "financing": 0},
                    {"month": "Mar", "operating": 240000, "investing": -100000, "financing": 50000}
                ]
            },
            "exchange_rates": {
                "EUR_to_USD": 1.08,
                "effective_date": "2024-12-06"
            }
        },
        "description": "Complex nested financial structure with multiple components"
    },
    
    "multi_currency_data": {
        "data": [
            {"region": "North America", "revenue": 5000000, "currency": "USD", "expenses": 3000000},
            {"region": "Europe", "revenue": 4500000, "currency": "EUR", "expenses": 2800000},
            {"region": "Asia Pacific", "revenue": 3200000, "currency": "JPY", "expenses": 2000000},
            {"region": "Latin America", "revenue": 1800000, "currency": "BRL", "expenses": 1200000}
        ],
        "description": "Multi-regional data with different currencies"
    },
    
    "time_series_financial": {
        "data": {
            "stock_data": [
                {"date": "2024-01-01", "open": 150.00, "high": 155.00, "low": 148.00, "close": 153.00, "volume": 1000000, "currency": "USD"},
                {"date": "2024-01-02", "open": 153.00, "high": 158.00, "low": 151.00, "close": 156.00, "volume": 1200000, "currency": "USD"},
                {"date": "2024-01-03", "open": 156.00, "high": 160.00, "low": 154.00, "close": 159.00, "volume": 950000, "currency": "USD"}
            ],
            "company_metrics": {
                "market_cap": 15000000000,
                "pe_ratio": 18.5,
                "dividend_yield": 2.1,
                "currency": "USD"
            }
        },
        "description": "Time series stock data with company metrics"
    },
    
    "large_dataset": {
        "data": [
            {
                "transaction_id": f"TXN_{i:06d}",
                "amount": round(random.uniform(100, 10000), 2),
                "currency": random.choice(["USD", "EUR", "GBP", "JPY"]),
                "category": random.choice(["Sales", "Marketing", "Operations", "R&D"]),
                "date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
            }
            for i in range(1, 501)  # 500 transactions
        ],
        "description": "Large dataset with 500 financial transactions"
    },
    
    "unicode_and_special_chars": {
        "data": {
            "company": "Ñandú Technologies™",
            "description": "AI company with émphasis on innovation & growth 🚀",
            "financials": {
                "revenue_€": 2000000,
                "profit_¥": 150000000,
                "growth_%": 25.5
            },
            "locations": ["São Paulo", "München", "東京", "Москва"]
        },
        "description": "Data with Unicode characters and special symbols"
    }
}

# 2. INVALID JSON TEST CASES
INVALID_JSON_TESTS = {
    "malformed_json_missing_quotes": {
        "data": '{"company": TechCorp, "revenue": 1000000}',
        "description": "JSON with missing quotes around string value"
    },
    
    "malformed_json_trailing_comma": {
        "data": '{"company": "TechCorp", "revenue": 1000000,}',
        "description": "JSON with trailing comma"
    },
    
    "malformed_json_missing_bracket": {
        "data": '{"company": "TechCorp", "revenue": 1000000',
        "description": "JSON with missing closing bracket"
    },
    
    "completely_invalid_json": {
        "data": 'this is not json at all, just random text with numbers 123 and symbols !@#',
        "description": "Completely invalid JSON - just text"
    },
    
    "mixed_valid_invalid_json": {
        "data": '{"valid": "data"} invalid text {"more": "data"}',
        "description": "Mix of valid JSON objects and invalid text"
    },
    
    "empty_string": {
        "data": '',
        "description": "Empty string"
    },
    
    "only_whitespace": {
        "data": '   \n\t  ',
        "description": "Only whitespace characters"
    },
    
    "json_with_undefined": {
        "data": '{"company": "TechCorp", "revenue": undefined, "profit": null}',
        "description": "JSON with undefined value (JavaScript-style)"
    }
}

# 3. EDGE CASES
EDGE_CASE_TESTS = {
    "very_large_numbers": {
        "data": {
            "market_cap": 999999999999999999999,
            "revenue": 1.23456789012345e+20,
            "microscopic_value": 1.23e-15,
            "currency": "USD"
        },
        "description": "Very large and very small numbers"
    },
    
    "null_and_empty_values": {
        "data": {
            "company": "Test Corp",
            "revenue": None,
            "expenses": 0,
            "description": "",
            "tags": [],
            "metadata": {},
            "currency": "USD"
        },
        "description": "Null and empty values"
    },
    
    "deeply_nested_structure": {
        "data": {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "deep_value": "Found at level 5!",
                                "financial_data": {"revenue": 1000000, "currency": "USD"}
                            }
                        }
                    }
                }
            }
        },
        "description": "Deeply nested JSON structure"
    },
    
    "array_of_mixed_types": {
        "data": [
            {"type": "revenue", "value": 1000000, "currency": "USD"},
            {"type": "expense", "value": 500000, "currency": "USD"},
            "summary_note",
            123456,
            True,
            None,
            ["nested", "array", "inside"]
        ],
        "description": "Array containing mixed data types"
    }
}

def test_api_endpoint(test_name: str, json_data: Any, description: str, processing_mode: str = "auto") -> Dict:
    """Test a single API endpoint with given data"""
    
    # Convert data to JSON string if it's not already
    if isinstance(json_data, str):
        json_string = json_data
    else:
        json_string = json.dumps(json_data, indent=2)
    
    payload = {
        "json_data": json_string,
        "file_name": f"test_{test_name}",
        "description": description,
        "model": "gemini-2.5-flash-preview-05-20",
        "processing_mode": processing_mode,
        "chunk_size": 1000,
        "user_id": TEST_USER_ID,
        "session_id": f"session_{test_name}"
    }
    
    try:
        print_info(f"Testing: {test_name}")
        print_info(f"Description: {description}")
        print_info(f"Processing mode: {processing_mode}")
        print_info(f"Data preview: {json_string[:200]}{'...' if len(json_string) > 200 else ''}")
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/process", json=payload, timeout=60)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        result = {
            "test_name": test_name,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "response_size": len(response.content) if response.content else 0,
            "success": False,
            "error": None,
            "response_data": None
        }
        
        if response.status_code == 200:
            response_data = response.json()
            result["success"] = response_data.get("success", False)
            result["response_data"] = response_data
            
            if result["success"]:
                print_success(f"✓ Success - Processing time: {processing_time:.2f}s")
                if response_data.get("file_id"):
                    print_info(f"  File ID: {response_data['file_id']}")
                    print_info(f"  Download URL: {response_data.get('download_url', 'N/A')}")
                if response_data.get("processing_method"):
                    print_info(f"  Processing method: {response_data['processing_method']}")
            else:
                print_warning(f"⚠ API returned success=false")
                if response_data.get("error"):
                    print_error(f"  Error: {response_data['error']}")
        else:
            print_error(f"✗ HTTP {response.status_code} - {response.text[:500]}")
            result["error"] = response.text
            
    except requests.exceptions.Timeout:
        print_error("✗ Request timeout (60s)")
        result = {
            "test_name": test_name,
            "status_code": 408,
            "processing_time": 60.0,
            "success": False,
            "error": "Request timeout"
        }
    except requests.exceptions.RequestException as e:
        print_error(f"✗ Request failed: {e}")
        result = {
            "test_name": test_name,
            "status_code": 0,
            "processing_time": 0,
            "success": False,
            "error": str(e)
        }
    
    return result

def test_metrics_endpoint():
    """Test the metrics endpoint"""
    print_test_header("System Metrics Test")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            print_success("Metrics endpoint working")
            print_info(f"Total requests: {metrics.get('total_requests', 'N/A')}")
            print_info(f"Success rate: {metrics.get('success_rate', 'N/A')}%")
            print_info(f"Average processing time: {metrics.get('average_processing_time', 'N/A')}s")
            print_info(f"Active files: {metrics.get('active_files', 'N/A')}")
            return True
        else:
            print_error(f"Metrics endpoint failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Metrics endpoint error: {e}")
        return False

def run_test_suite():
    """Run the complete test suite"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("IntelliExtract Agno AI Backend - Comprehensive Test Suite")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print_success("✓ Backend server is running")
        else:
            print_error("✗ Backend server returned unexpected status")
            return
    except:
        print_error("✗ Cannot connect to backend server. Please start the server first:")
        print_info("  uvicorn app.main:app --reload")
        return
    
    print_info(f"Test User ID: {TEST_USER_ID}")
    print_info(f"Base URL: {BASE_URL}")
    
    all_results = []
    
    # Test metrics endpoint first
    test_metrics_endpoint()
    
    # 1. Test Valid JSON Cases
    print_test_header("Valid JSON Test Cases")
    for test_name, test_data in VALID_JSON_TESTS.items():
        result = test_api_endpoint(
            test_name, 
            test_data["data"], 
            test_data["description"]
        )
        all_results.append(result)
        time.sleep(1)  # Small delay between tests
    
    # 2. Test Invalid JSON Cases (Direct mode only - faster)
    print_test_header("Invalid JSON Test Cases (Direct Mode)")
    for test_name, test_data in INVALID_JSON_TESTS.items():
        result = test_api_endpoint(
            test_name, 
            test_data["data"], 
            test_data["description"],
            processing_mode="direct_only"
        )
        all_results.append(result)
        time.sleep(0.5)  # Shorter delay for quick tests
    
    # 3. Test Edge Cases
    print_test_header("Edge Case Test Cases")
    for test_name, test_data in EDGE_CASE_TESTS.items():
        result = test_api_endpoint(
            test_name, 
            test_data["data"], 
            test_data["description"]
        )
        all_results.append(result)
        time.sleep(1)
    
    # 4. Test Different Processing Modes with Valid Data
    print_test_header("Processing Mode Tests")
    sample_data = VALID_JSON_TESTS["simple_financial_dict"]
    
    for mode in ["auto", "ai_only", "direct_only"]:
        result = test_api_endpoint(
            f"processing_mode_{mode}",
            sample_data["data"],
            f"Test {mode} processing mode",
            processing_mode=mode
        )
        all_results.append(result)
        time.sleep(1)
    
    # Final Summary
    print_test_header("Test Results Summary")
    
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r["success"])
    failed_tests = total_tests - successful_tests
    
    print_info(f"Total Tests: {total_tests}")
    print_success(f"Successful: {successful_tests}")
    print_error(f"Failed: {failed_tests}")
    print_info(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Average processing time for successful tests
    successful_times = [r["processing_time"] for r in all_results if r["success"]]
    if successful_times:
        avg_time = sum(successful_times) / len(successful_times)
        print_info(f"Average Processing Time: {avg_time:.2f}s")
    
    # List failed tests
    if failed_tests > 0:
        print_warning("\nFailed Tests:")
        for result in all_results:
            if not result["success"]:
                print_error(f"  - {result['test_name']}: {result.get('error', 'Unknown error')}")
    
    # Test metrics endpoint again to see updated stats
    print_test_header("Final Metrics Check")
    test_metrics_endpoint()
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}Test Suite Complete!{Colors.ENDC}")

if __name__ == "__main__":
    run_test_suite()