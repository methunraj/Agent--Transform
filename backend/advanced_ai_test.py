#!/usr/bin/env python3
"""
Advanced AI Test Suite for IntelliExtract Backend
Tests complex financial scenarios with automatic file downloads
"""

import requests
import json
import time
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

# Configuration
BASE_URL = "http://localhost:8000"
DOWNLOAD_DIR = "downloaded_files"
TEST_USER_ID = f"advanced_test_user_{uuid.uuid4().hex[:8]}"

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_test_header(test_name: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}🧪 ADVANCED AI TEST: {test_name}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")

def print_success(message: str):
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_error(message: str):
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def print_warning(message: str):
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")

def print_info(message: str):
    print(f"{Colors.OKBLUE}ℹ️  {message}{Colors.ENDC}")

def print_ai_insight(message: str):
    print(f"{Colors.HEADER}🤖 AI INSIGHT: {message}{Colors.ENDC}")

def ensure_download_dir():
    """Ensure download directory exists"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print_info(f"Created download directory: {DOWNLOAD_DIR}")

def download_file(file_id: str, filename: str) -> bool:
    """Download a file from the API"""
    try:
        download_url = f"{BASE_URL}/download/{file_id}"
        response = requests.get(download_url, timeout=30)
        
        if response.status_code == 200:
            file_path = os.path.join(DOWNLOAD_DIR, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print_success(f"Downloaded: {file_path} ({len(response.content)} bytes)")
            return True
        else:
            print_error(f"Download failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Download error: {e}")
        return False

# Complex Test Data Structures

def generate_multinational_corporation_data():
    """Generate complex multinational corporation financial data"""
    return {
        "company_info": {
            "name": "GlobalTech Dynamics Corp",
            "headquarters": "New York, USA",
            "founded": "1995",
            "employees": 125000,
            "ceo": "Sarah Chen",
            "fiscal_year_end": "December 31, 2024",
            "stock_symbol": "GTDC",
            "market_cap_usd": 250000000000
        },
        "financial_summary": {
            "total_revenue_usd": 75000000000,
            "total_expenses_usd": 58000000000,
            "net_income_usd": 17000000000,
            "ebitda_usd": 22000000000,
            "free_cash_flow_usd": 15000000000,
            "debt_to_equity_ratio": 0.35,
            "return_on_equity": 0.18
        },
        "regional_performance": [
            {
                "region": "North America",
                "revenue": 35000000000,
                "currency": "USD",
                "growth_rate": 0.12,
                "market_share": 0.28,
                "employees": 45000,
                "major_cities": ["New York", "San Francisco", "Toronto", "Mexico City"]
            },
            {
                "region": "Europe",
                "revenue": 28000000000,
                "currency": "EUR",
                "growth_rate": 0.08,
                "market_share": 0.22,
                "employees": 35000,
                "major_cities": ["London", "Berlin", "Paris", "Amsterdam"]
            },
            {
                "region": "Asia Pacific",
                "revenue": 18000000000,
                "currency": "JPY",
                "growth_rate": 0.25,
                "market_share": 0.15,
                "employees": 30000,
                "major_cities": ["Tokyo", "Singapore", "Sydney", "Mumbai"]
            },
            {
                "region": "China",
                "revenue": 12000000000,
                "currency": "CNY",
                "growth_rate": 0.35,
                "market_share": 0.18,
                "employees": 15000,
                "major_cities": ["Shanghai", "Beijing", "Shenzhen", "Guangzhou"]
            }
        ],
        "quarterly_performance": [
            {
                "quarter": "Q1 2024",
                "revenue_usd": 18500000000,
                "expenses_usd": 14200000000,
                "net_income_usd": 4300000000,
                "eps": 2.85,
                "revenue_growth": 0.15
            },
            {
                "quarter": "Q2 2024",
                "revenue_usd": 19200000000,
                "expenses_usd": 14800000000,
                "net_income_usd": 4400000000,
                "eps": 2.92,
                "revenue_growth": 0.18
            },
            {
                "quarter": "Q3 2024",
                "revenue_usd": 18800000000,
                "expenses_usd": 14500000000,
                "net_income_usd": 4300000000,
                "eps": 2.88,
                "revenue_growth": 0.12
            },
            {
                "quarter": "Q4 2024",
                "revenue_usd": 18500000000,
                "expenses_usd": 14500000000,
                "net_income_usd": 4000000000,
                "eps": 2.65,
                "revenue_growth": 0.08
            }
        ],
        "business_segments": [
            {
                "segment": "Cloud Infrastructure",
                "revenue_usd": 32000000000,
                "operating_margin": 0.28,
                "growth_rate": 0.35,
                "market_position": "Leader",
                "key_products": ["GlobalCloud Pro", "DataCenter Solutions", "AI Compute Platform"]
            },
            {
                "segment": "Enterprise Software",
                "revenue_usd": 25000000000,
                "operating_margin": 0.42,
                "growth_rate": 0.15,
                "market_position": "Strong Competitor",
                "key_products": ["ERP Suite", "CRM Platform", "Analytics Dashboard"]
            },
            {
                "segment": "Consumer Electronics",
                "revenue_usd": 12000000000,
                "operating_margin": 0.18,
                "growth_rate": 0.05,
                "market_position": "Challenger",
                "key_products": ["SmartPhone X", "Tablet Pro", "Smart Home Hub"]
            },
            {
                "segment": "AI & Machine Learning",
                "revenue_usd": 6000000000,
                "operating_margin": 0.15,
                "growth_rate": 0.85,
                "market_position": "Emerging Leader",
                "key_products": ["AI Assistant API", "ML Training Platform", "Computer Vision SDK"]
            }
        ],
        "balance_sheet": {
            "assets": {
                "current_assets": {
                    "cash_and_equivalents_usd": 45000000000,
                    "accounts_receivable_usd": 8500000000,
                    "inventory_usd": 3200000000,
                    "prepaid_expenses_usd": 1800000000
                },
                "non_current_assets": {
                    "property_plant_equipment_usd": 28000000000,
                    "intangible_assets_usd": 35000000000,
                    "goodwill_usd": 22000000000,
                    "investments_usd": 15000000000
                }
            },
            "liabilities": {
                "current_liabilities": {
                    "accounts_payable_usd": 5500000000,
                    "accrued_expenses_usd": 4200000000,
                    "short_term_debt_usd": 2800000000,
                    "deferred_revenue_usd": 6500000000
                },
                "non_current_liabilities": {
                    "long_term_debt_usd": 18000000000,
                    "deferred_tax_liabilities_usd": 3500000000,
                    "pension_obligations_usd": 2200000000
                }
            },
            "equity": {
                "common_stock_usd": 500000000,
                "retained_earnings_usd": 85000000000,
                "accumulated_other_comprehensive_income_usd": 2500000000
            }
        },
        "cash_flow_statement": [
            {
                "period": "2024",
                "operating_activities": {
                    "net_income_usd": 17000000000,
                    "depreciation_amortization_usd": 5500000000,
                    "working_capital_changes_usd": -1200000000,
                    "other_operating_usd": 800000000,
                    "total_operating_cash_flow_usd": 22100000000
                },
                "investing_activities": {
                    "capital_expenditures_usd": -8500000000,
                    "acquisitions_usd": -3200000000,
                    "asset_sales_usd": 500000000,
                    "investment_purchases_usd": -2800000000,
                    "total_investing_cash_flow_usd": -14000000000
                },
                "financing_activities": {
                    "debt_issuance_usd": 2000000000,
                    "debt_repayment_usd": -1500000000,
                    "dividend_payments_usd": -4500000000,
                    "share_repurchases_usd": -6000000000,
                    "total_financing_cash_flow_usd": -10000000000
                }
            }
        ],
        "key_metrics_and_ratios": {
            "profitability": {
                "gross_margin": 0.68,
                "operating_margin": 0.25,
                "net_margin": 0.23,
                "return_on_assets": 0.12,
                "return_on_equity": 0.18
            },
            "liquidity": {
                "current_ratio": 3.2,
                "quick_ratio": 2.8,
                "cash_ratio": 2.1
            },
            "efficiency": {
                "asset_turnover": 0.52,
                "inventory_turnover": 18.5,
                "receivables_turnover": 8.8
            },
            "leverage": {
                "debt_to_equity": 0.35,
                "debt_to_assets": 0.18,
                "interest_coverage": 15.2
            }
        },
        "market_data": {
            "stock_performance": {
                "current_price_usd": 285.50,
                "52_week_high_usd": 320.75,
                "52_week_low_usd": 198.30,
                "ytd_return": 0.18,
                "beta": 1.25
            },
            "valuation_metrics": {
                "pe_ratio": 24.5,
                "peg_ratio": 1.2,
                "price_to_book": 4.8,
                "price_to_sales": 3.3,
                "enterprise_value_usd": 265000000000
            }
        },
        "future_outlook": {
            "fy2025_guidance": {
                "revenue_growth_estimate": 0.15,
                "margin_expansion_target": 0.02,
                "capex_as_percent_revenue": 0.12
            },
            "strategic_initiatives": [
                "AI/ML platform expansion into enterprise market",
                "Acquisition of mid-market cloud companies",
                "Geographic expansion into Latin America and Africa",
                "Sustainability initiative: carbon neutral by 2030"
            ],
            "risk_factors": [
                "Regulatory challenges in data privacy",
                "Increased competition in cloud infrastructure",
                "Currency fluctuation impacts",
                "Talent acquisition and retention in AI/ML"
            ]
        },
        "esg_metrics": {
            "environmental": {
                "carbon_footprint_tons_co2": 2500000,
                "renewable_energy_percentage": 0.65,
                "waste_reduction_target": 0.30
            },
            "social": {
                "diversity_leadership_percentage": 0.45,
                "employee_satisfaction_score": 4.2,
                "community_investment_usd": 150000000
            },
            "governance": {
                "board_independence_percentage": 0.80,
                "female_board_members": 4,
                "audit_committee_meetings": 12
            }
        }
    }

def generate_cryptocurrency_fund_data():
    """Generate complex cryptocurrency fund data"""
    return {
        "fund_info": {
            "name": "Digital Assets Growth Fund",
            "fund_type": "Hedge Fund",
            "inception_date": "2021-03-15",
            "aum_usd": 2500000000,
            "management_fee": 0.02,
            "performance_fee": 0.20,
            "fund_manager": "Blockchain Capital Management",
            "domicile": "Cayman Islands",
            "base_currency": "USD"
        },
        "portfolio_holdings": [
            {
                "asset": "Bitcoin",
                "symbol": "BTC",
                "allocation_percentage": 0.35,
                "current_price_usd": 45250.00,
                "quantity": 19245.5,
                "market_value_usd": 870851237.5,
                "cost_basis_usd": 42800.00,
                "unrealized_pnl_usd": 47147537.5
            },
            {
                "asset": "Ethereum",
                "symbol": "ETH",
                "allocation_percentage": 0.25,
                "current_price_usd": 2850.00,
                "quantity": 219298.25,
                "market_value_usd": 625000012.5,
                "cost_basis_usd": 2200.00,
                "unrealized_pnl_usd": 142643862.5
            },
            {
                "asset": "Binance Coin",
                "symbol": "BNB",
                "allocation_percentage": 0.08,
                "current_price_usd": 285.50,
                "quantity": 701576.13,
                "market_value_usd": 200350000,
                "cost_basis_usd": 320.00,
                "unrealized_pnl_usd": -24204337.5
            },
            {
                "asset": "Cardano",
                "symbol": "ADA",
                "allocation_percentage": 0.06,
                "current_price_usd": 0.85,
                "quantity": 176470588.24,
                "market_value_usd": 150000000,
                "cost_basis_usd": 1.20,
                "unrealized_pnl_usd": -61764705.88
            },
            {
                "asset": "Solana",
                "symbol": "SOL",
                "allocation_percentage": 0.05,
                "current_price_usd": 125.00,
                "quantity": 1000000,
                "market_value_usd": 125000000,
                "cost_basis_usd": 95.00,
                "unrealized_pnl_usd": 30000000
            },
            {
                "asset": "Chainlink",
                "symbol": "LINK",
                "allocation_percentage": 0.04,
                "current_price_usd": 18.50,
                "quantity": 5405405.41,
                "market_value_usd": 100000000,
                "cost_basis_usd": 22.00,
                "unrealized_pnl_usd": -18918918.92
            },
            {
                "asset": "Polygon",
                "symbol": "MATIC",
                "allocation_percentage": 0.03,
                "current_price_usd": 1.25,
                "quantity": 60000000,
                "market_value_usd": 75000000,
                "cost_basis_usd": 1.80,
                "unrealized_pnl_usd": -33000000
            },
            {
                "asset": "Avalanche",
                "symbol": "AVAX",
                "allocation_percentage": 0.025,
                "current_price_usd": 42.50,
                "quantity": 1470588.24,
                "market_value_usd": 62500000,
                "cost_basis_usd": 85.00,
                "unrealized_pnl_usd": -62500000
            },
            {
                "asset": "Cash and Cash Equivalents",
                "symbol": "USD",
                "allocation_percentage": 0.135,
                "current_price_usd": 1.00,
                "quantity": 337500000,
                "market_value_usd": 337500000,
                "cost_basis_usd": 1.00,
                "unrealized_pnl_usd": 0
            }
        ],
        "performance_metrics": {
            "ytd_return": 0.125,
            "1_year_return": 0.85,
            "3_year_return": 2.45,
            "since_inception_return": 3.25,
            "volatility": 0.68,
            "sharpe_ratio": 1.15,
            "max_drawdown": -0.45,
            "calmar_ratio": 0.75
        },
        "monthly_returns": [
            {"month": "2024-01", "return": 0.08, "benchmark_return": 0.05},
            {"month": "2024-02", "return": -0.12, "benchmark_return": -0.08},
            {"month": "2024-03", "return": 0.15, "benchmark_return": 0.12},
            {"month": "2024-04", "return": 0.22, "benchmark_return": 0.18},
            {"month": "2024-05", "return": -0.08, "benchmark_return": -0.05},
            {"month": "2024-06", "return": 0.18, "benchmark_return": 0.15},
            {"month": "2024-07", "return": 0.12, "benchmark_return": 0.09},
            {"month": "2024-08", "return": -0.15, "benchmark_return": -0.12},
            {"month": "2024-09", "return": 0.25, "benchmark_return": 0.20},
            {"month": "2024-10", "return": 0.18, "benchmark_return": 0.14},
            {"month": "2024-11", "return": 0.32, "benchmark_return": 0.28},
            {"month": "2024-12", "return": -0.05, "benchmark_return": -0.03}
        ],
        "risk_analytics": {
            "value_at_risk_95": -125000000,
            "expected_shortfall": -185000000,
            "beta_to_bitcoin": 1.25,
            "correlation_to_sp500": 0.35,
            "portfolio_concentration": 0.60
        },
        "transaction_history": [
            {
                "date": "2024-12-01",
                "asset": "BTC",
                "transaction_type": "BUY",
                "quantity": 500,
                "price_usd": 44200.00,
                "total_value_usd": 22100000,
                "fees_usd": 11050
            },
            {
                "date": "2024-11-28",
                "asset": "ETH",
                "transaction_type": "SELL",
                "quantity": 1000,
                "price_usd": 2820.00,
                "total_value_usd": 2820000,
                "fees_usd": 1410
            },
            {
                "date": "2024-11-25",
                "asset": "SOL",
                "transaction_type": "BUY",
                "quantity": 10000,
                "price_usd": 122.50,
                "total_value_usd": 1225000,
                "fees_usd": 612.5
            }
        ],
        "fund_expenses": {
            "management_fees_ytd_usd": 50000000,
            "performance_fees_ytd_usd": 62500000,
            "operational_expenses_usd": 8500000,
            "custody_fees_usd": 2500000,
            "audit_legal_fees_usd": 1500000,
            "total_expenses_usd": 125000000
        },
        "investor_base": {
            "total_investors": 485,
            "institutional_percentage": 0.75,
            "retail_percentage": 0.25,
            "average_investment_usd": 5154639,
            "largest_investor_percentage": 0.12,
            "geographic_distribution": {
                "north_america": 0.45,
                "europe": 0.30,
                "asia": 0.20,
                "other": 0.05
            }
        }
    }

def generate_real_estate_portfolio_data():
    """Generate complex real estate portfolio data"""
    return {
        "portfolio_overview": {
            "portfolio_name": "Premium Global Real Estate Portfolio",
            "total_properties": 247,
            "total_value_usd": 8500000000,
            "total_rental_income_annual_usd": 680000000,
            "average_cap_rate": 0.08,
            "occupancy_rate": 0.92,
            "portfolio_manager": "Global Property Partners LLC",
            "inception_date": "2018-06-01"
        },
        "property_segments": [
            {
                "segment": "Office Buildings",
                "property_count": 45,
                "total_value_usd": 3200000000,
                "annual_rental_income_usd": 256000000,
                "average_cap_rate": 0.075,
                "occupancy_rate": 0.88,
                "average_lease_term_years": 8.5,
                "major_tenants": ["TechCorp", "Financial Services Inc", "Law Firm Partners"]
            },
            {
                "segment": "Retail Centers",
                "property_count": 78,
                "total_value_usd": 2100000000,
                "annual_rental_income_usd": 189000000,
                "average_cap_rate": 0.09,
                "occupancy_rate": 0.94,
                "average_lease_term_years": 5.2,
                "major_tenants": ["SuperMart", "Fashion Brands Ltd", "Restaurant Chain Co"]
            },
            {
                "segment": "Industrial Warehouses",
                "property_count": 62,
                "total_value_usd": 1850000000,
                "annual_rental_income_usd": 148000000,
                "average_cap_rate": 0.085,
                "occupancy_rate": 0.96,
                "average_lease_term_years": 12.3,
                "major_tenants": ["Logistics Corp", "E-commerce Giant", "Manufacturing Ltd"]
            },
            {
                "segment": "Multifamily Residential",
                "property_count": 35,
                "total_value_usd": 1050000000,
                "annual_rental_income_usd": 63000000,
                "average_cap_rate": 0.06,
                "occupancy_rate": 0.95,
                "average_lease_term_years": 1.2,
                "major_tenants": ["Individual Residents"]
            },
            {
                "segment": "Mixed Use",
                "property_count": 27,
                "total_value_usd": 300000000,
                "annual_rental_income_usd": 24000000,
                "average_cap_rate": 0.08,
                "occupancy_rate": 0.90,
                "average_lease_term_years": 6.8,
                "major_tenants": ["Various Commercial and Residential"]
            }
        ],
        "geographic_distribution": [
            {
                "region": "United States",
                "property_count": 98,
                "total_value_usd": 3400000000,
                "currency": "USD",
                "key_markets": ["New York", "Los Angeles", "Chicago", "Dallas"],
                "market_growth_rate": 0.06
            },
            {
                "region": "United Kingdom",
                "property_count": 45,
                "total_value_usd": 1530000000,
                "currency": "GBP",
                "key_markets": ["London", "Manchester", "Birmingham"],
                "market_growth_rate": 0.04
            },
            {
                "region": "Germany",
                "property_count": 38,
                "total_value_usd": 1275000000,
                "currency": "EUR",
                "key_markets": ["Berlin", "Munich", "Frankfurt", "Hamburg"],
                "market_growth_rate": 0.05
            },
            {
                "region": "Canada",
                "property_count": 32,
                "total_value_usd": 1020000000,
                "currency": "CAD",
                "key_markets": ["Toronto", "Vancouver", "Montreal"],
                "market_growth_rate": 0.07
            },
            {
                "region": "Australia",
                "property_count": 22,
                "total_value_usd": 765000000,
                "currency": "AUD",
                "key_markets": ["Sydney", "Melbourne", "Brisbane"],
                "market_growth_rate": 0.08
            },
            {
                "region": "Japan",
                "property_count": 12,
                "total_value_usd": 510000000,
                "currency": "JPY",
                "key_markets": ["Tokyo", "Osaka"],
                "market_growth_rate": 0.03
            }
        ],
        "financial_performance": {
            "quarterly_metrics": [
                {
                    "quarter": "Q1 2024",
                    "rental_income_usd": 165000000,
                    "operating_expenses_usd": 49500000,
                    "net_operating_income_usd": 115500000,
                    "capital_expenditures_usd": 25000000,
                    "acquisitions_usd": 150000000,
                    "dispositions_usd": 75000000
                },
                {
                    "quarter": "Q2 2024",
                    "rental_income_usd": 170000000,
                    "operating_expenses_usd": 51000000,
                    "net_operating_income_usd": 119000000,
                    "capital_expenditures_usd": 30000000,
                    "acquisitions_usd": 200000000,
                    "dispositions_usd": 50000000
                },
                {
                    "quarter": "Q3 2024",
                    "rental_income_usd": 172500000,
                    "operating_expenses_usd": 51750000,
                    "net_operating_income_usd": 120750000,
                    "capital_expenditures_usd": 28000000,
                    "acquisitions_usd": 100000000,
                    "dispositions_usd": 125000000
                },
                {
                    "quarter": "Q4 2024",
                    "rental_income_usd": 172500000,
                    "operating_expenses_usd": 52250000,
                    "net_operating_income_usd": 120250000,
                    "capital_expenditures_usd": 35000000,
                    "acquisitions_usd": 75000000,
                    "dispositions_usd": 100000000
                }
            ]
        },
        "debt_and_financing": {
            "total_debt_usd": 3825000000,
            "weighted_average_interest_rate": 0.045,
            "loan_to_value_ratio": 0.45,
            "debt_service_coverage_ratio": 2.8,
            "financing_breakdown": [
                {
                    "lender": "Bank of America",
                    "loan_amount_usd": 850000000,
                    "interest_rate": 0.042,
                    "maturity_date": "2029-03-15",
                    "loan_type": "Fixed Rate"
                },
                {
                    "lender": "JPMorgan Chase",
                    "loan_amount_usd": 765000000,
                    "interest_rate": 0.038,
                    "maturity_date": "2028-11-30",
                    "loan_type": "Fixed Rate"
                },
                {
                    "lender": "Wells Fargo",
                    "loan_amount_usd": 680000000,
                    "interest_rate": 0.051,
                    "maturity_date": "2030-07-01",
                    "loan_type": "Variable Rate"
                },
                {
                    "lender": "Goldman Sachs",
                    "loan_amount_usd": 595000000,
                    "interest_rate": 0.046,
                    "maturity_date": "2031-12-31",
                    "loan_type": "Fixed Rate"
                },
                {
                    "lender": "CMBS Pool",
                    "loan_amount_usd": 935000000,
                    "interest_rate": 0.048,
                    "maturity_date": "2027-09-15",
                    "loan_type": "Commercial Mortgage Backed Securities"
                }
            ]
        },
        "market_analysis": {
            "comparable_transactions": [
                {
                    "property_type": "Office",
                    "location": "Manhattan, NY",
                    "sale_price_per_sqft_usd": 1250,
                    "cap_rate": 0.065,
                    "date": "2024-10-15"
                },
                {
                    "property_type": "Industrial",
                    "location": "Los Angeles, CA",
                    "sale_price_per_sqft_usd": 425,
                    "cap_rate": 0.075,
                    "date": "2024-09-22"
                },
                {
                    "property_type": "Retail",
                    "location": "Miami, FL",
                    "sale_price_per_sqft_usd": 650,
                    "cap_rate": 0.085,
                    "date": "2024-11-08"
                }
            ]
        }
    }

def run_advanced_ai_test(test_name: str, test_data: Dict[str, Any], description: str, expected_ai_features: List[str]) -> Dict[str, Any]:
    """Run an advanced AI test with automatic download"""
    print_test_header(test_name)
    print_info(f"Description: {description}")
    print_info(f"Expected AI features: {', '.join(expected_ai_features)}")
    print_info(f"Data complexity: {len(json.dumps(test_data))} characters")
    
    # Create request payload
    request_payload = {
        "json_data": json.dumps(test_data, indent=2),
        "file_name": f"advanced_ai_{test_name.lower().replace(' ', '_')}",
        "description": description,
        "processing_mode": "ai_only",  # Force AI processing
        "user_id": TEST_USER_ID,
        "session_id": f"session_{test_name.lower().replace(' ', '_')}",
        "model": "gemini-2.5-pro-preview-06-05"
    }
    
    print_info("🚀 Starting AI processing (this may take 2-3 minutes)...")
    start_time = time.time()
    
    try:
        response = requests.post(f"{BASE_URL}/process", json=request_payload, timeout=300)  # 5 minute timeout
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = {
            "test_name": test_name,
            "processing_time": processing_time,
            "success": False,
            "ai_analysis_length": 0,
            "download_success": False,
            "file_path": None,
            "insights": []
        }
        
        if response.status_code == 200:
            response_data = response.json()
            
            if response_data.get("success"):
                print_success(f"AI processing completed in {processing_time:.1f} seconds")
                
                # Check for AI analysis
                ai_analysis = response_data.get("ai_analysis", "")
                if ai_analysis:
                    result["ai_analysis_length"] = len(ai_analysis)
                    print_success(f"AI analysis generated: {len(ai_analysis)} characters")
                    
                    # Extract insights from AI analysis
                    insights = []
                    if "currency" in ai_analysis.lower():
                        insights.append("Currency conversion detected")
                    if "forecast" in ai_analysis.lower() or "projection" in ai_analysis.lower():
                        insights.append("Financial forecasting included")
                    if "excel" in ai_analysis.lower() and "sheet" in ai_analysis.lower():
                        insights.append("Multi-sheet Excel generation")
                    if "chart" in ai_analysis.lower() or "graph" in ai_analysis.lower():
                        insights.append("Chart/visualization creation")
                    if "summary" in ai_analysis.lower():
                        insights.append("Executive summary generated")
                    
                    result["insights"] = insights
                    
                    for insight in insights:
                        print_ai_insight(insight)
                    
                    # Show preview of AI analysis
                    preview = ai_analysis[:500]
                    print_info(f"AI Analysis Preview:\n{preview}{'...' if len(ai_analysis) > 500 else ''}")
                
                # Download the file
                file_id = response_data.get("file_id")
                if file_id:
                    filename = f"{test_name.replace(' ', '_')}_{file_id}.xlsx"
                    if download_file(file_id, filename):
                        result["download_success"] = True
                        result["file_path"] = os.path.join(DOWNLOAD_DIR, filename)
                        print_success(f"File successfully downloaded: {filename}")
                    else:
                        print_error("File download failed")
                
                result["success"] = True
                
            else:
                print_error(f"AI processing failed: {response_data.get('error', 'Unknown error')}")
        else:
            print_error(f"HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        print_error("AI processing timed out (5 minutes)")
        result = {"test_name": test_name, "success": False, "error": "Timeout"}
    except Exception as e:
        print_error(f"Test failed: {e}")
        result = {"test_name": test_name, "success": False, "error": str(e)}
    
    return result

def run_advanced_test_suite():
    """Run the complete advanced AI test suite"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("🚀 INTELLIEXTRACT ADVANCED AI TEST SUITE")
    print("=" * 80)
    print(f"Testing complex financial scenarios with automatic downloads{Colors.ENDC}")
    
    ensure_download_dir()
    
    # Check server connectivity
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print_success("✅ Backend server is running")
        else:
            print_error("❌ Backend server returned unexpected status")
            return
    except:
        print_error("❌ Cannot connect to backend server")
        print_info("Please start the server: uvicorn app.main:app --reload")
        return
    
    print_info(f"Test User ID: {TEST_USER_ID}")
    print_info(f"Download Directory: {os.path.abspath(DOWNLOAD_DIR)}")
    
    all_results = []
    
    # Test 1: Multinational Corporation
    print("\n" + "="*80)
    multinational_data = generate_multinational_corporation_data()
    result1 = run_advanced_ai_test(
        "Multinational Corporation Analysis",
        multinational_data,
        "Complex multinational corporation with multi-currency operations, detailed financial statements, and regional performance data",
        ["Multi-currency conversion", "Regional analysis", "Financial forecasting", "Multi-sheet Excel", "Executive dashboard"]
    )
    all_results.append(result1)
    
    # Test 2: Cryptocurrency Fund
    print("\n" + "="*80)
    crypto_data = generate_cryptocurrency_fund_data()
    result2 = run_advanced_ai_test(
        "Cryptocurrency Fund Portfolio",
        crypto_data,
        "Digital assets hedge fund with real-time pricing, risk analytics, and performance metrics",
        ["Real-time crypto pricing", "Portfolio risk analysis", "Performance attribution", "Volatility modeling", "Sharpe ratio calculations"]
    )
    all_results.append(result2)
    
    # Test 3: Real Estate Portfolio
    print("\n" + "="*80)
    realestate_data = generate_real_estate_portfolio_data()
    result3 = run_advanced_ai_test(
        "Global Real Estate Portfolio",
        realestate_data,
        "Large-scale real estate portfolio with multiple property types, geographic diversification, and debt analysis",
        ["Cap rate analysis", "Geographic performance", "Debt service calculations", "Property valuation", "Market comparables"]
    )
    all_results.append(result3)
    
    # Final Summary
    print_test_header("TEST RESULTS SUMMARY")
    
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.get("success", False))
    total_downloads = sum(1 for r in all_results if r.get("download_success", False))
    
    print_info(f"Total Tests: {total_tests}")
    print_success(f"Successful AI Processing: {successful_tests}")
    print_success(f"Successful Downloads: {total_downloads}")
    
    if successful_tests > 0:
        avg_time = sum(r.get("processing_time", 0) for r in all_results if r.get("success")) / successful_tests
        total_analysis = sum(r.get("ai_analysis_length", 0) for r in all_results)
        print_info(f"Average Processing Time: {avg_time:.1f} seconds")
        print_info(f"Total AI Analysis Generated: {total_analysis:,} characters")
    
    # Show download summary
    if total_downloads > 0:
        print_success(f"\n📁 Downloaded Files Location: {os.path.abspath(DOWNLOAD_DIR)}")
        for result in all_results:
            if result.get("download_success"):
                print_success(f"   📄 {result['test_name']}: {result['file_path']}")
    
    # Show AI insights summary
    all_insights = []
    for result in all_results:
        all_insights.extend(result.get("insights", []))
    
    if all_insights:
        print_ai_insight("AI Features Successfully Demonstrated:")
        unique_insights = list(set(all_insights))
        for insight in unique_insights:
            print_ai_insight(f"   ✅ {insight}")
    
    # Show failed tests
    failed_tests = [r for r in all_results if not r.get("success", False)]
    if failed_tests:
        print_warning("\n⚠️ Failed Tests:")
        for failed in failed_tests:
            print_error(f"   ❌ {failed['test_name']}: {failed.get('error', 'Unknown error')}")
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}🎉 Advanced AI Test Suite Complete!{Colors.ENDC}")
    
    if total_downloads > 0:
        print_info(f"\n💡 Open the downloaded Excel files to see the AI-generated:")
        print_info("   • Multi-sheet financial reports")
        print_info("   • Currency conversions with current exchange rates")
        print_info("   • Financial analysis and forecasts")
        print_info("   • Executive summaries and insights")
        print_info("   • Charts and visualizations")

if __name__ == "__main__":
    run_advanced_test_suite()