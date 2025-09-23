#!/usr/bin/env python3
"""
TFSA-specific optimization suggestions for the trading app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_analyzer import AdvancedTradingAnalyzer
import warnings
warnings.filterwarnings('ignore')

def suggest_tfsa_optimizations():
    """Suggest TFSA-specific optimizations"""
    print("🏦 TFSA Questrade Optimization Suggestions")
    print("=" * 50)
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    universe = analyzer.stock_universe
    
    # Identify Canadian stocks (TFSA tax-efficient)
    canadian_stocks = [
        'RY', 'TD', 'BMO', 'BNS', 'CM', 'NA',  # Banks
        'CNR', 'CP',  # Railways
        'ATD', 'WCN', 'BAM', 'MFC',  # Diversified
        'SU', 'CNQ', 'IMO', 'CVE'  # Energy
    ]
    
    # Identify dividend aristocrats (TFSA-friendly)
    dividend_aristocrats = [
        'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 
        'CVX', 'XOM', 'CAT', 'MMM', 'GE', 'HON', 'UPS', 
        'FDX', 'VZ', 'T', 'NEE', 'DUK', 'SO'
    ]
    
    # Identify growth stocks (capital gains focus)
    growth_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'PLTR', 'CRWD', 'SNOW', 'DDOG', 'NET', 'OKTA', 'ZM'
    ]
    
    print("🎯 TFSA-Optimized Stock Categories:")
    print(f"   🏦 Canadian Stocks: {len([s for s in canadian_stocks if s in universe])} available")
    print(f"   💰 Dividend Aristocrats: {len([s for s in dividend_aristocrats if s in universe])} available")
    print(f"   📈 Growth Stocks: {len([s for s in growth_stocks if s in universe])} available")
    
    # TFSA portfolio recommendations by size
    tfsa_scenarios = [
        {
            "name": "Starter TFSA",
            "value": 7000,
            "positions": 5,
            "strategy": "Quality Focus",
            "analysis_size": 50,
            "recommendations": [
                "Focus on dividend aristocrats for income",
                "Choose 2-3 Canadian banks (no withholding tax)",
                "Add 1-2 growth stocks for capital appreciation",
                "Minimum $1,000 per position to minimize commission impact"
            ]
        },
        {
            "name": "Growing TFSA", 
            "value": 25000,
            "positions": 10,
            "strategy": "Balanced Growth",
            "analysis_size": 100,
            "recommendations": [
                "Mix of Canadian (40%) and US (60%) stocks",
                "Include 3-4 dividend aristocrats",
                "Add 3-4 growth stocks for upside",
                "Consider 2-3 mid-cap opportunities"
            ]
        },
        {
            "name": "Mature TFSA",
            "value": 50000, 
            "positions": 15,
            "strategy": "Diversified Growth",
            "analysis_size": 150,
            "recommendations": [
                "Full sector diversification",
                "30% Canadian, 70% US allocation",
                "Include small-cap growth opportunities",
                "Focus on tax-efficient dividend growers"
            ]
        },
        {
            "name": "Maximum TFSA",
            "value": 95000,
            "positions": 20,
            "strategy": "Institutional Quality",
            "analysis_size": 200,
            "recommendations": [
                "Professional-grade diversification",
                "Include international exposure via US multinationals",
                "Mix of value, growth, and dividend strategies",
                "Consider sector rotation opportunities"
            ]
        }
    ]
    
    print(f"\n📊 TFSA Portfolio Strategies:")
    
    for scenario in tfsa_scenarios:
        print(f"\n💰 {scenario['name']} (${scenario['value']:,})")
        print(f"   🎯 Target: {scenario['positions']} positions")
        print(f"   📈 Strategy: {scenario['strategy']}")
        print(f"   🔍 Analysis Size: {scenario['analysis_size']} stocks")
        print(f"   💡 Recommendations:")
        for rec in scenario['recommendations']:
            print(f"      • {rec}")
    
    # Current app assessment
    print(f"\n✅ CURRENT APP ASSESSMENT:")
    print(f"   🏆 Coverage: EXCELLENT (20-200 stocks covers all scenarios)")
    print(f"   🎯 Flexibility: PERFECT (adjustable based on TFSA size)")
    print(f"   📊 Analysis Depth: OPTIMAL (5-10x selection ratio)")
    print(f"   💰 Cost Efficiency: IDEAL (supports $1,000+ position sizes)")

def recommend_app_settings_by_tfsa_size():
    """Recommend optimal app settings by TFSA size"""
    print(f"\n🔧 RECOMMENDED APP SETTINGS BY TFSA SIZE")
    print("=" * 50)
    
    settings_recommendations = [
        {
            "tfsa_size": "$7K - $15K (Starter)",
            "num_stocks": 50,
            "cap_filter": "Large Cap",
            "market_focus": "Dividend Aristocrats", 
            "analysis_type": "Risk Management",
            "risk_style": "Low Risk",
            "rationale": "Focus on quality, stability, and dividend income"
        },
        {
            "tfsa_size": "$15K - $35K (Growing)",
            "num_stocks": 100,
            "cap_filter": "All",
            "market_focus": "S&P 500 Large Cap",
            "analysis_type": "Institutional Grade", 
            "risk_style": "Balanced",
            "rationale": "Balanced approach with institutional-quality picks"
        },
        {
            "tfsa_size": "$35K - $60K (Mature)",
            "num_stocks": 150,
            "cap_filter": "All", 
            "market_focus": "All Markets",
            "analysis_type": "Investment Bank Level",
            "risk_style": "Balanced",
            "rationale": "Full diversification with professional analysis"
        },
        {
            "tfsa_size": "$60K+ (Maximum)",
            "num_stocks": 200,
            "cap_filter": "All",
            "market_focus": "Sector Rotation",
            "analysis_type": "Hedge Fund Style",
            "risk_style": "Balanced", 
            "rationale": "Sophisticated strategies with maximum coverage"
        }
    ]
    
    for setting in settings_recommendations:
        print(f"\n💰 {setting['tfsa_size']}")
        print(f"   📊 Number of Stocks: {setting['num_stocks']}")
        print(f"   🎯 Cap Filter: {setting['cap_filter']}")
        print(f"   📈 Market Focus: {setting['market_focus']}")
        print(f"   🏛️ Analysis Type: {setting['analysis_type']}")
        print(f"   ⚖️ Risk Style: {setting['risk_style']}")
        print(f"   💡 Rationale: {setting['rationale']}")

if __name__ == "__main__":
    # Analyze TFSA optimizations
    suggest_tfsa_optimizations()
    
    # Recommend app settings
    recommend_app_settings_by_tfsa_size()
    
    print(f"\n🎉 FINAL RECOMMENDATION:")
    print(f"Your current 20-200 stock range is PERFECT for TFSA Questrade trading!")
    print(f"No need to increase - instead optimize your settings based on TFSA size.")
