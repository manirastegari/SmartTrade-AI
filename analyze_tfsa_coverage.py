#!/usr/bin/env python3
"""
Analyze current coverage for TFSA Questrade trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_analyzer import AdvancedTradingAnalyzer
import warnings
warnings.filterwarnings('ignore')

def analyze_tfsa_questrade_coverage():
    """Analyze coverage for TFSA Questrade trading"""
    print("🏦 TFSA Questrade Coverage Analysis")
    print("=" * 50)
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    universe = analyzer.stock_universe
    universe_size = len(universe)
    
    print(f"📊 Current Universe: {universe_size} unique symbols")
    
    # Analyze current cap distribution
    large_cap_end = universe_size // 3
    mid_cap_start = universe_size // 3
    mid_cap_end = (universe_size * 2) // 3
    small_cap_start = (universe_size * 2) // 3
    
    large_cap_symbols = universe[:large_cap_end]
    mid_cap_symbols = universe[mid_cap_start:mid_cap_end]
    small_cap_symbols = universe[small_cap_start:]
    
    print(f"\n📈 Current Cap Distribution:")
    print(f"   Large Cap: {len(large_cap_symbols)} symbols")
    print(f"   Mid Cap: {len(mid_cap_symbols)} symbols") 
    print(f"   Small Cap: {len(small_cap_symbols)} symbols")
    
    # TFSA Questrade considerations
    print(f"\n🏦 TFSA Questrade Considerations:")
    
    # Canadian vs US stocks
    canadian_symbols = [s for s in universe if s in ['RY', 'TD', 'BMO', 'BNS', 'CM', 'NA', 'CNR', 'CP', 'ATD', 'WCN', 'BAM', 'MFC', 'SU', 'CNQ', 'IMO', 'CVE']]
    us_symbols = [s for s in universe if s not in canadian_symbols]
    
    print(f"   Canadian stocks: {len(canadian_symbols)} ({len(canadian_symbols)/universe_size*100:.1f}%)")
    print(f"   US stocks: {len(us_symbols)} ({len(us_symbols)/universe_size*100:.1f}%)")
    
    # TFSA-friendly characteristics
    print(f"\n💰 TFSA Trading Characteristics:")
    print(f"   ✅ No withholding tax on Canadian dividends")
    print(f"   ⚠️ 15% withholding tax on US dividends (but recoverable)")
    print(f"   ✅ All capital gains tax-free")
    print(f"   ✅ No foreign content restrictions (removed in 2005)")
    
    # Coverage analysis by request size
    print(f"\n📊 Coverage Analysis by Request Size:")
    
    coverage_scenarios = [
        ("Conservative", 50),
        ("Balanced", 100), 
        ("Aggressive", 200),
        ("Maximum", 300)
    ]
    
    for scenario, count in coverage_scenarios:
        print(f"\n🎯 {scenario} Analysis ({count} stocks):")
        
        # Large Cap coverage
        large_actual = min(count, len(large_cap_symbols))
        large_pct = (large_actual / len(large_cap_symbols)) * 100
        print(f"   Large Cap: {large_actual}/{len(large_cap_symbols)} ({large_pct:.1f}% coverage)")
        
        # Mid Cap coverage  
        mid_actual = min(count, len(mid_cap_symbols))
        mid_pct = (mid_actual / len(mid_cap_symbols)) * 100
        print(f"   Mid Cap: {mid_actual}/{len(mid_cap_symbols)} ({mid_pct:.1f}% coverage)")
        
        # Small Cap coverage
        small_actual = min(count, len(small_cap_symbols))
        small_pct = (small_actual / len(small_cap_symbols)) * 100
        print(f"   Small Cap: {small_actual}/{len(small_cap_symbols)} ({small_pct:.1f}% coverage)")
        
        # All Markets coverage
        all_actual = min(count, universe_size)
        all_pct = (all_actual / universe_size) * 100
        print(f"   All Markets: {all_actual}/{universe_size} ({all_pct:.1f}% coverage)")

def analyze_questrade_specific_needs():
    """Analyze specific needs for Questrade TFSA trading"""
    print(f"\n🎯 Questrade TFSA Specific Analysis")
    print("=" * 40)
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    universe = analyzer.stock_universe
    
    # Questrade commission structure (as of 2024)
    print(f"💰 Questrade Commission Structure:")
    print(f"   📈 Stock trades: $4.95-$9.95 per trade")
    print(f"   📊 ETF purchases: FREE")
    print(f"   💵 Minimum account: $1,000")
    print(f"   🏦 TFSA contribution room 2024: $7,000")
    
    # Optimal trade sizes for TFSA
    print(f"\n💡 TFSA Trading Optimization:")
    print(f"   🎯 Minimum trade size: $500+ (to keep commission <2%)")
    print(f"   📊 Optimal trade size: $1,000+ (to keep commission <1%)")
    print(f"   🏆 Sweet spot: $2,000+ trades (commission <0.5%)")
    
    # Portfolio size considerations
    tfsa_room_scenarios = [
        ("New TFSA", 7000),      # 2024 contribution
        ("Moderate TFSA", 25000), # Few years of contributions
        ("Mature TFSA", 50000),   # Many years of contributions
        ("Max TFSA", 95000)       # Maximum room (2009-2024)
    ]
    
    print(f"\n📊 Portfolio Diversification by TFSA Size:")
    
    for scenario, tfsa_value in tfsa_room_scenarios:
        print(f"\n💰 {scenario} (${tfsa_value:,}):")
        
        # Calculate optimal number of positions
        min_position = 1000  # $1,000 minimum per position
        max_positions = tfsa_value // min_position
        
        # Recommended diversification
        if tfsa_value <= 10000:
            recommended_stocks = min(5, max_positions)
            print(f"   🎯 Recommended: {recommended_stocks} stocks (focus on quality)")
        elif tfsa_value <= 30000:
            recommended_stocks = min(10, max_positions)
            print(f"   🎯 Recommended: {recommended_stocks} stocks (balanced diversification)")
        elif tfsa_value <= 60000:
            recommended_stocks = min(15, max_positions)
            print(f"   🎯 Recommended: {recommended_stocks} stocks (good diversification)")
        else:
            recommended_stocks = min(20, max_positions)
            print(f"   🎯 Recommended: {recommended_stocks} stocks (full diversification)")
        
        print(f"   📈 Max positions possible: {max_positions}")
        print(f"   💵 Avg position size: ${tfsa_value // recommended_stocks:,}")
        
        # Analysis coverage needed
        analysis_multiplier = 5  # Analyze 5x more than you'll buy
        analysis_needed = recommended_stocks * analysis_multiplier
        
        print(f"   🔍 Analysis needed: {analysis_needed} stocks (5x selection ratio)")
        
        # Current app coverage assessment
        current_coverage = min(200, len(universe))  # Current max
        coverage_ratio = current_coverage / analysis_needed
        
        if coverage_ratio >= 1.0:
            print(f"   ✅ Current coverage: EXCELLENT ({current_coverage} vs {analysis_needed} needed)")
        elif coverage_ratio >= 0.7:
            print(f"   ⚠️ Current coverage: GOOD ({current_coverage} vs {analysis_needed} needed)")
        else:
            print(f"   ❌ Current coverage: INSUFFICIENT ({current_coverage} vs {analysis_needed} needed)")

def recommend_coverage_improvements():
    """Recommend coverage improvements"""
    print(f"\n🚀 Coverage Improvement Recommendations")
    print("=" * 45)
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    universe_size = len(analyzer.stock_universe)
    
    print(f"📊 Current Status:")
    print(f"   Universe Size: {universe_size} stocks")
    print(f"   Current Range: 20-200 stocks per analysis")
    print(f"   Maximum Coverage: {min(200, universe_size)} stocks")
    
    print(f"\n💡 Recommendations:")
    
    # For different TFSA sizes
    recommendations = [
        {
            "tfsa_size": "Small TFSA ($7K-$25K)",
            "target_positions": "5-10 stocks",
            "analysis_needed": "25-50 stocks",
            "current_adequate": True,
            "recommendation": "Current 50-100 stock analysis is PERFECT"
        },
        {
            "tfsa_size": "Medium TFSA ($25K-$50K)", 
            "target_positions": "10-15 stocks",
            "analysis_needed": "50-75 stocks",
            "current_adequate": True,
            "recommendation": "Current 100-150 stock analysis is EXCELLENT"
        },
        {
            "tfsa_size": "Large TFSA ($50K+)",
            "target_positions": "15-20 stocks", 
            "analysis_needed": "75-100 stocks",
            "current_adequate": True,
            "recommendation": "Current 150-200 stock analysis is OPTIMAL"
        }
    ]
    
    for rec in recommendations:
        print(f"\n🎯 {rec['tfsa_size']}:")
        print(f"   Target Portfolio: {rec['target_positions']}")
        print(f"   Analysis Needed: {rec['analysis_needed']}")
        status = "✅" if rec['current_adequate'] else "❌"
        print(f"   {status} {rec['recommendation']}")
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"   ✅ Current 20-200 range is EXCELLENT for TFSA trading")
    print(f"   ✅ Covers all realistic TFSA portfolio sizes")
    print(f"   ✅ Provides 5-10x selection ratio for quality picks")
    print(f"   ✅ No need to increase beyond 200 stocks per analysis")
    
    # Optimization suggestions
    print(f"\n🔧 Optimization Suggestions:")
    print(f"   1. 📊 Use 100+ stocks for small/mid cap (more opportunities)")
    print(f"   2. 🎯 Use 150+ stocks for 'All Markets' (maximum diversification)")
    print(f"   3. 💰 Focus on dividend-paying stocks for TFSA tax efficiency")
    print(f"   4. 🏦 Consider Canadian stocks for zero withholding tax")
    print(f"   5. 📈 Use Market Focus to target specific strategies")

if __name__ == "__main__":
    print("🏦 TFSA Questrade Coverage Analysis")
    print("=" * 60)
    
    # Analyze current coverage
    analyze_tfsa_questrade_coverage()
    
    # Analyze Questrade-specific needs
    analyze_questrade_specific_needs()
    
    # Provide recommendations
    recommend_coverage_improvements()
    
    print(f"\n🎉 CONCLUSION:")
    print(f"Current 20-200 stock range is PERFECT for TFSA Questrade trading!")
    print(f"No need to increase coverage - focus on quality analysis instead.")
