#!/usr/bin/env python3
"""
Analyze current app behavior to understand consistency issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_analyzer import AdvancedTradingAnalyzer
import warnings
warnings.filterwarnings('ignore')

def get_symbols_by_cap(analyzer, cap_filter: str, count: int):
    """Current cap selection logic"""
    universe = analyzer.stock_universe
    universe_size = len(universe)
    
    if cap_filter == "Large Cap":
        end_idx = min(universe_size // 3, 200)
        return universe[:max(10, min(count, end_idx))]
    elif cap_filter == "Mid Cap":
        start = universe_size // 3
        end = (universe_size * 2) // 3
        available = universe[start:end]
        return available[:max(10, min(count, len(available)))]
    elif cap_filter == "Small Cap":
        start = (universe_size * 2) // 3
        available = universe[start:]
        return available[:max(10, min(count, len(available)))]
    else:
        return universe[:max(10, min(count, universe_size))]

def analyze_current_behavior():
    """Analyze current app behavior"""
    print("🔍 Analyzing Current App Behavior")
    print("=" * 50)
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    universe_size = len(analyzer.stock_universe)
    
    print(f"📊 Total universe size: {universe_size} unique symbols")
    
    # Test different cap filters with different stock counts
    cap_filters = ["Large Cap", "Mid Cap", "Small Cap", "All"]
    stock_counts = [10, 50, 100, 200]
    
    print("\n🧪 Testing Cap Filter Consistency:")
    for cap_filter in cap_filters:
        print(f"\n📈 {cap_filter}:")
        for count in stock_counts:
            symbols = get_symbols_by_cap(analyzer, cap_filter, count)
            print(f"   {count} requested → {len(symbols)} actual symbols")
            if len(symbols) >= 5:
                print(f"   First 5: {symbols[:5]}")
                print(f"   Last 5: {symbols[-5:]}")
    
    # Test multiple runs for consistency
    print(f"\n🔄 Testing Multiple Runs for Consistency:")
    cap_filter = "Small Cap"
    count = 50
    
    runs = []
    for i in range(3):
        symbols = get_symbols_by_cap(analyzer, cap_filter, count)
        runs.append(symbols)
        print(f"   Run {i+1}: {len(symbols)} symbols, first 5: {symbols[:5]}")
    
    # Check if runs are identical
    all_same = all(run == runs[0] for run in runs)
    print(f"   ✅ All runs identical: {all_same}")
    
    # Analyze market focus impact (currently not used)
    print(f"\n⚠️ Current Issues Identified:")
    print("   1. Market Focus parameter is not used in symbol selection")
    print("   2. Analysis Type doesn't affect stock selection (only UI)")
    print("   3. Small stock counts may miss best opportunities")
    print("   4. No guarantee of analyzing same stocks across analysis types")
    
    return {
        'universe_size': universe_size,
        'consistency': all_same,
        'cap_coverage': {cap: len(get_symbols_by_cap(analyzer, cap, 200)) for cap in cap_filters}
    }

if __name__ == "__main__":
    results = analyze_current_behavior()
    
    print(f"\n📋 Summary:")
    print(f"   Universe Size: {results['universe_size']}")
    print(f"   Selection Consistency: {results['consistency']}")
    print(f"   Max Coverage: {results['cap_coverage']}")
    
    print(f"\n💡 Recommendations:")
    print("   1. Implement session-based stock selection caching")
    print("   2. Use Market Focus to filter/prioritize symbols")
    print("   3. Increase default analysis size for better coverage")
    print("   4. Make Analysis Type affect scoring weights, not stock selection")
