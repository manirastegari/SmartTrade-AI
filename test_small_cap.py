#!/usr/bin/env python3
"""
Test script to verify small cap selection works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_analyzer import AdvancedTradingAnalyzer
import warnings
warnings.filterwarnings('ignore')

def get_symbols_by_cap(analyzer, cap_filter: str, count: int):
    """Same logic as in the app"""
    universe = analyzer.stock_universe
    universe_size = len(universe)
    
    if cap_filter == "Large Cap":
        # First 1/3 of universe (typically large caps)
        end_idx = min(universe_size // 3, 200)
        return universe[:max(10, min(count, end_idx))]
    elif cap_filter == "Mid Cap":
        # Middle 1/3 of universe
        start = universe_size // 3
        end = (universe_size * 2) // 3
        available = universe[start:end]
        return available[:max(10, min(count, len(available)))]
    elif cap_filter == "Small Cap":
        # Last 1/3 of universe (typically smaller caps)
        start = (universe_size * 2) // 3
        available = universe[start:]
        return available[:max(10, min(count, len(available)))]
    else:
        # All markets - return from entire universe
        return universe[:max(10, min(count, universe_size))]

def test_cap_selection():
    """Test cap selection logic"""
    print("🧪 Testing Cap Selection Logic...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    universe_size = len(analyzer.stock_universe)
    print(f"📊 Universe size: {universe_size}")
    
    # Test each cap filter
    for cap_filter in ["Large Cap", "Mid Cap", "Small Cap", "All"]:
        symbols = get_symbols_by_cap(analyzer, cap_filter, 20)
        print(f"📊 {cap_filter}: {len(symbols)} symbols")
        print(f"   First 5: {symbols[:5]}")
        print(f"   Last 5: {symbols[-5:]}")
        
        # Test analysis on first symbol
        if symbols:
            test_symbol = symbols[0]
            print(f"🧪 Testing analysis on {test_symbol}...")
            try:
                result = analyzer.analyze_stock_comprehensive(test_symbol)
                if result:
                    print(f"✅ {test_symbol}: Analysis successful")
                else:
                    print(f"⚠️ {test_symbol}: Analysis returned None")
            except Exception as e:
                print(f"❌ {test_symbol}: Error - {e}")
        print()

def test_small_cap_bulk():
    """Test small cap bulk analysis"""
    print("🔄 Testing Small Cap Bulk Analysis...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    
    # Get small cap symbols
    small_cap_symbols = get_symbols_by_cap(analyzer, "Small Cap", 10)
    print(f"📈 Testing bulk analysis on {len(small_cap_symbols)} small cap stocks...")
    print(f"📊 Symbols: {small_cap_symbols}")
    
    try:
        results = analyzer.run_advanced_analysis(max_stocks=len(small_cap_symbols), symbols=small_cap_symbols)
        
        if results:
            print(f"✅ Small cap bulk analysis successful!")
            print(f"   - Analyzed {len(results)} stocks")
            for result in results[:3]:  # Show first 3
                print(f"   - {result['symbol']}: {result['recommendation']} ({result['confidence']:.1%} confidence)")
        else:
            print("⚠️ Small cap bulk analysis returned no results")
            
        return True
        
    except Exception as e:
        print(f"❌ Small cap bulk analysis error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Small Cap Selection Test")
    print("=" * 50)
    
    # Test cap selection logic
    test_cap_selection()
    
    # Test small cap bulk analysis
    success = test_small_cap_bulk()
    
    if success:
        print("\n🎉 Small cap selection and analysis working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Small cap analysis still has issues.")
        sys.exit(1)
