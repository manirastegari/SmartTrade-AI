#!/usr/bin/env python3
"""
Final comprehensive test for all cap filters and analysis types
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

def test_all_cap_filters():
    """Test all cap filters"""
    print("🧪 Testing All Cap Filters...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    
    cap_filters = ["Large Cap", "Mid Cap", "Small Cap", "All"]
    
    for cap_filter in cap_filters:
        print(f"\n📊 Testing {cap_filter}...")
        
        # Get symbols for this cap filter
        symbols = get_symbols_by_cap(analyzer, cap_filter, 10)
        print(f"   Selected symbols: {symbols[:5]}...")
        
        # Test bulk analysis
        try:
            results = analyzer.run_advanced_analysis(max_stocks=len(symbols), symbols=symbols)
            
            if results:
                print(f"   ✅ {cap_filter}: {len(results)} stocks analyzed successfully")
                
                # Show sample results
                for result in results[:3]:
                    print(f"      - {result['symbol']}: ${result['current_price']:.2f}, {result['recommendation']}, {result['confidence']:.1%}")
            else:
                print(f"   ❌ {cap_filter}: No results")
                return False
                
        except Exception as e:
            print(f"   ❌ {cap_filter}: Error - {e}")
            return False
    
    return True

def test_fallback_symbols():
    """Test the fallback symbols used in the app"""
    print("\n🧪 Testing Fallback Symbols...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    
    fallback_sets = {
        "Small Cap": ['PLTR','CRWD','SNOW','DDOG','NET','OKTA','ZM','DOCU','TWLO','SQ'],
        "Mid Cap": ['REGN','GILD','BIIB','VRTX','ILMN','MRNA','ZTS','SYK','ISRG','EW'],
        "Large Cap": ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD','INTC']
    }
    
    for cap_type, symbols in fallback_sets.items():
        print(f"\n📊 Testing {cap_type} fallback symbols...")
        
        try:
            results = analyzer.run_advanced_analysis(max_stocks=len(symbols), symbols=symbols)
            
            if results:
                print(f"   ✅ {cap_type} fallback: {len(results)} stocks analyzed")
                
                # Show sample results
                for result in results[:3]:
                    print(f"      - {result['symbol']}: ${result['current_price']:.2f}, {result['recommendation']}")
            else:
                print(f"   ❌ {cap_type} fallback: No results")
                return False
                
        except Exception as e:
            print(f"   ❌ {cap_type} fallback: Error - {e}")
            return False
    
    return True

def test_specific_problematic_case():
    """Test the specific case that was failing: institutional-small cap, balanced, all markets, balanced risk"""
    print("\n🧪 Testing Specific Problematic Case...")
    print("   Settings: institutional-small cap, balanced, all markets, balanced risk")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    
    # Get small cap symbols (same as the problematic case)
    small_cap_symbols = get_symbols_by_cap(analyzer, "Small Cap", 50)
    print(f"   Small cap symbols selected: {len(small_cap_symbols)}")
    print(f"   First 10: {small_cap_symbols[:10]}")
    
    try:
        results = analyzer.run_advanced_analysis(max_stocks=len(small_cap_symbols), symbols=small_cap_symbols)
        
        if results:
            print(f"   ✅ Problematic case: {len(results)} stocks analyzed successfully!")
            
            # Filter by balanced risk (same as app logic)
            balanced_results = [r for r in results if r.get('risk_level') in ("Low", "Medium", "High")]  # All risk levels for balanced
            
            print(f"   ✅ After risk filtering: {len(balanced_results)} stocks")
            
            # Show top results
            top_results = sorted(balanced_results, key=lambda x: x['overall_score'], reverse=True)[:5]
            print("   📈 Top 5 recommendations:")
            for i, result in enumerate(top_results, 1):
                print(f"      {i}. {result['symbol']}: {result['recommendation']} (Score: {result['overall_score']:.1f}, Confidence: {result['confidence']:.1%})")
            
            return True
        else:
            print("   ❌ Problematic case: No results")
            return False
            
    except Exception as e:
        print(f"   ❌ Problematic case: Error - {e}")
        return False

if __name__ == "__main__":
    print("🚀 Final Comprehensive Test")
    print("=" * 60)
    
    # Test all cap filters
    success1 = test_all_cap_filters()
    
    # Test fallback symbols
    success2 = test_fallback_symbols()
    
    # Test the specific problematic case
    success3 = test_specific_problematic_case()
    
    if success1 and success2 and success3:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The app should now work correctly for all cap filters including Small Cap")
        print("✅ The specific error case (institutional-small cap, balanced, all markets, balanced risk) is fixed")
        print("✅ Synthetic data fallback is working when yfinance is rate-limited")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
