#!/usr/bin/env python3
"""
Final test for small cap analysis with improved fallback
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_analyzer import AdvancedTradingAnalyzer
import warnings
warnings.filterwarnings('ignore')

def test_small_cap_fallback():
    """Test small cap analysis with fallback symbols"""
    print("🧪 Testing Small Cap Analysis with Fallback...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    
    # Test with reliable small cap symbols (same as app fallback)
    small_cap_symbols = [
        'PLTR','CRWD','SNOW','DDOG','NET','OKTA','ZM','DOCU','TWLO','SQ'
    ]
    
    print(f"📈 Testing analysis on {len(small_cap_symbols)} reliable small cap stocks...")
    print(f"📊 Symbols: {small_cap_symbols}")
    
    try:
        results = analyzer.run_advanced_analysis(max_stocks=len(small_cap_symbols), symbols=small_cap_symbols)
        
        if results:
            print(f"✅ Small cap analysis successful!")
            print(f"   - Analyzed {len(results)} stocks")
            for result in results:
                print(f"   - {result['symbol']}: {result['recommendation']} ({result['confidence']:.1%} confidence, {result['prediction']:+.2f}% prediction)")
            return True
        else:
            print("⚠️ Small cap analysis returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Small cap analysis error: {e}")
        return False

def test_individual_small_caps():
    """Test individual small cap stocks"""
    print("\n🔍 Testing Individual Small Cap Stocks...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    
    test_symbols = ['PLTR', 'CRWD', 'SNOW', 'DDOG', 'NET']
    
    for symbol in test_symbols:
        try:
            result = analyzer.analyze_stock_comprehensive(symbol)
            if result:
                print(f"✅ {symbol}: ${result['current_price']:.2f}, {result['recommendation']}, {result['confidence']:.1%} confidence")
            else:
                print(f"⚠️ {symbol}: Analysis returned None")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

if __name__ == "__main__":
    print("🚀 Final Small Cap Analysis Test")
    print("=" * 50)
    
    # Test individual stocks first
    test_individual_small_caps()
    
    # Test bulk analysis
    success = test_small_cap_fallback()
    
    if success:
        print("\n🎉 Small cap analysis working correctly with fallback!")
        print("The app should now work properly with Small Cap selection.")
        sys.exit(0)
    else:
        print("\n❌ Small cap analysis still has issues.")
        sys.exit(1)
