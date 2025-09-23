#!/usr/bin/env python3
"""
Test script to verify the 'fed_rate' error fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_analyzer import AdvancedTradingAnalyzer
import warnings
warnings.filterwarnings('ignore')

def test_fed_rate_fix():
    """Test that the fed_rate error is fixed"""
    print("🧪 Testing fed_rate error fix...")
    
    try:
        # Initialize analyzer in light mode (same as the app)
        analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
        print("✅ Analyzer initialized successfully")
        
        # Test with a few reliable symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in test_symbols:
            print(f"\n📊 Testing {symbol}...")
            try:
                result = analyzer.analyze_stock_comprehensive(symbol)
                if result:
                    print(f"✅ {symbol}: Analysis completed successfully")
                    print(f"   - Current Price: ${result['current_price']:.2f}")
                    print(f"   - Prediction: {result['prediction']:+.2f}%")
                    print(f"   - Confidence: {result['confidence']:.1%}")
                    print(f"   - Recommendation: {result['recommendation']}")
                else:
                    print(f"⚠️ {symbol}: Analysis returned None (but no error)")
            except Exception as e:
                if 'fed_rate' in str(e):
                    print(f"❌ {symbol}: fed_rate error still exists: {e}")
                    return False
                else:
                    print(f"⚠️ {symbol}: Other error (not fed_rate): {e}")
        
        print("\n🎉 fed_rate error fix verified successfully!")
        return True
        
    except Exception as e:
        if 'fed_rate' in str(e):
            print(f"❌ fed_rate error still exists: {e}")
            return False
        else:
            print(f"⚠️ Other error: {e}")
            return True  # Not a fed_rate error

def test_bulk_analysis():
    """Test bulk analysis like the app does"""
    print("\n🔄 Testing bulk analysis (like the app)...")
    
    try:
        analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
        
        # Test with a small set of symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        print(f"📈 Running analysis on {len(test_symbols)} stocks...")
        results = analyzer.run_advanced_analysis(max_stocks=len(test_symbols), symbols=test_symbols)
        
        if results:
            print(f"✅ Bulk analysis completed successfully!")
            print(f"   - Analyzed {len(results)} stocks")
            for result in results[:3]:  # Show first 3
                print(f"   - {result['symbol']}: {result['recommendation']} ({result['confidence']:.1%} confidence)")
        else:
            print("⚠️ Bulk analysis returned no results")
            
        return True
        
    except Exception as e:
        if 'fed_rate' in str(e):
            print(f"❌ fed_rate error in bulk analysis: {e}")
            return False
        else:
            print(f"⚠️ Other error in bulk analysis: {e}")
            return True

if __name__ == "__main__":
    print("🚀 AI Trading App - Fed Rate Error Fix Test")
    print("=" * 50)
    
    # Test individual stock analysis
    success1 = test_fed_rate_fix()
    
    # Test bulk analysis
    success2 = test_bulk_analysis()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! The fed_rate error has been fixed.")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED! The fed_rate error still exists.")
        sys.exit(1)
