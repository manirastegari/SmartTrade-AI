"""
Test script for the trading analyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_trading_analyzer import SimpleTradingAnalyzer
import pandas as pd

def test_basic_functionality():
    """Test basic functionality of the analyzer"""
    print("🧪 Testing Simple Trading Analyzer...")
    
    # Initialize analyzer
    analyzer = SimpleTradingAnalyzer()
    
    # Test single stock analysis
    print("\n📊 Testing single stock analysis (AAPL)...")
    result = analyzer.analyze_stock_auto('AAPL')
    
    if result:
        print("✅ Single stock analysis successful!")
        print(f"Symbol: {result['symbol']}")
        print(f"Price: ${result['current_price']:.2f}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Prediction: {result['prediction']:.2f}%")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Technical Score: {result['technical_score']}/100")
        print(f"Fundamental Score: {result['fundamental_score']}/100")
        print(f"Signals: {len(result['signals'])} signals generated")
    else:
        print("❌ Single stock analysis failed!")
        return False
    
    # Test multiple stocks
    print("\n📈 Testing multiple stock analysis...")
    results = analyzer.run_automated_analysis(max_stocks=5)
    
    if results:
        print(f"✅ Multiple stock analysis successful! Analyzed {len(results)} stocks")
        
        # Show top picks
        top_picks = analyzer.get_top_picks(results, 3)
        print("\n🏆 Top 3 Picks:")
        for i, pick in enumerate(top_picks, 1):
            print(f"{i}. {pick['symbol']} - {pick['recommendation']} - {pick['action']}")
        
        # Show summary statistics
        df = pd.DataFrame(results)
        print(f"\n📊 Summary Statistics:")
        print(f"Average Prediction: {df['prediction'].mean():.2f}%")
        print(f"Average Confidence: {df['confidence'].mean():.1%}")
        print(f"Strong Buys: {len(df[df['recommendation'] == 'STRONG BUY'])}")
        print(f"Buys: {len(df[df['recommendation'] == 'BUY'])}")
        print(f"Holds: {len(df[df['recommendation'] == 'HOLD'])}")
        
        return True
    else:
        print("❌ Multiple stock analysis failed!")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 All tests passed! The analyzer is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the error messages above.")
