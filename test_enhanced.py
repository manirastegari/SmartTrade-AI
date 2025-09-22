"""
Test script for the enhanced trading analyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_analyzer import EnhancedTradingAnalyzer
import pandas as pd

def test_enhanced_functionality():
    """Test enhanced functionality of the analyzer"""
    print("🚀 Testing Enhanced Trading Analyzer...")
    
    # Initialize analyzer
    analyzer = EnhancedTradingAnalyzer()
    
    # Test single stock analysis
    print("\n📊 Testing enhanced single stock analysis (AAPL)...")
    result = analyzer.analyze_stock_comprehensive('AAPL')
    
    if result:
        print("✅ Enhanced single stock analysis successful!")
        print(f"Symbol: {result['symbol']}")
        print(f"Price: ${result['current_price']:.2f}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Prediction: {result['prediction']:.2f}%")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Technical Score: {result['technical_score']}/100")
        print(f"Fundamental Score: {result['fundamental_score']}/100")
        print(f"Sentiment Score: {result['sentiment_score']}/100")
        print(f"Momentum Score: {result['momentum_score']}/100")
        print(f"Volume Score: {result['volume_score']}/100")
        print(f"Volatility Score: {result['volatility_score']}/100")
        print(f"Overall Score: {result['overall_score']}/100")
        print(f"Enhanced Signals: {len(result['signals'])} signals generated")
        
        # Show some signals
        print("\n🔍 Sample Signals:")
        for signal in result['signals'][:5]:
            print(f"  • {signal}")
    else:
        print("❌ Enhanced single stock analysis failed!")
        return False
    
    # Test multiple stocks
    print("\n📈 Testing enhanced multiple stock analysis...")
    results = analyzer.run_enhanced_analysis(max_stocks=10)
    
    if results:
        print(f"✅ Enhanced multiple stock analysis successful! Analyzed {len(results)} stocks")
        
        # Show top picks
        top_picks = analyzer.get_top_picks_enhanced(results, 5)
        print("\n🏆 Top 5 Enhanced Picks:")
        for i, pick in enumerate(top_picks, 1):
            print(f"{i}. {pick['symbol']} - {pick['recommendation']} - {pick['action']} (Score: {pick['overall_score']:.1f})")
        
        # Show summary statistics
        df = pd.DataFrame(results)
        print(f"\n📊 Enhanced Summary Statistics:")
        print(f"Average Prediction: {df['prediction'].mean():.2f}%")
        print(f"Average Confidence: {df['confidence'].mean():.1%}")
        print(f"Average Overall Score: {df['overall_score'].mean():.1f}")
        print(f"Strong Buys: {len(df[df['recommendation'].str.contains('STRONG BUY', na=False)])}")
        print(f"Buys: {len(df[df['recommendation'].str.contains('BUY', na=False)])}")
        print(f"Holds: {len(df[df['recommendation'] == 'HOLD'])}")
        print(f"High Risk: {len(df[df['risk_level'] == 'High'])}")
        print(f"Low Risk: {len(df[df['risk_level'] == 'Low'])}")
        
        # Show score breakdown
        print(f"\n📈 Score Breakdown:")
        print(f"Average Technical Score: {df['technical_score'].mean():.1f}")
        print(f"Average Fundamental Score: {df['fundamental_score'].mean():.1f}")
        print(f"Average Sentiment Score: {df['sentiment_score'].mean():.1f}")
        print(f"Average Momentum Score: {df['momentum_score'].mean():.1f}")
        print(f"Average Volume Score: {df['volume_score'].mean():.1f}")
        print(f"Average Volatility Score: {df['volatility_score'].mean():.1f}")
        
        return True
    else:
        print("❌ Enhanced multiple stock analysis failed!")
        return False

if __name__ == "__main__":
    success = test_enhanced_functionality()
    if success:
        print("\n🎉 All enhanced tests passed! The analyzer is working correctly.")
    else:
        print("\n💥 Some enhanced tests failed. Check the error messages above.")
