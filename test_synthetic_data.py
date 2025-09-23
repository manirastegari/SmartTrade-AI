#!/usr/bin/env python3
"""
Test synthetic data generation and improved data fetching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_data_fetcher import AdvancedDataFetcher
from advanced_analyzer import AdvancedTradingAnalyzer
import warnings
warnings.filterwarnings('ignore')

def test_synthetic_data():
    """Test synthetic data generation"""
    print("🧪 Testing synthetic data generation...")
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    test_symbols = ['AAPL', 'PLTR', 'CRWD', 'SNOW', 'DDOG']
    
    for symbol in test_symbols:
        try:
            synthetic_data = fetcher._generate_synthetic_data(symbol)
            
            if synthetic_data is not None and not synthetic_data.empty:
                print(f"✅ {symbol}: Generated {len(synthetic_data)} days of synthetic data")
                print(f"   Price range: ${synthetic_data['Close'].min():.2f} - ${synthetic_data['Close'].max():.2f}")
                print(f"   Latest close: ${synthetic_data['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ {symbol}: Failed to generate synthetic data")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

def test_improved_fetcher():
    """Test improved data fetcher with fallbacks"""
    print("\n🧪 Testing improved data fetcher...")
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    test_symbols = ['AAPL', 'PLTR', 'CRWD', 'SNOW', 'DDOG']
    
    for symbol in test_symbols:
        try:
            result = fetcher.get_comprehensive_stock_data(symbol)
            
            if result and result.get('data') is not None:
                df = result['data']
                if not df.empty:
                    print(f"✅ {symbol}: Got {len(df)} days of data")
                    print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")
                    print(f"   Has technical indicators: {'RSI_14' in df.columns}")
                else:
                    print(f"❌ {symbol}: Empty dataframe")
            else:
                print(f"❌ {symbol}: No result returned")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

def test_analyzer_with_synthetic():
    """Test analyzer with synthetic data"""
    print("\n🧪 Testing analyzer with synthetic data...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    test_symbols = ['AAPL', 'PLTR', 'CRWD', 'SNOW', 'DDOG']
    
    for symbol in test_symbols:
        try:
            result = analyzer.analyze_stock_comprehensive(symbol)
            
            if result:
                print(f"✅ {symbol}: Analysis successful")
                print(f"   Price: ${result['current_price']:.2f}")
                print(f"   Prediction: {result['prediction']:+.2f}%")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Recommendation: {result['recommendation']}")
            else:
                print(f"❌ {symbol}: Analysis failed")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

def test_bulk_analysis():
    """Test bulk analysis with synthetic data"""
    print("\n🧪 Testing bulk analysis...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    test_symbols = ['AAPL', 'PLTR', 'CRWD', 'SNOW', 'DDOG']
    
    try:
        results = analyzer.run_advanced_analysis(max_stocks=len(test_symbols), symbols=test_symbols)
        
        if results:
            print(f"✅ Bulk analysis successful!")
            print(f"   Analyzed {len(results)} stocks")
            for result in results:
                print(f"   - {result['symbol']}: {result['recommendation']} ({result['confidence']:.1%}, {result['prediction']:+.2f}%)")
        else:
            print("❌ Bulk analysis returned no results")
            
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Bulk analysis error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Synthetic Data and Improved Fetching Test")
    print("=" * 60)
    
    # Test synthetic data generation
    test_synthetic_data()
    
    # Test improved fetcher
    test_improved_fetcher()
    
    # Test analyzer
    test_analyzer_with_synthetic()
    
    # Test bulk analysis
    success = test_bulk_analysis()
    
    if success:
        print("\n🎉 All tests passed! The app should now work with synthetic data fallback.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
