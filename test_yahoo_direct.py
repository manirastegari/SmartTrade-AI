#!/usr/bin/env python3
"""
Test Yahoo Direct API integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_data_fetcher import AdvancedDataFetcher

def test_yahoo_direct():
    """Test the Yahoo Direct API method"""
    
    print("🧪 Testing Yahoo Direct API Integration")
    print("=" * 50)
    
    # Initialize fetcher
    fetcher = AdvancedDataFetcher(data_mode="light")
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}:")
        
        try:
            # Test Yahoo Direct API method directly
            data = fetcher._try_yahoo_direct_api(symbol)
            
            if data is not None and not data.empty:
                print(f"  ✅ Yahoo Direct API: {len(data)} days")
                print(f"  📈 Latest price: ${data['Close'].iloc[-1]:.2f}")
                print(f"  📊 Latest volume: {data['Volume'].iloc[-1]:,.0f}")
                print(f"  📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                
                # Test validation
                is_valid = fetcher._validate_market_data(data, symbol)
                print(f"  🛡️ Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
                
            else:
                print(f"  ❌ Yahoo Direct API: No data")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test complete!")

if __name__ == "__main__":
    test_yahoo_direct()
