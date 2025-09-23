#!/usr/bin/env python3
"""
Debug data fetching issues
"""

import yfinance as yf
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_data_fetcher import AdvancedDataFetcher
import warnings
warnings.filterwarnings('ignore')

def test_yfinance_direct():
    """Test yfinance directly"""
    print("🧪 Testing yfinance directly...")
    
    test_symbols = ['AAPL', 'PLTR', 'CRWD', 'SNOW']
    
    for symbol in test_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo", interval="1d")
            
            if hist is not None and not hist.empty:
                print(f"✅ {symbol}: Got {len(hist)} days of data")
                print(f"   Latest close: ${hist['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ {symbol}: No data returned")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

def test_data_fetcher():
    """Test our data fetcher"""
    print("\n🧪 Testing AdvancedDataFetcher...")
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    test_symbols = ['AAPL', 'PLTR', 'CRWD', 'SNOW']
    
    for symbol in test_symbols:
        try:
            result = fetcher.get_comprehensive_stock_data(symbol)
            
            if result and result.get('data') is not None:
                df = result['data']
                if not df.empty:
                    print(f"✅ {symbol}: Got {len(df)} days of data")
                    print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")
                else:
                    print(f"❌ {symbol}: Empty dataframe")
            else:
                print(f"❌ {symbol}: No result returned")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

def test_bulk_fetch():
    """Test bulk history fetch"""
    print("\n🧪 Testing bulk history fetch...")
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    test_symbols = ['AAPL', 'PLTR', 'CRWD', 'SNOW', 'DDOG']
    
    try:
        hist_map = fetcher.get_bulk_history(test_symbols, period="1mo", interval="1d")
        
        print(f"📊 Bulk fetch returned {len(hist_map)} results")
        
        for symbol in test_symbols:
            df = hist_map.get(symbol)
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"✅ {symbol}: Got {len(df)} days of data")
            else:
                print(f"❌ {symbol}: No valid data")
                
    except Exception as e:
        print(f"❌ Bulk fetch error: {e}")

if __name__ == "__main__":
    print("🔍 Data Fetching Debug")
    print("=" * 50)
    
    # Test yfinance directly
    test_yfinance_direct()
    
    # Test our data fetcher
    test_data_fetcher()
    
    # Test bulk fetch
    test_bulk_fetch()
