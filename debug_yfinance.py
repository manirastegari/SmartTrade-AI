#!/usr/bin/env python3
"""
Debug yfinance issues with different parameters
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_different_periods():
    """Test different time periods"""
    print("🧪 Testing different time periods...")
    
    symbol = 'AAPL'
    periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y']
    
    for period in periods:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist is not None and not hist.empty:
                print(f"✅ {symbol} {period}: Got {len(hist)} days of data")
                print(f"   Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
                print(f"   Latest close: ${hist['Close'].iloc[-1]:.2f}")
                return True  # Found working period
            else:
                print(f"❌ {symbol} {period}: No data")
                
        except Exception as e:
            print(f"❌ {symbol} {period}: Error - {e}")
    
    return False

def test_different_symbols():
    """Test different symbols to see if any work"""
    print("\n🧪 Testing different symbols...")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'IWM', 'PLTR', 'TSLA']
    
    working_symbols = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if hist is not None and not hist.empty:
                print(f"✅ {symbol}: Got {len(hist)} days of data")
                print(f"   Latest close: ${hist['Close'].iloc[-1]:.2f}")
                working_symbols.append(symbol)
            else:
                print(f"❌ {symbol}: No data")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    return working_symbols

def test_yfinance_download():
    """Test yfinance download function"""
    print("\n🧪 Testing yfinance download function...")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    try:
        data = yf.download(symbols, period="5d", group_by='ticker', progress=False)
        
        if data is not None and not data.empty:
            print(f"✅ Download successful: {data.shape}")
            print(f"   Columns: {data.columns.tolist()[:10]}...")  # Show first 10 columns
            
            # Check individual symbols
            for symbol in symbols:
                if symbol in data.columns.levels[0]:
                    symbol_data = data[symbol]
                    if not symbol_data.empty:
                        print(f"   ✅ {symbol}: {len(symbol_data)} days")
                    else:
                        print(f"   ❌ {symbol}: Empty")
                else:
                    print(f"   ❌ {symbol}: Not found in data")
        else:
            print("❌ Download returned no data")
            
    except Exception as e:
        print(f"❌ Download error: {e}")

def test_network_connectivity():
    """Test basic network connectivity"""
    print("\n🧪 Testing network connectivity...")
    
    import requests
    
    try:
        response = requests.get("https://finance.yahoo.com", timeout=10)
        if response.status_code == 200:
            print("✅ Yahoo Finance website accessible")
        else:
            print(f"⚠️ Yahoo Finance returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Network error: {e}")

if __name__ == "__main__":
    print("🔍 yfinance Debug Analysis")
    print("=" * 50)
    
    # Test network first
    test_network_connectivity()
    
    # Test different periods
    working = test_different_periods()
    
    if not working:
        # Test different symbols
        working_symbols = test_different_symbols()
        
        if working_symbols:
            print(f"\n✅ Found {len(working_symbols)} working symbols: {working_symbols}")
        else:
            print("\n❌ No symbols are working - likely yfinance API issue")
    
    # Test download function
    test_yfinance_download()
