#!/usr/bin/env python3
"""
Fix Data Sources - Test and improve yfinance reliability
"""

import yfinance as yf
import pandas as pd
import time
from datetime import datetime

def test_yfinance_direct():
    """Test yfinance directly to see what's happening"""
    
    print("🔧 Testing yfinance directly...")
    print("=" * 50)
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        try:
            # Test basic ticker creation
            ticker = yf.Ticker(symbol)
            print(f"  ✅ Ticker created")
            
            # Test history fetch
            hist = ticker.history(period="5d", interval="1d")
            
            if hist is not None and not hist.empty:
                print(f"  ✅ Data fetched: {len(hist)} days")
                print(f"  📈 Latest price: ${hist['Close'].iloc[-1]:.2f}")
                print(f"  📊 Latest volume: {hist['Volume'].iloc[-1]:,.0f}")
                print(f"  📅 Date range: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"  ❌ No data returned")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        time.sleep(0.5)  # Rate limiting

def test_alternative_methods():
    """Test alternative data fetching methods"""
    
    print("\n🔄 Testing alternative methods...")
    print("=" * 50)
    
    symbol = 'AAPL'
    
    # Method 1: Bulk download
    print(f"\n1️⃣ Testing bulk download for {symbol}:")
    try:
        data = yf.download(symbol, period="5d", progress=False)
        if data is not None and not data.empty:
            print(f"  ✅ Bulk download success: {len(data)} days")
            print(f"  📈 Latest price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ❌ Bulk download failed")
    except Exception as e:
        print(f"  ❌ Bulk download error: {e}")
    
    # Method 2: Different period
    print(f"\n2️⃣ Testing different period for {symbol}:")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1mo")
        if data is not None and not data.empty:
            print(f"  ✅ 1 month data success: {len(data)} days")
            print(f"  📈 Latest price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ❌ 1 month data failed")
    except Exception as e:
        print(f"  ❌ 1 month data error: {e}")
    
    # Method 3: Start/End dates
    print(f"\n3️⃣ Testing start/end dates for {symbol}:")
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if data is not None and not data.empty:
            print(f"  ✅ Date range success: {len(data)} days")
            print(f"  📈 Latest price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ❌ Date range failed")
    except Exception as e:
        print(f"  ❌ Date range error: {e}")

def main():
    print("🛠️ Data Source Diagnostic Tool")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test direct yfinance
    test_yfinance_direct()
    
    # Test alternative methods
    test_alternative_methods()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")
    print("\nIf all tests show ❌, the issue is likely:")
    print("  1. Network connectivity problems")
    print("  2. yfinance API is temporarily down")
    print("  3. Rate limiting from too many requests")
    print("  4. Market is closed and no recent data available")
    print("\nIf some tests show ✅, we can use those methods!")

if __name__ == "__main__":
    main()
