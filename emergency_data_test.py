#!/usr/bin/env python3
"""
Emergency Data Test - Try alternative data sources
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def test_yahoo_finance_direct():
    """Test Yahoo Finance directly via web scraping"""
    print("🌐 Testing Yahoo Finance Direct...")
    
    symbol = "AAPL"
    try:
        # Yahoo Finance quote API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        params = {
            'range': '1mo',
            'interval': '1d'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                
                if 'timestamp' in result and 'indicators' in result:
                    timestamps = result['timestamp']
                    quotes = result['indicators']['quote'][0]
                    
                    # Get latest data
                    latest_idx = -1
                    latest_price = quotes['close'][latest_idx]
                    latest_volume = quotes['volume'][latest_idx]
                    latest_date = datetime.fromtimestamp(timestamps[latest_idx])
                    
                    print(f"  ✅ Yahoo Direct API Success!")
                    print(f"  📈 {symbol}: ${latest_price:.2f}")
                    print(f"  📊 Volume: {latest_volume:,.0f}")
                    print(f"  📅 Date: {latest_date.strftime('%Y-%m-%d')}")
                    
                    return True
        
        print(f"  ❌ Yahoo Direct API failed")
        return False
        
    except Exception as e:
        print(f"  ❌ Yahoo Direct error: {e}")
        return False

def test_alpha_vantage_demo():
    """Test Alpha Vantage demo API"""
    print("\n📡 Testing Alpha Vantage Demo...")
    
    symbol = "IBM"  # Demo API works better with IBM
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '60min',
            'apikey': 'demo'
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'Time Series (60min)' in data:
                time_series = data['Time Series (60min)']
                
                # Get latest data
                latest_time = max(time_series.keys())
                latest_data = time_series[latest_time]
                
                price = float(latest_data['4. close'])
                volume = int(latest_data['5. volume'])
                
                print(f"  ✅ Alpha Vantage Demo Success!")
                print(f"  📈 {symbol}: ${price:.2f}")
                print(f"  📊 Volume: {volume:,.0f}")
                print(f"  📅 Time: {latest_time}")
                
                return True
            else:
                print(f"  ❌ Alpha Vantage: {data.get('Note', 'No data')}")
                return False
        
        print(f"  ❌ Alpha Vantage failed")
        return False
        
    except Exception as e:
        print(f"  ❌ Alpha Vantage error: {e}")
        return False

def test_market_status():
    """Check if market is open"""
    print("\n🕐 Checking Market Status...")
    
    try:
        now = datetime.now()
        
        # Market hours: 9:30 AM - 4:00 PM ET (Mon-Fri)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
        is_market_hours = market_open <= now <= market_close
        
        print(f"  📅 Current time: {now.strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"  📊 Weekday: {'Yes' if is_weekday else 'No'}")
        print(f"  🕘 Market hours: {'Yes' if is_market_hours else 'No'}")
        
        if is_weekday and is_market_hours:
            print(f"  ✅ Market is OPEN")
            return True
        else:
            if not is_weekday:
                print(f"  ❌ Market is CLOSED (Weekend)")
            else:
                print(f"  ❌ Market is CLOSED (After hours)")
                print(f"  ⏰ Market opens at 9:30 AM ET")
            return False
            
    except Exception as e:
        print(f"  ❌ Market status error: {e}")
        return False

def test_simple_yfinance():
    """Test yfinance with minimal setup"""
    print("\n🔄 Testing Simple yfinance...")
    
    try:
        import yfinance as yf
        
        # Try with a very simple approach
        symbol = "AAPL"
        ticker = yf.Ticker(symbol)
        
        # Try just getting basic info (no history)
        info = ticker.info
        
        if info and 'regularMarketPrice' in info:
            price = info['regularMarketPrice']
            print(f"  ✅ yfinance basic info success!")
            print(f"  📈 {symbol}: ${price:.2f}")
            return True
        else:
            print(f"  ❌ yfinance basic info failed")
            return False
            
    except Exception as e:
        print(f"  ❌ yfinance error: {e}")
        return False

def main():
    print("🚨 EMERGENCY DATA SOURCE TEST")
    print("=" * 60)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Test market status first
    market_open = test_market_status()
    
    # Test various data sources
    tests = [
        ("Yahoo Finance Direct", test_yahoo_finance_direct),
        ("Alpha Vantage Demo", test_alpha_vantage_demo),
        ("Simple yfinance", test_simple_yfinance)
    ]
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("🎯 EMERGENCY TEST RESULTS")
    print("=" * 60)
    
    working_sources = []
    for test_name, success in results:
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{test_name:20} | {status}")
        if success:
            working_sources.append(test_name)
    
    print()
    
    if working_sources:
        print("🎉 GOOD NEWS: Some data sources are working!")
        print("✅ Working sources:")
        for source in working_sources:
            print(f"  - {source}")
        
        print("\n💡 RECOMMENDATIONS:")
        if not market_open:
            print("  ⏰ Wait for market hours (9:30 AM - 4:00 PM ET)")
            print("  📊 Some sources may work better during market hours")
        else:
            print("  🚀 Try running the main application now")
            print("  📊 Use working data sources for analysis")
        
    else:
        print("🚨 BAD NEWS: No data sources are working")
        print("\n💡 POSSIBLE CAUSES:")
        print("  🌐 Network connectivity issues")
        print("  🔒 Firewall/proxy blocking requests")
        print("  ⏰ Market is closed (try during 9:30 AM - 4:00 PM ET)")
        print("  🚫 All APIs are temporarily down")
        
        print("\n🔧 SOLUTIONS:")
        print("  1. Check internet connection")
        print("  2. Try from different network")
        print("  3. Wait for market hours")
        print("  4. Consider paid data sources")
    
    print("\n🛡️ Emergency test complete!")

if __name__ == "__main__":
    main()
