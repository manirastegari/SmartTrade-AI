#!/usr/bin/env python3
"""
Data Integrity Checker for AI Trading Application
Validates that real market data is being used, not synthetic data
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_data_fetcher import AdvancedDataFetcher

def check_data_integrity():
    """Check data integrity for a sample of stocks"""
    
    print("🛡️ AI Trading Application - Data Integrity Check")
    print("=" * 60)
    print(f"📅 Check Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'JPM', 'BAC', 'KO']
    
    # Initialize data fetcher
    fetcher = AdvancedDataFetcher(data_mode="light")
    
    print(f"🔍 Testing data integrity for {len(test_symbols)} symbols...")
    print("-" * 40)
    
    results = {
        'valid_data': [],
        'invalid_data': [],
        'no_data': []
    }
    
    for symbol in test_symbols:
        print(f"📊 Checking {symbol}...", end=" ")
        
        try:
            # Get stock data
            stock_data = fetcher.get_comprehensive_stock_data(symbol)
            
            if not stock_data or stock_data['data'].empty:
                print("❌ No data")
                results['no_data'].append(symbol)
                continue
            
            df = stock_data['data']
            
            # Validate data
            if fetcher._validate_market_data(df, symbol):
                current_price = df['Close'].iloc[-1]
                volume = df['Volume'].iloc[-1]
                data_points = len(df)
                date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
                
                print(f"✅ Valid - ${current_price:.2f}, Vol: {volume:,.0f}, {data_points} days ({date_range})")
                results['valid_data'].append({
                    'symbol': symbol,
                    'price': current_price,
                    'volume': volume,
                    'data_points': data_points,
                    'date_range': date_range
                })
            else:
                print("❌ Invalid data (failed validation)")
                results['invalid_data'].append(symbol)
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            results['no_data'].append(symbol)
    
    print()
    print("📈 DATA INTEGRITY SUMMARY")
    print("=" * 60)
    
    total = len(test_symbols)
    valid = len(results['valid_data'])
    invalid = len(results['invalid_data'])
    no_data = len(results['no_data'])
    
    print(f"✅ Valid data:    {valid:2d}/{total} ({valid/total*100:5.1f}%)")
    print(f"❌ Invalid data:  {invalid:2d}/{total} ({invalid/total*100:5.1f}%)")
    print(f"🚫 No data:       {no_data:2d}/{total} ({no_data/total*100:5.1f}%)")
    
    print()
    
    if valid >= 8:  # At least 80% valid
        print("🎉 DATA INTEGRITY: EXCELLENT")
        print("✅ Your application is using real market data")
    elif valid >= 6:  # At least 60% valid
        print("⚠️ DATA INTEGRITY: ACCEPTABLE")
        print("🔄 Some data sources may be having issues")
    else:
        print("🚨 DATA INTEGRITY: CRITICAL ISSUE")
        print("❌ Most data appears to be invalid or missing")
        print("🔧 Immediate action required!")
    
    print()
    
    # Detailed results
    if results['valid_data']:
        print("📊 VALID DATA DETAILS")
        print("-" * 40)
        for data in results['valid_data']:
            print(f"{data['symbol']:5} | ${data['price']:8.2f} | Vol: {data['volume']:>10,.0f} | {data['data_points']} days")
    
    if results['invalid_data']:
        print()
        print("❌ INVALID DATA SYMBOLS")
        print("-" * 40)
        print(", ".join(results['invalid_data']))
    
    if results['no_data']:
        print()
        print("🚫 NO DATA SYMBOLS")
        print("-" * 40)
        print(", ".join(results['no_data']))
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 40)
    
    if valid >= 8:
        print("✅ Continue using the application - data quality is excellent")
    elif valid >= 6:
        print("🔄 Monitor data quality - consider running this check regularly")
        print("📡 Check your internet connection and try again later")
    else:
        print("🚨 DO NOT USE FOR TRADING - data quality is too poor")
        print("🔧 Check network connectivity and API access")
        print("⏰ Wait for market hours (9:30 AM - 4:00 PM ET)")
        print("🔄 Try running the application again later")
    
    print()
    print("🛡️ Data integrity check complete!")
    
    return results

if __name__ == "__main__":
    try:
        results = check_data_integrity()
    except KeyboardInterrupt:
        print("\n⏹️  Check interrupted by user")
    except Exception as e:
        print(f"\n❌ Check failed with error: {e}")
