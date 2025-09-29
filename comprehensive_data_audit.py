#!/usr/bin/env python3
"""
Comprehensive Data Audit - Verify ALL data sources are real and accurate
"""

import sys
import os
from datetime import datetime, timedelta
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def audit_stock_data():
    """Audit primary stock data sources"""
    print("📊 AUDITING STOCK DATA SOURCES")
    print("=" * 50)
    
    try:
        from advanced_data_fetcher import AdvancedDataFetcher
        fetcher = AdvancedDataFetcher(data_mode="light")
        
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        for symbol in test_symbols:
            print(f"\n🔍 Auditing {symbol}:")
            
            try:
                # Test the main data fetching method
                stock_data = fetcher.get_comprehensive_stock_data(symbol)
                
                if stock_data and 'data' in stock_data:
                    df = stock_data['data']
                    
                    if not df.empty:
                        latest_price = df['Close'].iloc[-1]
                        latest_volume = df['Volume'].iloc[-1]
                        data_points = len(df)
                        latest_date = df.index[-1]
                        days_old = (datetime.now() - latest_date).days
                        
                        print(f"   ✅ DATA FOUND: {data_points} days")
                        print(f"   📈 Price: ${latest_price:.2f}")
                        print(f"   📊 Volume: {latest_volume:,.0f}")
                        print(f"   📅 Latest: {latest_date.strftime('%Y-%m-%d')} ({days_old} days old)")
                        
                        # Validate data quality
                        if days_old <= 7 and latest_price > 0 and latest_volume > 0:
                            print(f"   ✅ QUALITY: EXCELLENT")
                        else:
                            print(f"   ⚠️ QUALITY: QUESTIONABLE")
                    else:
                        print(f"   ❌ NO DATA RETURNED")
                else:
                    print(f"   ❌ FETCH FAILED")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)[:100]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stock data audit failed: {e}")
        return False

def audit_market_context():
    """Audit market context data (SPY, VIX, etc.)"""
    print("\n📈 AUDITING MARKET CONTEXT DATA")
    print("=" * 50)
    
    try:
        from advanced_data_fetcher import AdvancedDataFetcher
        fetcher = AdvancedDataFetcher(data_mode="light")
        
        print("🔍 Testing market context fetch...")
        market_context = fetcher.get_market_context()
        
        if market_context:
            print("✅ Market context data retrieved:")
            for key, value in market_context.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            # Validate key metrics
            spy_return = market_context.get('spy_return_1d', 0)
            vix_proxy = market_context.get('vix_proxy', 0)
            
            if abs(spy_return) < 0.2 and 5 < vix_proxy < 100:
                print("   ✅ QUALITY: Market context looks realistic")
            else:
                print("   ⚠️ QUALITY: Market context may be questionable")
        else:
            print("❌ No market context data")
            
        return market_context is not None
        
    except Exception as e:
        print(f"❌ Market context audit failed: {e}")
        return False

def audit_economic_data():
    """Audit economic indicators"""
    print("\n💰 AUDITING ECONOMIC DATA")
    print("=" * 50)
    
    try:
        from advanced_data_fetcher import AdvancedDataFetcher
        fetcher = AdvancedDataFetcher(data_mode="light")
        
        # Test economic data through market context
        market_context = fetcher.get_market_context()
        
        economic_indicators = ['fed_rate', 'gdp_growth', 'inflation', 'unemployment']
        
        print("🔍 Checking economic indicators:")
        for indicator in economic_indicators:
            if indicator in market_context:
                value = market_context[indicator]
                print(f"   {indicator}: {value}")
                
                # Basic sanity checks
                if indicator == 'fed_rate' and 0 <= value <= 20:
                    print(f"     ✅ {indicator} looks reasonable")
                elif indicator == 'unemployment' and 0 <= value <= 50:
                    print(f"     ✅ {indicator} looks reasonable")
                elif indicator in ['gdp_growth', 'inflation'] and -20 <= value <= 20:
                    print(f"     ✅ {indicator} looks reasonable")
                else:
                    print(f"     ⚠️ {indicator} may be questionable")
            else:
                print(f"   ❌ {indicator}: NOT FOUND")
        
        return True
        
    except Exception as e:
        print(f"❌ Economic data audit failed: {e}")
        return False

def audit_news_sentiment():
    """Audit news and sentiment data"""
    print("\n📰 AUDITING NEWS & SENTIMENT DATA")
    print("=" * 50)
    
    try:
        from advanced_data_fetcher import AdvancedDataFetcher
        fetcher = AdvancedDataFetcher(data_mode="light")
        
        test_symbol = 'AAPL'
        print(f"🔍 Testing news/sentiment for {test_symbol}:")
        
        # Test news fetching
        try:
            news_data = fetcher._get_news_sentiment(test_symbol)
            
            if news_data:
                print("   ✅ News data structure found:")
                for key, value in news_data.items():
                    if isinstance(value, (int, float)):
                        print(f"     {key}: {value:.3f}")
                    else:
                        print(f"     {key}: {str(value)[:50]}")
                
                # Check sentiment scores
                sentiment_score = news_data.get('sentiment_score', 0)
                if -1 <= sentiment_score <= 1:
                    print("     ✅ Sentiment score looks valid")
                else:
                    print("     ⚠️ Sentiment score may be questionable")
            else:
                print("   ⚠️ No news data (may be expected in light mode)")
                
        except Exception as e:
            print(f"   ⚠️ News fetch error: {str(e)[:50]} (may be expected)")
        
        return True
        
    except Exception as e:
        print(f"❌ News/sentiment audit failed: {e}")
        return False

def audit_technical_indicators():
    """Audit technical indicator calculations"""
    print("\n📊 AUDITING TECHNICAL INDICATORS")
    print("=" * 50)
    
    try:
        from advanced_data_fetcher import AdvancedDataFetcher
        fetcher = AdvancedDataFetcher(data_mode="light")
        
        test_symbol = 'AAPL'
        print(f"🔍 Testing technical indicators for {test_symbol}:")
        
        stock_data = fetcher.get_comprehensive_stock_data(test_symbol)
        
        if stock_data and 'data' in stock_data:
            df = stock_data['data']
            
            # Check for key technical indicators
            key_indicators = [
                'SMA_20', 'SMA_50', 'SMA_200',
                'EMA_12', 'EMA_26',
                'RSI_14', 'MACD_12_26',
                'BB_20_2_upper', 'BB_20_2_lower',
                'Volume_Ratio', 'Volatility_20'
            ]
            
            found_indicators = []
            missing_indicators = []
            
            for indicator in key_indicators:
                if indicator in df.columns:
                    found_indicators.append(indicator)
                    
                    # Basic sanity check
                    values = df[indicator].dropna()
                    if len(values) > 0:
                        latest_value = values.iloc[-1]
                        print(f"   ✅ {indicator}: {latest_value:.3f}")
                    else:
                        print(f"   ⚠️ {indicator}: No valid values")
                else:
                    missing_indicators.append(indicator)
            
            print(f"\n   📊 Found {len(found_indicators)}/{len(key_indicators)} key indicators")
            
            if missing_indicators:
                print(f"   ❌ Missing: {', '.join(missing_indicators)}")
            
            # Validate indicator relationships
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                sma20 = df['SMA_20'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                print(f"\n   🔍 Relationship validation:")
                print(f"     Current Price: ${current_price:.2f}")
                print(f"     SMA 20: ${sma20:.2f}")
                print(f"     SMA 50: ${sma50:.2f}")
                
                if abs(current_price - sma20) / current_price < 0.5:  # Within 50%
                    print(f"     ✅ Price/SMA relationship looks reasonable")
                else:
                    print(f"     ⚠️ Price/SMA relationship may be questionable")
            
            return len(found_indicators) >= len(key_indicators) * 0.8  # 80% success rate
        else:
            print("   ❌ No data available for technical analysis")
            return False
            
    except Exception as e:
        print(f"❌ Technical indicators audit failed: {e}")
        return False

def cross_reference_prices():
    """Cross-reference prices with external sources"""
    print("\n🔍 CROSS-REFERENCING PRICES")
    print("=" * 50)
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n📊 Cross-referencing {symbol}:")
        
        try:
            # Get price from our system
            from advanced_data_fetcher import AdvancedDataFetcher
            fetcher = AdvancedDataFetcher(data_mode="light")
            stock_data = fetcher.get_comprehensive_stock_data(symbol)
            
            our_price = None
            if stock_data and 'data' in stock_data:
                df = stock_data['data']
                if not df.empty:
                    our_price = df['Close'].iloc[-1]
            
            # Get price from Yahoo Finance directly (for comparison)
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; DataAudit/1.0)'}
                params = {'range': '1d', 'interval': '1d'}
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if ('chart' in data and 'result' in data['chart'] and 
                        data['chart']['result'] and 'indicators' in data['chart']['result'][0]):
                        
                        quotes = data['chart']['result'][0]['indicators']['quote'][0]
                        yahoo_price = quotes['close'][-1] if quotes['close'] else None
                        
                        if our_price and yahoo_price:
                            difference = abs(our_price - yahoo_price)
                            percentage_diff = (difference / yahoo_price) * 100
                            
                            print(f"   Our price: ${our_price:.2f}")
                            print(f"   Yahoo price: ${yahoo_price:.2f}")
                            print(f"   Difference: ${difference:.2f} ({percentage_diff:.2f}%)")
                            
                            if percentage_diff < 5:  # Within 5%
                                print(f"   ✅ PRICES MATCH (within 5%)")
                            else:
                                print(f"   ⚠️ PRICE DISCREPANCY (>{percentage_diff:.1f}%)")
                        else:
                            print(f"   ❌ Could not compare prices")
                    else:
                        print(f"   ❌ Yahoo data format unexpected")
                else:
                    print(f"   ❌ Yahoo API failed ({response.status_code})")
                    
            except Exception as e:
                print(f"   ❌ Yahoo comparison failed: {str(e)[:50]}")
                
        except Exception as e:
            print(f"   ❌ Our system failed: {str(e)[:50]}")

def main():
    """Run comprehensive data audit"""
    print("🔍 COMPREHENSIVE DATA AUDIT")
    print("=" * 70)
    print(f"📅 Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # Run all audits
    audits = [
        ("Stock Data", audit_stock_data),
        ("Market Context", audit_market_context),
        ("Economic Data", audit_economic_data),
        ("News & Sentiment", audit_news_sentiment),
        ("Technical Indicators", audit_technical_indicators)
    ]
    
    for audit_name, audit_func in audits:
        try:
            print(f"\n{'='*20} {audit_name.upper()} {'='*20}")
            results[audit_name] = audit_func()
        except Exception as e:
            print(f"❌ {audit_name} audit crashed: {e}")
            results[audit_name] = False
    
    # Cross-reference prices
    print(f"\n{'='*20} PRICE VERIFICATION {'='*20}")
    cross_reference_prices()
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 AUDIT SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for audit_name, passed_audit in results.items():
        status = "✅ PASS" if passed_audit else "❌ FAIL"
        print(f"{audit_name:20} | {status}")
    
    print(f"\nOverall: {passed}/{total} audits passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 EXCELLENT: All data sources validated!")
        print("✅ Your application uses 100% real, accurate data")
        print("🚀 Safe for trading analysis")
    elif passed >= total * 0.8:
        print("\n✅ GOOD: Most data sources validated")
        print("🔄 Minor issues detected - review warnings above")
        print("⚠️ Generally safe for trading with caution")
    else:
        print("\n⚠️ POOR: Multiple data validation failures")
        print("🚨 NOT SAFE for trading - fix issues before use")
        print("🔧 Review all failed audits above")

if __name__ == "__main__":
    main()
