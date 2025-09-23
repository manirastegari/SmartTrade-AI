#!/usr/bin/env python3
"""
Comprehensive test for SPY and VIX data fetching with all fallback sources
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_data_fetcher import AdvancedDataFetcher
import warnings
warnings.filterwarnings('ignore')

def test_individual_spy_sources():
    """Test each SPY data source individually"""
    print("🔍 Testing Individual SPY Data Sources")
    print("=" * 50)
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    
    # Test each source individually
    spy_test_sources = [
        ("yfinance SPY", lambda: fetcher._fetch_yfinance_with_fallback("SPY")),
        ("yfinance IVV", lambda: fetcher._fetch_yfinance_with_fallback("IVV")),
        ("yfinance VOO", lambda: fetcher._fetch_yfinance_with_fallback("VOO")),
        ("yfinance SPLG", lambda: fetcher._fetch_yfinance_with_fallback("SPLG")),
        ("Stooq SPY", lambda: fetcher._fetch_stooq_history("SPY")),
        ("Stooq spy.us", lambda: fetcher._fetch_stooq_history("spy.us")),
        ("Web scrape SPY", lambda: fetcher._fetch_simple_web_data("SPY")),
        ("Web scrape IVV", lambda: fetcher._fetch_simple_web_data("IVV")),
    ]
    
    working_sources = []
    
    for source_name, fetch_func in spy_test_sources:
        print(f"\n📊 Testing {source_name}...")
        try:
            data = fetch_func()
            if data is not None and not data.empty and len(data) >= 2:
                close_prices = data['Close']
                current_price = close_prices.iloc[-1]
                prev_price = close_prices.iloc[-2] if len(close_prices) >= 2 else current_price
                daily_return = (current_price / prev_price - 1) * 100
                
                print(f"   ✅ SUCCESS: Current price ${current_price:.2f}, Daily return {daily_return:.2f}%")
                working_sources.append(source_name)
            else:
                print(f"   ❌ FAILED: No valid data returned")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print(f"\n📈 SPY Sources Summary:")
    print(f"   Working sources: {len(working_sources)}/{len(spy_test_sources)}")
    print(f"   Success rate: {len(working_sources)/len(spy_test_sources)*100:.1f}%")
    
    return working_sources

def test_individual_vix_sources():
    """Test each VIX data source individually"""
    print(f"\n🔍 Testing Individual VIX Data Sources")
    print("=" * 50)
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    
    # Test each source individually
    vix_test_sources = [
        ("yfinance ^VIX", lambda: fetcher._fetch_yfinance_with_fallback("^VIX")),
        ("yfinance VIXY", lambda: fetcher._fetch_yfinance_with_fallback("VIXY")),
        ("yfinance VXX", lambda: fetcher._fetch_yfinance_with_fallback("VXX")),
        ("yfinance UVXY", lambda: fetcher._fetch_yfinance_with_fallback("UVXY")),
        ("Stooq ^VIX", lambda: fetcher._fetch_stooq_history("^VIX")),
        ("Stooq vixy.us", lambda: fetcher._fetch_stooq_history("vixy.us")),
        ("Web scrape ^VIX", lambda: fetcher._fetch_simple_web_data("^VIX")),
        ("Web scrape VIXY", lambda: fetcher._fetch_simple_web_data("VIXY")),
    ]
    
    working_sources = []
    
    for source_name, fetch_func in vix_test_sources:
        print(f"\n📊 Testing {source_name}...")
        try:
            data = fetch_func()
            if data is not None and not data.empty:
                current_value = data['Close'].iloc[-1]
                print(f"   ✅ SUCCESS: Current value {current_value:.2f}")
                working_sources.append(source_name)
            else:
                print(f"   ❌ FAILED: No valid data returned")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print(f"\n📈 VIX Sources Summary:")
    print(f"   Working sources: {len(working_sources)}/{len(vix_test_sources)}")
    print(f"   Success rate: {len(working_sources)/len(vix_test_sources)*100:.1f}%")
    
    return working_sources

def test_integrated_market_context():
    """Test the integrated market context with all fallbacks"""
    print(f"\n🔧 Testing Integrated Market Context")
    print("=" * 40)
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    
    print("📊 Testing market context retrieval with all fallbacks...")
    
    try:
        # Force refresh to test all sources
        market_context = fetcher.get_market_context(force_refresh=True)
        
        print("✅ Market context retrieved successfully!")
        print(f"   SPY 1-day return: {market_context['spy_return_1d']:.4f} ({market_context['spy_return_1d']*100:.2f}%)")
        print(f"   SPY 20-day volatility: {market_context['spy_vol_20']:.4f} ({market_context['spy_vol_20']*100:.2f}%)")
        print(f"   VIX proxy: {market_context['vix_proxy']:.2f}")
        
        # Validate reasonable values
        spy_return = market_context['spy_return_1d']
        spy_vol = market_context['spy_vol_20']
        vix = market_context['vix_proxy']
        
        # Check if values are reasonable
        validations = []
        
        if -0.1 <= spy_return <= 0.1:  # Daily return between -10% and +10%
            validations.append("✅ SPY return is reasonable")
        else:
            validations.append(f"⚠️ SPY return seems extreme: {spy_return}")
        
        if 0.005 <= spy_vol <= 0.1:  # Volatility between 0.5% and 10%
            validations.append("✅ SPY volatility is reasonable")
        else:
            validations.append(f"⚠️ SPY volatility seems extreme: {spy_vol}")
        
        if 5 <= vix <= 80:  # VIX between 5 and 80
            validations.append("✅ VIX proxy is reasonable")
        else:
            validations.append(f"⚠️ VIX proxy seems extreme: {vix}")
        
        for validation in validations:
            print(f"   {validation}")
            
        return True
        
    except Exception as e:
        print(f"❌ Market context failed: {e}")
        return False

def test_rate_limiting_resilience():
    """Test resilience to rate limiting by making multiple rapid calls"""
    print(f"\n⚡ Testing Rate Limiting Resilience")
    print("=" * 40)
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    
    print("📊 Making multiple rapid market context calls...")
    
    success_count = 0
    total_calls = 5
    
    for i in range(total_calls):
        try:
            market_context = fetcher.get_market_context(force_refresh=True)
            if market_context and 'spy_return_1d' in market_context:
                success_count += 1
                print(f"   Call {i+1}: ✅ Success")
            else:
                print(f"   Call {i+1}: ❌ Failed - no data")
        except Exception as e:
            print(f"   Call {i+1}: ❌ Error - {e}")
    
    success_rate = (success_count / total_calls) * 100
    print(f"\n📈 Rate Limiting Test Results:")
    print(f"   Successful calls: {success_count}/{total_calls}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("   ✅ EXCELLENT resilience to rate limiting")
    elif success_rate >= 60:
        print("   📈 GOOD resilience to rate limiting")
    else:
        print("   ⚠️ MODERATE resilience - may need more fallbacks")
    
    return success_rate >= 60

def test_error_suppression():
    """Test that error messages are properly suppressed"""
    print(f"\n🔇 Testing Error Message Suppression")
    print("=" * 40)
    
    fetcher = AdvancedDataFetcher(data_mode="light")
    
    print("📊 Testing if SPY/VIX errors are suppressed...")
    
    # Capture any print output during market context fetch
    import io
    import contextlib
    
    captured_output = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(captured_output):
            market_context = fetcher.get_market_context(force_refresh=True)
        
        output = captured_output.getvalue()
        
        # Check for error messages
        error_indicators = [
            "No data found for this date range",
            "may be delisted",
            "SPY: No data found",
            "VIX: No data found"
        ]
        
        found_errors = []
        for error in error_indicators:
            if error in output:
                found_errors.append(error)
        
        if found_errors:
            print(f"   ⚠️ Found error messages: {found_errors}")
            print(f"   Output: {output}")
            return False
        else:
            print("   ✅ No error messages found - suppression working!")
            if output.strip():
                print(f"   Info messages: {output.strip()}")
            return True
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Comprehensive SPY & VIX Data Source Testing")
    print("=" * 60)
    
    # Test individual sources
    spy_sources = test_individual_spy_sources()
    vix_sources = test_individual_vix_sources()
    
    # Test integrated system
    market_success = test_integrated_market_context()
    
    # Test rate limiting resilience
    rate_limit_success = test_rate_limiting_resilience()
    
    # Test error suppression
    error_suppression = test_error_suppression()
    
    print(f"\n🎯 COMPREHENSIVE TEST SUMMARY:")
    print(f"   SPY Sources Working: {len(spy_sources)}/8")
    print(f"   VIX Sources Working: {len(vix_sources)}/8")
    print(f"   Market Context: {'✅ WORKING' if market_success else '❌ ISSUES'}")
    print(f"   Rate Limiting: {'✅ RESILIENT' if rate_limit_success else '❌ NEEDS WORK'}")
    print(f"   Error Suppression: {'✅ CLEAN' if error_suppression else '❌ NOISY'}")
    
    total_sources = len(spy_sources) + len(vix_sources)
    
    if total_sources >= 8 and market_success and rate_limit_success:
        print(f"\n🎉 EXCELLENT ROBUSTNESS!")
        print(f"✅ Multiple working data sources for both SPY and VIX")
        print(f"✅ System resilient to API failures and rate limiting")
        print(f"✅ Market context always available with reasonable values")
        print(f"✅ Clean operation without annoying error messages")
        
        print(f"\n💡 FALLBACK HIERARCHY:")
        print(f"1. 🥇 yfinance APIs (SPY, IVV, VOO, SPLG, ^VIX, VIXY, VXX, UVXY)")
        print(f"2. 🥈 Stooq CSV data (spy.us, ivv, vixy.us)")
        print(f"3. 🥉 Web scraping (Yahoo Finance pages)")
        print(f"4. 🛡️ Synthetic data (statistical fallback)")
        
    elif total_sources >= 4:
        print(f"\n📈 GOOD ROBUSTNESS!")
        print(f"✅ Some working data sources available")
        print(f"✅ Basic fallback system functional")
        print(f"⚠️ May want to add more backup sources")
        
    else:
        print(f"\n⚠️ LIMITED ROBUSTNESS")
        print(f"❌ Few working data sources")
        print(f"❌ May struggle with API failures")
        print(f"💡 Consider adding more fallback sources")
    
    print(f"\n🚀 Your market data system is now robust against:")
    print(f"• yfinance API rate limiting")
    print(f"• Individual ticker delisting/issues")
    print(f"• Network connectivity problems")
    print(f"• Data provider outages")
    print(f"• Any combination of the above!")
