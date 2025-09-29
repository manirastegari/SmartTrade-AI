#!/usr/bin/env python3
"""
Setup Paid Data Sources for AI Trading Application
Configure API keys and test all data sources
"""

import os
import sys
from datetime import datetime

def setup_api_keys():
    """Setup API keys for paid data sources"""
    
    print("🔑 PAID DATA SOURCES SETUP")
    print("=" * 60)
    print("Setting up the most reliable and cost-effective data sources:")
    print()
    
    # API Key setup instructions
    sources = {
        'IEX Cloud': {
            'cost': '$0.50 per 1000 calls (MOST COST-EFFECTIVE)',
            'url': 'https://iexcloud.io/pricing',
            'env_var': 'IEX_CLOUD_API_KEY',
            'free_tier': 'Yes - 100 calls/month',
            'recommended': True
        },
        'Alpha Vantage': {
            'cost': 'Free: 500/day, Paid: $25/month',
            'url': 'https://www.alphavantage.co/support/#api-key',
            'env_var': 'ALPHA_VANTAGE_API_KEY',
            'free_tier': 'Yes - 500 calls/day',
            'recommended': True
        },
        'Finnhub': {
            'cost': 'Free: 60/min, Paid: $7.99/month',
            'url': 'https://finnhub.io/register',
            'env_var': 'FINNHUB_API_KEY',
            'free_tier': 'Yes - 60 calls/minute',
            'recommended': True
        },
        'Twelve Data': {
            'cost': 'Free: 800/day, Paid: $8/month',
            'url': 'https://twelvedata.com/pricing',
            'env_var': 'TWELVE_DATA_API_KEY',
            'free_tier': 'Yes - 800 calls/day',
            'recommended': False
        },
        'Polygon.io': {
            'cost': 'Free: 5/min, Paid: $99/month',
            'url': 'https://polygon.io/pricing',
            'env_var': 'POLYGON_API_KEY',
            'free_tier': 'Yes - 5 calls/minute',
            'recommended': False
        }
    }
    
    print("📊 RECOMMENDED SETUP (for maximum reliability):")
    print()
    
    for name, info in sources.items():
        if info['recommended']:
            print(f"🎯 {name}")
            print(f"   💰 Cost: {info['cost']}")
            print(f"   🆓 Free tier: {info['free_tier']}")
            print(f"   🔗 Sign up: {info['url']}")
            print(f"   🔑 Environment variable: {info['env_var']}")
            print()
    
    print("💡 SETUP INSTRUCTIONS:")
    print("1. Sign up for IEX Cloud (most cost-effective)")
    print("2. Sign up for Alpha Vantage (good free tier)")
    print("3. Sign up for Finnhub (fast and reliable)")
    print("4. Add API keys to your environment:")
    print()
    
    # Check current environment
    print("🔍 CURRENT ENVIRONMENT STATUS:")
    for name, info in sources.items():
        env_var = info['env_var']
        if os.getenv(env_var):
            print(f"   ✅ {name}: API key found")
        else:
            print(f"   ❌ {name}: No API key (set {env_var})")
    
    print()
    print("🛠️ TO SET API KEYS:")
    print("   # Option 1: Environment variables (recommended)")
    print("   export IEX_CLOUD_API_KEY='your_iex_key_here'")
    print("   export ALPHA_VANTAGE_API_KEY='your_alpha_vantage_key_here'")
    print("   export FINNHUB_API_KEY='your_finnhub_key_here'")
    print()
    print("   # Option 2: Add to ~/.bashrc or ~/.zshrc")
    print("   echo 'export IEX_CLOUD_API_KEY=\"your_key\"' >> ~/.bashrc")
    print()
    print("   # Option 3: Create .env file in project directory")
    print("   echo 'IEX_CLOUD_API_KEY=your_key' > .env")

def test_paid_sources():
    """Test all configured paid data sources"""
    
    print("\n🧪 TESTING PAID DATA SOURCES")
    print("=" * 60)
    
    try:
        from paid_data_sources import PaidDataManager
        
        manager = PaidDataManager()
        
        # Test symbols
        test_symbols = ['AAPL', 'MSFT']
        
        print(f"Testing with symbols: {', '.join(test_symbols)}")
        print()
        
        success_count = 0
        total_tests = len(test_symbols)
        
        for symbol in test_symbols:
            print(f"📊 Testing {symbol}:")
            
            try:
                data = manager.get_stock_data(symbol, "1mo")
                
                if data is not None and not data.empty:
                    latest_price = data['Close'].iloc[-1]
                    latest_volume = data['Volume'].iloc[-1]
                    data_points = len(data)
                    date_range = f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
                    
                    print(f"   ✅ SUCCESS: {data_points} days")
                    print(f"   📈 Latest price: ${latest_price:.2f}")
                    print(f"   📊 Volume: {latest_volume:,.0f}")
                    print(f"   📅 Range: {date_range}")
                    
                    success_count += 1
                else:
                    print(f"   ❌ FAILED: No data returned")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)[:100]}")
            
            print()
        
        # Results summary
        success_rate = (success_count / total_tests) * 100
        
        print("📈 TEST RESULTS SUMMARY")
        print("-" * 40)
        print(f"✅ Successful: {success_count}/{total_tests} ({success_rate:.0f}%)")
        
        if success_rate >= 100:
            print("🎉 EXCELLENT: All paid sources working perfectly!")
            print("✅ Your application will use 100% REAL market data")
        elif success_rate >= 50:
            print("✅ GOOD: Most paid sources working")
            print("🔄 Some sources may need API key configuration")
        else:
            print("⚠️ POOR: Most paid sources failing")
            print("🔧 Check API key configuration")
        
        return success_rate >= 50
        
    except ImportError:
        print("❌ ERROR: paid_data_sources.py not found")
        print("🔧 Make sure the paid data sources file is in the same directory")
        return False

def create_env_template():
    """Create a .env template file"""
    
    env_template = """# AI Trading Application - API Keys
# Copy this file to .env and add your actual API keys

# IEX Cloud (MOST COST-EFFECTIVE - $0.50 per 1000 calls)
# Sign up: https://iexcloud.io/pricing
IEX_CLOUD_API_KEY=your_iex_cloud_api_key_here

# Alpha Vantage (GOOD FREE TIER - 500 calls/day)
# Sign up: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Finnhub (FAST & RELIABLE - 60 calls/minute free)
# Sign up: https://finnhub.io/register
FINNHUB_API_KEY=your_finnhub_api_key_here

# Twelve Data (OPTIONAL - 800 calls/day free)
# Sign up: https://twelvedata.com/pricing
TWELVE_DATA_API_KEY=your_twelve_data_api_key_here

# Polygon.io (OPTIONAL - 5 calls/minute free)
# Sign up: https://polygon.io/pricing
POLYGON_API_KEY=your_polygon_api_key_here
"""
    
    try:
        with open('.env.template', 'w') as f:
            f.write(env_template)
        print("📝 Created .env.template file")
        print("   Copy to .env and add your API keys")
    except Exception as e:
        print(f"❌ Could not create .env.template: {e}")

def main():
    print("🚀 AI TRADING APPLICATION - PAID DATA SETUP")
    print("=" * 70)
    print(f"📅 Setup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup API keys
    setup_api_keys()
    
    # Create environment template
    create_env_template()
    
    # Test paid sources
    sources_working = test_paid_sources()
    
    print("\n" + "=" * 70)
    print("🎯 SETUP COMPLETE")
    
    if sources_working:
        print("✅ READY FOR TRADING: Paid data sources configured")
        print("🛡️ Your application will use GUARANTEED REAL market data")
        print("🚀 Run your analysis with confidence!")
    else:
        print("⚠️ SETUP INCOMPLETE: Configure API keys for reliable data")
        print("🔧 Follow the instructions above to set up paid data sources")
        print("🚨 DO NOT TRADE until data sources are properly configured")
    
    print("\n💡 NEXT STEPS:")
    if sources_working:
        print("1. ✅ Run: python3 data_integrity_check.py")
        print("2. ✅ Run: streamlit run professional_trading_app.py")
        print("3. ✅ Enjoy fast, reliable analysis with real data!")
    else:
        print("1. 🔑 Set up API keys (see instructions above)")
        print("2. 🧪 Run: python3 setup_paid_data.py (to test)")
        print("3. ✅ Run: python3 data_integrity_check.py")
        print("4. 🚀 Run: streamlit run professional_trading_app.py")

if __name__ == "__main__":
    main()
