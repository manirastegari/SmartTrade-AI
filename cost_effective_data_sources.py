#!/usr/bin/env python3
"""
Cost-Effective Data Sources - Accurate data at minimal cost
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from typing import Optional

class CostEffectiveDataManager:
    """Manages the most cost-effective reliable data sources"""
    
    def __init__(self):
        self.sources = {
            'alpha_vantage_free': AlphaVantageFree(),
            'finnhub_free': FinnhubFree(), 
            'iex_cloud_free': IEXCloudFree(),
            'fmp_free': FMPFree(),
            'yahoo_direct': YahooDirectAPI()
        }
        
        # ACTUAL costs (corrected)
        self.monthly_costs = {
            'alpha_vantage_free': 0,      # 500 calls/day FREE
            'finnhub_free': 0,            # 60 calls/minute FREE  
            'iex_cloud_free': 0,          # 100 calls/month FREE
            'fmp_free': 0,                # 250 calls/day FREE
            'yahoo_direct': 0,            # Unlimited FREE (but unstable)
            'alpha_vantage_paid': 25,     # $25/month for 75,000 calls
            'finnhub_paid': 7.99,         # $7.99/month unlimited
            'fmp_paid': 15,               # $15/month for 10,000 calls/day
            'iex_cloud_paid': 9           # $9/month for 500,000 calls
        }
        
        print("💰 COST-EFFECTIVE DATA SOURCES:")
        print("   🆓 Alpha Vantage: FREE (500 calls/day)")
        print("   🆓 Finnhub: FREE (60 calls/minute = 86,400/day)")
        print("   🆓 FMP: FREE (250 calls/day)")
        print("   🆓 IEX Cloud: FREE (100 calls/month)")
        print("   🆓 Yahoo Direct: FREE (unlimited but unstable)")
        print()
        print("💡 RECOMMENDED: Use FREE tiers = $0/month!")
        print("   📊 Total capacity: 1,000+ stocks/day FREE")
    
    def get_stock_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get stock data using free sources first"""
        
        # Try free sources in order of capacity (Yahoo first - unlimited)
        free_sources = ['yahoo_direct', 'finnhub_free', 'alpha_vantage_free', 'fmp_free']
        
        for source_name in free_sources:
            try:
                print(f"🆓 Trying {source_name} for {symbol}...")
                source = self.sources[source_name]
                data = source.get_historical_data(symbol, period)
                
                if data is not None and not data.empty and len(data) > 20:
                    if self._validate_data(data, symbol):
                        print(f"✅ {source_name} SUCCESS: {len(data)} days for {symbol} (FREE)")
                        return data
                    else:
                        print(f"⚠️ {source_name} data validation failed for {symbol}")
                else:
                    print(f"❌ {source_name} returned no data for {symbol}")
                    
            except Exception as e:
                print(f"❌ {source_name} error for {symbol}: {str(e)[:50]}")
                continue
        
        print(f"🚨 ALL FREE SOURCES FAILED for {symbol}")
        return None
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality"""
        try:
            # Basic validation
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                return False
            
            close_prices = df['Close'].dropna()
            if len(close_prices) < 20:
                return False
            
            # Price reasonableness
            min_price = close_prices.min()
            max_price = close_prices.max()
            if min_price <= 0 or max_price > 50000:
                return False
            
            # Volume check
            volumes = df['Volume'].dropna()
            if len(volumes) < 20 or volumes.max() <= 0:
                return False
            
            return True
            
        except Exception:
            return False

class AlphaVantageFree:
    """Alpha Vantage Free Tier - 500 calls/day"""
    
    def __init__(self):
        # Get free API key from: https://www.alphavantage.co/support/#api-key
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = "https://www.alphavantage.co/query"
        self.last_call = 0
        self.rate_limit = 12  # 5 calls per minute
        self.daily_calls = 0
        self.daily_limit = 500
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get data from Alpha Vantage free tier"""
        
        if self.daily_calls >= self.daily_limit:
            print(f"   ⚠️ Alpha Vantage daily limit reached ({self.daily_limit})")
            return None
        
        # Rate limiting
        time_since_last = time.time() - self.last_call
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            self.last_call = time.time()
            self.daily_calls += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    time_series = data['Time Series (Daily)']
                    
                    df_data = []
                    for date_str, values in time_series.items():
                        df_data.append({
                            'Date': pd.to_datetime(date_str),
                            'Open': float(values['1. open']),
                            'High': float(values['2. high']),
                            'Low': float(values['3. low']),
                            'Close': float(values['4. close']),
                            'Volume': int(values['6. volume'])
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        return df
                        
        except Exception:
            pass
        
        return None

class FinnhubFree:
    """Finnhub Free Tier - 60 calls/minute (86,400/day)"""
    
    def __init__(self):
        # Get free API key from: https://finnhub.io/register
        self.api_key = os.getenv('FINNHUB_API_KEY', 'demo')
        self.base_url = "https://finnhub.io/api/v1"
        self.last_call = 0
        self.rate_limit = 1  # 60 per minute = 1 per second
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get data from Finnhub free tier"""
        
        # Rate limiting
        time_since_last = time.time() - self.last_call
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            if period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            else:  # 2y
                start_date = end_date - timedelta(days=730)
            
            url = f"{self.base_url}/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': 'D',
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp()),
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            self.last_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('s') == 'ok' and 't' in data:
                    df_data = []
                    for i in range(len(data['t'])):
                        df_data.append({
                            'Date': pd.to_datetime(data['t'][i], unit='s'),
                            'Open': float(data['o'][i]),
                            'High': float(data['h'][i]),
                            'Low': float(data['l'][i]),
                            'Close': float(data['c'][i]),
                            'Volume': int(data['v'][i])
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        return df
                        
        except Exception:
            pass
        
        return None

class FMPFree:
    """Financial Modeling Prep Free Tier - 250 calls/day"""
    
    def __init__(self):
        # Get free API key from: https://financialmodelingprep.com/developer/docs
        self.api_key = os.getenv('FMP_API_KEY', 'demo')
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.last_call = 0
        self.rate_limit = 0.2  # 5 calls per second
        self.daily_calls = 0
        self.daily_limit = 250
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get data from FMP free tier"""
        
        if self.daily_calls >= self.daily_limit:
            print(f"   ⚠️ FMP daily limit reached ({self.daily_limit})")
            return None
        
        # Rate limiting
        time_since_last = time.time() - self.last_call
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        
        try:
            url = f"{self.base_url}/historical-price-full/{symbol}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            self.last_call = time.time()
            self.daily_calls += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if 'historical' in data:
                    historical = data['historical']
                    
                    df_data = []
                    for day in historical:
                        df_data.append({
                            'Date': pd.to_datetime(day['date']),
                            'Open': float(day['open']),
                            'High': float(day['high']),
                            'Low': float(day['low']),
                            'Close': float(day['close']),
                            'Volume': int(day['volume'])
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Filter by period
                        if period == "1y":
                            cutoff = datetime.now() - timedelta(days=365)
                            df = df[df.index >= cutoff]
                        elif period == "6mo":
                            cutoff = datetime.now() - timedelta(days=180)
                            df = df[df.index >= cutoff]
                        
                        return df
                        
        except Exception:
            pass
        
        return None

class IEXCloudFree:
    """IEX Cloud Free Tier - 100 calls/month"""
    
    def __init__(self):
        # Get free API key from: https://iexcloud.io/
        self.api_key = os.getenv('IEX_CLOUD_API_KEY', 'pk_test_demo')
        self.base_url = "https://cloud.iexapis.com/stable"
        self.monthly_calls = 0
        self.monthly_limit = 100
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get data from IEX Cloud free tier"""
        
        if self.monthly_calls >= self.monthly_limit:
            print(f"   ⚠️ IEX Cloud monthly limit reached ({self.monthly_limit})")
            return None
        
        try:
            # Map period to IEX range
            range_map = {
                "1mo": "1m",
                "3mo": "3m", 
                "6mo": "6m",
                "1y": "1y",
                "2y": "2y"
            }
            
            iex_range = range_map.get(period, "1y")  # Use 1y for free tier
            
            url = f"{self.base_url}/stock/{symbol}/chart/{iex_range}"
            params = {'token': self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            self.monthly_calls += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if data and isinstance(data, list):
                    df_data = []
                    for day in data:
                        if all(key in day for key in ['date', 'open', 'high', 'low', 'close', 'volume']):
                            df_data.append({
                                'Date': pd.to_datetime(day['date']),
                                'Open': float(day['open']),
                                'High': float(day['high']),
                                'Low': float(day['low']),
                                'Close': float(day['close']),
                                'Volume': int(day['volume'])
                            })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        return df
                        
        except Exception:
            pass
        
        return None

class YahooDirectAPI:
    """Yahoo Direct API - Unlimited but unstable"""
    
    def __init__(self):
        self.last_call = 0
        self.rate_limit = 0.5  # Be gentle with Yahoo
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get data from Yahoo Direct API"""
        
        # Rate limiting
        time_since_last = time.time() - self.last_call
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            params = {
                'range': period,
                'interval': '1d'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            self.last_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if ('chart' in data and 'result' in data['chart'] and 
                    data['chart']['result'] and len(data['chart']['result']) > 0):
                    
                    result = data['chart']['result'][0]
                    
                    if ('timestamp' in result and 'indicators' in result and 
                        'quote' in result['indicators'] and len(result['indicators']['quote']) > 0):
                        
                        timestamps = result['timestamp']
                        quotes = result['indicators']['quote'][0]
                        
                        df_data = []
                        for i, ts in enumerate(timestamps):
                            try:
                                if (quotes['open'][i] is not None and 
                                    quotes['high'][i] is not None and
                                    quotes['low'][i] is not None and
                                    quotes['close'][i] is not None):
                                    
                                    df_data.append({
                                        'Date': pd.to_datetime(ts, unit='s'),
                                        'Open': float(quotes['open'][i]),
                                        'High': float(quotes['high'][i]),
                                        'Low': float(quotes['low'][i]),
                                        'Close': float(quotes['close'][i]),
                                        'Volume': int(quotes['volume'][i]) if quotes['volume'][i] is not None else 0
                                    })
                            except (IndexError, TypeError, ValueError):
                                continue
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            df.set_index('Date', inplace=True)
                            df.sort_index(inplace=True)
                            df = df.dropna(how='all')
                            return df
                            
        except Exception:
            pass
        
        return None

if __name__ == "__main__":
    # Test cost-effective sources
    manager = CostEffectiveDataManager()
    
    test_symbols = ['AAPL', 'MSFT']
    
    print(f"\n🧪 TESTING COST-EFFECTIVE SOURCES")
    print("=" * 50)
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        data = manager.get_stock_data(symbol, "1mo")
        
        if data is not None:
            print(f"✅ SUCCESS: {len(data)} days - ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ FAILED: No data retrieved")
    
    print(f"\n💰 TOTAL COST: $0/month (using free tiers only!)")
