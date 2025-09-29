#!/usr/bin/env python3
"""
Paid Data Sources for AI Trading Application
Reliable, accurate, and cost-effective data providers
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any

class PaidDataManager:
    """Manages multiple paid data sources for maximum reliability"""
    
    def __init__(self):
        self.sources = {
            'alpha_vantage': AlphaVantageProvider(),
            'polygon': PolygonProvider(),
            'iex_cloud': IEXCloudProvider(),
            'finnhub': FinnhubProvider(),
            'twelve_data': TwelveDataProvider()
        }
        
        # Cost per 1000 API calls (approximate)
        self.costs = {
            'alpha_vantage': 0.00,  # Free tier: 5/min, 500/day, then $25/month
            'polygon': 0.00,        # Free tier: 5/min, then $99/month  
            'iex_cloud': 0.0005,    # $0.50 per 1000 calls
            'finnhub': 0.00,        # Free tier: 60/min, then $7.99/month
            'twelve_data': 0.00     # Free tier: 800/day, then $8/month
        }
        
        print("💰 Paid Data Sources Initialized:")
        print("   📊 Alpha Vantage: Free tier (500/day) + $25/month")
        print("   📈 Polygon.io: Free tier (5/min) + $99/month") 
        print("   💎 IEX Cloud: $0.50 per 1000 calls (most cost-effective)")
        print("   🔥 Finnhub: Free tier (60/min) + $7.99/month")
        print("   ⚡ Twelve Data: Free tier (800/day) + $8/month")
    
    def get_stock_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get stock data from the most reliable source"""
        
        # Try sources in order of reliability and cost-effectiveness
        source_order = ['iex_cloud', 'alpha_vantage', 'finnhub', 'twelve_data', 'polygon']
        
        for source_name in source_order:
            try:
                print(f"📡 Trying {source_name} for {symbol}...")
                source = self.sources[source_name]
                data = source.get_historical_data(symbol, period)
                
                if data is not None and not data.empty and len(data) > 20:
                    if self._validate_data(data, symbol):
                        print(f"✅ {source_name} SUCCESS: {len(data)} days for {symbol}")
                        return data
                    else:
                        print(f"⚠️ {source_name} data validation failed for {symbol}")
                else:
                    print(f"❌ {source_name} returned no data for {symbol}")
                    
            except Exception as e:
                print(f"❌ {source_name} error for {symbol}: {str(e)[:50]}")
                continue
        
        print(f"🚨 ALL PAID SOURCES FAILED for {symbol}")
        return None
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Comprehensive data validation"""
        try:
            # Basic structure validation
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Data quality validation
            close_prices = df['Close'].dropna()
            if len(close_prices) < 20:
                return False
            
            # Price reasonableness
            min_price = close_prices.min()
            max_price = close_prices.max()
            if min_price <= 0 or max_price > 50000 or min_price > max_price:
                return False
            
            # Volume validation
            volumes = df['Volume'].dropna()
            if len(volumes) < 20 or volumes.max() <= 0:
                return False
            
            # OHLC relationship validation
            for idx in df.index[-5:]:
                row = df.loc[idx]
                if pd.isna(row['High']) or pd.isna(row['Low']) or pd.isna(row['Open']) or pd.isna(row['Close']):
                    continue
                if not (row['Low'] <= row['Open'] <= row['High'] and 
                       row['Low'] <= row['Close'] <= row['High']):
                    return False
            
            # Recency validation (data should be recent)
            latest_date = df.index[-1]
            days_old = (datetime.now() - latest_date).days
            if days_old > 7:  # Data older than 1 week is suspicious
                print(f"⚠️ Data for {symbol} is {days_old} days old")
            
            return True
            
        except Exception:
            return False

class AlphaVantageProvider:
    """Alpha Vantage - Free tier: 5/min, 500/day"""
    
    def __init__(self):
        # Get API key from environment or use demo
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = "https://www.alphavantage.co/query"
        self.last_call = 0
        self.rate_limit = 12  # 5 calls per minute = 12 seconds between calls
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage"""
        try:
            # Rate limiting
            time_since_last = time.time() - self.last_call
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
            
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            self.last_call = time.time()
            
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

class IEXCloudProvider:
    """IEX Cloud - Most cost-effective: $0.50 per 1000 calls"""
    
    def __init__(self):
        # Get API key from environment
        self.api_key = os.getenv('IEX_CLOUD_API_KEY', 'pk_test_demo')  # Demo key
        self.base_url = "https://cloud.iexapis.com/stable"
        self.last_call = 0
        self.rate_limit = 0.1  # Very generous rate limits
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get historical data from IEX Cloud"""
        try:
            # Rate limiting
            time_since_last = time.time() - self.last_call
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
            
            # Map period to IEX range
            range_map = {
                "1mo": "1m",
                "3mo": "3m", 
                "6mo": "6m",
                "1y": "1y",
                "2y": "2y",
                "5y": "5y"
            }
            
            iex_range = range_map.get(period, "2y")
            
            url = f"{self.base_url}/stock/{symbol}/chart/{iex_range}"
            params = {'token': self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            self.last_call = time.time()
            
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

class FinnhubProvider:
    """Finnhub - Free tier: 60/min"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY', 'demo')
        self.base_url = "https://finnhub.io/api/v1"
        self.last_call = 0
        self.rate_limit = 1  # 60 per minute = 1 second between calls
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get historical data from Finnhub"""
        try:
            # Rate limiting
            time_since_last = time.time() - self.last_call
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
            
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

class PolygonProvider:
    """Polygon.io - Free tier: 5/min"""
    
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY', 'demo')
        self.base_url = "https://api.polygon.io"
        self.last_call = 0
        self.rate_limit = 12  # 5 per minute = 12 seconds between calls
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get historical data from Polygon"""
        try:
            # Rate limiting
            time_since_last = time.time() - self.last_call
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
            
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
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            self.last_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'OK' and 'results' in data:
                    df_data = []
                    for result in data['results']:
                        df_data.append({
                            'Date': pd.to_datetime(result['t'], unit='ms'),
                            'Open': float(result['o']),
                            'High': float(result['h']),
                            'Low': float(result['l']),
                            'Close': float(result['c']),
                            'Volume': int(result['v'])
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        return df
                        
        except Exception:
            pass
        
        return None

class TwelveDataProvider:
    """Twelve Data - Free tier: 800/day"""
    
    def __init__(self):
        self.api_key = os.getenv('TWELVE_DATA_API_KEY', 'demo')
        self.base_url = "https://api.twelvedata.com"
        self.last_call = 0
        self.rate_limit = 8  # Conservative rate limiting
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get historical data from Twelve Data"""
        try:
            # Rate limiting
            time_since_last = time.time() - self.last_call
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
            
            url = f"{self.base_url}/time_series"
            params = {
                'symbol': symbol,
                'interval': '1day',
                'outputsize': '5000',
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            self.last_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data and data['values']:
                    df_data = []
                    for value in data['values']:
                        df_data.append({
                            'Date': pd.to_datetime(value['datetime']),
                            'Open': float(value['open']),
                            'High': float(value['high']),
                            'Low': float(value['low']),
                            'Close': float(value['close']),
                            'Volume': int(value['volume'])
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

if __name__ == "__main__":
    # Test the paid data sources
    manager = PaidDataManager()
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n🧪 Testing {symbol}:")
        data = manager.get_stock_data(symbol, "1mo")
        
        if data is not None:
            print(f"✅ SUCCESS: {len(data)} days")
            print(f"📈 Latest price: ${data['Close'].iloc[-1]:.2f}")
            print(f"📊 Latest volume: {data['Volume'].iloc[-1]:,.0f}")
        else:
            print(f"❌ FAILED: No data retrieved")
