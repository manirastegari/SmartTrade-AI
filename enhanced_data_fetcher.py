"""
Enhanced Data Fetcher - Maximum Free Analysis Power
Fetches comprehensive data from all possible free sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import time
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataFetcher:
    """Enhanced data fetcher with maximum free analysis capabilities"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def get_comprehensive_stock_data(self, symbol):
        """Get comprehensive data from multiple free sources"""
        try:
            # Primary data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y", interval="1d")
            info = ticker.info
            
            if hist.empty:
                return None
            
            # Add comprehensive technical indicators
            hist = self._add_advanced_technical_indicators(hist)
            
            # Get additional data
            news_data = self._get_news_sentiment(symbol)
            insider_data = self._get_insider_trading(symbol)
            options_data = self._get_options_data(symbol)
            institutional_data = self._get_institutional_holdings(symbol)
            earnings_data = self._get_earnings_data(symbol)
            economic_data = self._get_economic_indicators()
            
            return {
                'symbol': symbol,
                'data': hist,
                'info': info,
                'news': news_data,
                'insider': insider_data,
                'options': options_data,
                'institutional': institutional_data,
                'earnings': earnings_data,
                'economic': economic_data,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching comprehensive data for {symbol}: {e}")
            return None
    
    def _add_advanced_technical_indicators(self, df):
        """Add 50+ advanced technical indicators"""
        try:
            # Price-based indicators
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_100'] = df['Close'].rolling(window=100).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_5'] = df['Close'].ewm(span=5).mean()
            df['EMA_10'] = df['Close'].ewm(span=10).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_21'] = df['Close'].ewm(span=21).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            df['EMA_100'] = df['Close'].ewm(span=100).mean()
            df['EMA_200'] = df['Close'].ewm(span=200).mean()
            
            # RSI variations
            df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
            df['RSI_21'] = self._calculate_rsi(df['Close'], 21)
            df['RSI_30'] = self._calculate_rsi(df['Close'], 30)
            
            # MACD variations
            df['MACD_12_26'] = df['EMA_12'] - df['EMA_26']
            df['MACD_5_35'] = df['EMA_5'] - df['Close'].ewm(span=35).mean()
            df['MACD_signal_12_26'] = df['MACD_12_26'].ewm(span=9).mean()
            df['MACD_histogram'] = df['MACD_12_26'] - df['MACD_signal_12_26']
            
            # Bollinger Bands variations
            bb_20_2 = self._calculate_bollinger_bands(df['Close'], 20, 2)
            df['BB_20_2_upper'] = bb_20_2['upper']
            df['BB_20_2_middle'] = bb_20_2['middle']
            df['BB_20_2_lower'] = bb_20_2['lower']
            
            bb_20_1 = self._calculate_bollinger_bands(df['Close'], 20, 1)
            df['BB_20_1_upper'] = bb_20_1['upper']
            df['BB_20_1_middle'] = bb_20_1['middle']
            df['BB_20_1_lower'] = bb_20_1['lower']
            
            bb_50_2 = self._calculate_bollinger_bands(df['Close'], 50, 2)
            df['BB_50_2_upper'] = bb_50_2['upper']
            df['BB_50_2_middle'] = bb_50_2['middle']
            df['BB_50_2_lower'] = bb_50_2['lower']
            
            # Stochastic Oscillator
            df['Stoch_K'] = self._calculate_stochastic(df['High'], df['Low'], df['Close'], 14)
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
            
            # Williams %R
            df['Williams_R'] = self._calculate_williams_r(df['High'], df['Low'], df['Close'], 14)
            
            # Commodity Channel Index
            df['CCI'] = self._calculate_cci(df['High'], df['Low'], df['Close'], 20)
            
            # Average True Range
            df['ATR'] = self._calculate_atr(df['High'], df['Low'], df['Close'], 14)
            
            # Average Directional Index
            df['ADX'] = self._calculate_adx(df['High'], df['Low'], df['Close'], 14)
            
            # Money Flow Index
            df['MFI'] = self._calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], 14)
            
            # On Balance Volume
            df['OBV'] = self._calculate_obv(df['Close'], df['Volume'])
            
            # Accumulation/Distribution Line
            df['ADL'] = self._calculate_adl(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Chaikin Money Flow
            df['CMF'] = self._calculate_cmf(df['High'], df['Low'], df['Close'], df['Volume'], 20)
            
            # Volume indicators
            df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1)  # Fill NaN values with 1
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Price patterns
            df['Doji'] = self._detect_doji(df['Open'], df['High'], df['Low'], df['Close'])
            df['Hammer'] = self._detect_hammer(df['Open'], df['High'], df['Low'], df['Close'])
            df['Shooting_Star'] = self._detect_shooting_star(df['Open'], df['High'], df['Low'], df['Close'])
            df['Engulfing'] = self._detect_engulfing(df['Open'], df['High'], df['Low'], df['Close'])
            
            # Support and Resistance
            df['Support_20'] = df['Low'].rolling(window=20).min()
            df['Resistance_20'] = df['High'].rolling(window=20).max()
            df['Support_50'] = df['Low'].rolling(window=50).min()
            df['Resistance_50'] = df['High'].rolling(window=50).max()
            
            # Price momentum
            df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
            df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
            
            # Volatility indicators
            df['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
            df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
            df['Volatility_50'] = df['Close'].pct_change().rolling(window=50).std()
            df['Volatility_10'] = df['Volatility_10'].fillna(0)
            df['Volatility_20'] = df['Volatility_20'].fillna(0)
            df['Volatility_50'] = df['Volatility_50'].fillna(0)
            
            # Trend indicators
            df['Trend_Strength'] = self._calculate_trend_strength(df['Close'], 20)
            df['Trend_Direction'] = self._calculate_trend_direction(df['Close'], 20)
            
            return df
            
        except Exception as e:
            print(f"Error adding advanced indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices, period):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices, period, std_dev):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return pd.DataFrame({'upper': upper, 'middle': sma, 'lower': lower})
    
    def _calculate_stochastic(self, high, low, close, period):
        """Calculate Stochastic %K"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        return 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    def _calculate_williams_r(self, high, low, close, period):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    def _calculate_cci(self, high, low, close, period):
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _calculate_atr(self, high, low, close, period):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_adx(self, high, low, close, period):
        """Calculate Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr = self._calculate_atr(high, low, close, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()
    
    def _calculate_mfi(self, high, low, close, volume, period):
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _calculate_obv(self, close, volume):
        """Calculate On Balance Volume"""
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0)).cumsum()
        return pd.Series(obv, index=close.index)
    
    def _calculate_adl(self, high, low, close, volume):
        """Calculate Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        return (clv * volume).cumsum()
    
    def _calculate_cmf(self, high, low, close, volume, period):
        """Calculate Chaikin Money Flow"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        return (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    def _detect_doji(self, open_price, high, low, close):
        """Detect Doji pattern"""
        body_size = abs(close - open_price)
        total_range = high - low
        return ((body_size <= total_range * 0.1) & (total_range > 0)).astype(int)
    
    def _detect_hammer(self, open_price, high, low, close):
        """Detect Hammer pattern"""
        body_size = abs(close - open_price)
        lower_shadow = np.minimum(open_price, close) - low
        upper_shadow = high - np.maximum(open_price, close)
        return ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
    
    def _detect_shooting_star(self, open_price, high, low, close):
        """Detect Shooting Star pattern"""
        body_size = abs(close - open_price)
        lower_shadow = np.minimum(open_price, close) - low
        upper_shadow = high - np.maximum(open_price, close)
        return ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)
    
    def _detect_engulfing(self, open_price, high, low, close):
        """Detect Engulfing pattern"""
        prev_body = abs(close.shift() - open_price.shift())
        curr_body = abs(close - open_price)
        return ((curr_body > prev_body) & (close > open_price) & (close.shift() < open_price.shift())).astype(int)
    
    def _calculate_trend_strength(self, prices, period):
        """Calculate trend strength"""
        sma = prices.rolling(window=period).mean()
        return abs(prices - sma) / sma
    
    def _calculate_trend_direction(self, prices, period):
        """Calculate trend direction"""
        sma = prices.rolling(window=period).mean()
        return np.where(prices > sma, 1, -1)
    
    def _get_news_sentiment(self, symbol):
        """Get news sentiment from multiple free sources"""
        try:
            # Yahoo Finance news
            yahoo_news = self._get_yahoo_news(symbol)
            
            # Google News (via web scraping)
            google_news = self._get_google_news(symbol)
            
            # Reddit sentiment
            reddit_sentiment = self._get_reddit_sentiment(symbol)
            
            # Twitter sentiment (via web scraping)
            twitter_sentiment = self._get_twitter_sentiment(symbol)
            
            # Combine all sentiment data
            all_news = yahoo_news + google_news
            
            # Calculate overall sentiment
            sentiment_scores = []
            for article in all_news:
                sentiment = TextBlob(article['title'] + ' ' + article.get('summary', '')).sentiment
                sentiment_scores.append(sentiment.polarity)
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            sentiment_score = (avg_sentiment + 1) * 50  # Convert to 0-100 scale
            
            return {
                'sentiment_score': sentiment_score,
                'news_count': len(all_news),
                'recent_news': all_news[:10],
                'reddit_sentiment': reddit_sentiment,
                'twitter_sentiment': twitter_sentiment,
                'overall_sentiment': 'positive' if sentiment_score > 60 else 'negative' if sentiment_score < 40 else 'neutral'
            }
            
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {e}")
            return {'sentiment_score': 50, 'news_count': 0, 'recent_news': [], 'overall_sentiment': 'neutral'}
    
    def _get_yahoo_news(self, symbol):
        """Get news from Yahoo Finance"""
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_items = []
            for item in soup.find_all('h3', class_='Mb(5px)'):
                title = item.get_text().strip()
                if title:
                    news_items.append({'title': title, 'source': 'Yahoo Finance'})
            
            return news_items[:20]  # Limit to 20 items
            
        except Exception as e:
            print(f"Error getting Yahoo news for {symbol}: {e}")
            return []
    
    def _get_google_news(self, symbol):
        """Get news from Google News"""
        try:
            url = f"https://news.google.com/search?q={symbol}+stock&hl=en&gl=US&ceid=US:en"
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_items = []
            for item in soup.find_all('h3', class_='ipQwMb'):
                title = item.get_text().strip()
                if title:
                    news_items.append({'title': title, 'source': 'Google News'})
            
            return news_items[:20]  # Limit to 20 items
            
        except Exception as e:
            print(f"Error getting Google news for {symbol}: {e}")
            return []
    
    def _get_reddit_sentiment(self, symbol):
        """Get Reddit sentiment (simplified)"""
        try:
            # This is a simplified version - in practice you'd use Reddit API
            return {'sentiment': 50, 'mentions': 0, 'subreddits': []}
        except Exception as e:
            return {'sentiment': 50, 'mentions': 0, 'subreddits': []}
    
    def _get_twitter_sentiment(self, symbol):
        """Get Twitter sentiment (simplified)"""
        try:
            # This is a simplified version - in practice you'd use Twitter API
            return {'sentiment': 50, 'mentions': 0, 'hashtags': []}
        except Exception as e:
            return {'sentiment': 50, 'mentions': 0, 'hashtags': []}
    
    def _get_insider_trading(self, symbol):
        """Get insider trading data from free sources"""
        try:
            # This would typically use SEC EDGAR API or similar
            # For now, return placeholder data
            return {
                'insider_buys': 0,
                'insider_sells': 0,
                'net_insider_activity': 0,
                'insider_confidence': 50
            }
        except Exception as e:
            return {'insider_buys': 0, 'insider_sells': 0, 'net_insider_activity': 0, 'insider_confidence': 50}
    
    def _get_options_data(self, symbol):
        """Get options data"""
        try:
            ticker = yf.Ticker(symbol)
            options = ticker.option_chain()
            
            if options.calls.empty and options.puts.empty:
                return {'put_call_ratio': 1.0, 'implied_volatility': 0.2, 'options_volume': 0}
            
            # Calculate put/call ratio
            put_volume = options.puts['volume'].sum() if not options.puts.empty else 0
            call_volume = options.calls['volume'].sum() if not options.calls.empty else 0
            put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
            
            # Calculate average implied volatility
            iv = 0
            if not options.calls.empty:
                iv = options.calls['impliedVolatility'].mean()
            elif not options.puts.empty:
                iv = options.puts['impliedVolatility'].mean()
            
            return {
                'put_call_ratio': put_call_ratio,
                'implied_volatility': iv if not pd.isna(iv) else 0.2,
                'options_volume': put_volume + call_volume
            }
            
        except Exception as e:
            print(f"Error getting options data for {symbol}: {e}")
            return {'put_call_ratio': 1.0, 'implied_volatility': 0.2, 'options_volume': 0}
    
    def _get_institutional_holdings(self, symbol):
        """Get institutional holdings data"""
        try:
            # This would typically use SEC 13F filings or similar
            # For now, return placeholder data
            return {
                'institutional_ownership': 0.5,
                'institutional_confidence': 50,
                'hedge_fund_activity': 0
            }
        except Exception as e:
            return {'institutional_ownership': 0.5, 'institutional_confidence': 50, 'hedge_fund_activity': 0}
    
    def _get_earnings_data(self, symbol):
        """Get earnings data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'next_earnings_date': info.get('earningsDate', None),
                'earnings_growth': info.get('earningsGrowth', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margins': info.get('profitMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0)
            }
        except Exception as e:
            return {'next_earnings_date': None, 'earnings_growth': 0, 'revenue_growth': 0, 'profit_margins': 0, 'return_on_equity': 0}
    
    def _get_economic_indicators(self):
        """Get economic indicators"""
        try:
            # This would typically use FRED API or similar
            # For now, return placeholder data
            return {
                'vix': 20.0,
                'fed_rate': 5.25,
                'gdp_growth': 2.5,
                'inflation': 3.0,
                'unemployment': 3.8
            }
        except Exception as e:
            return {'vix': 20.0, 'fed_rate': 5.25, 'gdp_growth': 2.5, 'inflation': 3.0, 'unemployment': 3.8}
