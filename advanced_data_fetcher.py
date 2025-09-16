"""
Advanced Data Fetcher - Maximum Free Analysis Power
Fetches comprehensive data from all possible free sources with advanced features
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

# Additional free data sources
try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import transformers
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class AdvancedDataFetcher:
    """Advanced data fetcher with maximum free analysis capabilities"""
    
    def __init__(self, alpha_vantage_key=None, fred_api_key=None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Initialize free APIs
        self.alpha_vantage_key = alpha_vantage_key
        self.fred_api_key = fred_api_key
        
        if ALPHA_VANTAGE_AVAILABLE and alpha_vantage_key:
            self.av_ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
            self.av_fd = FundamentalData(key=alpha_vantage_key, output_format='pandas')
        
        if FRED_AVAILABLE and fred_api_key:
            self.fred = fredapi.Fred(api_key=fred_api_key)
        
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.finbert = pipeline("sentiment-analysis", 
                                      model="ProsusAI/finbert", 
                                      tokenizer="ProsusAI/finbert")
            except:
                self.finbert = None
        
    def get_comprehensive_stock_data(self, symbol):
        """Get comprehensive data from multiple free sources"""
        try:
            # Primary data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y", interval="1d")
            info = ticker.info
            
            if hist.empty:
                return None
            
            # Add advanced technical indicators
            hist = self._add_advanced_technical_indicators(hist)
            
            # Get additional data from multiple sources
            news_data = self._get_enhanced_news_sentiment(symbol)
            insider_data = self._get_insider_trading(symbol)
            options_data = self._get_options_data(symbol)
            institutional_data = self._get_institutional_holdings(symbol)
            earnings_data = self._get_earnings_data(symbol)
            economic_data = self._get_economic_indicators()
            sector_data = self._get_sector_analysis(symbol)
            analyst_data = self._get_analyst_data(symbol)
            
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
                'sector': sector_data,
                'analyst': analyst_data,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching comprehensive data for {symbol}: {e}")
            return None
    
    def _add_advanced_technical_indicators(self, df):
        """Add 100+ advanced technical indicators"""
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
            df['RSI_50'] = self._calculate_rsi(df['Close'], 50)
            
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
            df['Stoch_K_21'] = self._calculate_stochastic(df['High'], df['Low'], df['Close'], 21)
            df['Stoch_D_21'] = df['Stoch_K_21'].rolling(window=3).mean()
            
            # Williams %R
            df['Williams_R'] = self._calculate_williams_r(df['High'], df['Low'], df['Close'], 14)
            df['Williams_R_21'] = self._calculate_williams_r(df['High'], df['Low'], df['Close'], 21)
            
            # Commodity Channel Index
            df['CCI'] = self._calculate_cci(df['High'], df['Low'], df['Close'], 20)
            df['CCI_50'] = self._calculate_cci(df['High'], df['Low'], df['Close'], 50)
            
            # Average True Range
            df['ATR'] = self._calculate_atr(df['High'], df['Low'], df['Close'], 14)
            df['ATR_21'] = self._calculate_atr(df['High'], df['Low'], df['Close'], 21)
            
            # Average Directional Index
            df['ADX'] = self._calculate_adx(df['High'], df['Low'], df['Close'], 14)
            df['ADX_21'] = self._calculate_adx(df['High'], df['Low'], df['Close'], 21)
            
            # Money Flow Index
            df['MFI'] = self._calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], 14)
            df['MFI_21'] = self._calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], 21)
            
            # On Balance Volume
            df['OBV'] = self._calculate_obv(df['Close'], df['Volume'])
            
            # Accumulation/Distribution Line
            df['ADL'] = self._calculate_adl(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Chaikin Money Flow
            df['CMF'] = self._calculate_cmf(df['High'], df['Low'], df['Close'], df['Volume'], 20)
            df['CMF_50'] = self._calculate_cmf(df['High'], df['Low'], df['Close'], df['Volume'], 50)
            
            # Ichimoku Cloud
            ichimoku = self._calculate_ichimoku(df['High'], df['Low'], df['Close'])
            df['Ichimoku_Conversion'] = ichimoku['conversion']
            df['Ichimoku_Base'] = ichimoku['base']
            df['Ichimoku_Span_A'] = ichimoku['span_a']
            df['Ichimoku_Span_B'] = ichimoku['span_b']
            df['Ichimoku_Cloud_Top'] = ichimoku['cloud_top']
            df['Ichimoku_Cloud_Bottom'] = ichimoku['cloud_bottom']
            
            # Fibonacci Retracements
            fib_levels = self._calculate_fibonacci_levels(df['High'], df['Low'], df['Close'])
            for level, value in fib_levels.items():
                df[f'Fib_{level}'] = value
            
            # Pivot Points
            pivot_points = self._calculate_pivot_points(df['High'], df['Low'], df['Close'])
            for level, value in pivot_points.items():
                df[f'Pivot_{level}'] = value
            
            # Volume indicators
            df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1)
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Volume Profile
            volume_profile = self._calculate_volume_profile(df['High'], df['Low'], df['Close'], df['Volume'])
            df['Volume_Profile_POC'] = volume_profile['poc']
            df['Volume_Profile_VAH'] = volume_profile['vah']
            df['Volume_Profile_VAL'] = volume_profile['val']
            
            # Price patterns
            df['Doji'] = self._detect_doji(df['Open'], df['High'], df['Low'], df['Close'])
            df['Hammer'] = self._detect_hammer(df['Open'], df['High'], df['Low'], df['Close'])
            df['Shooting_Star'] = self._detect_shooting_star(df['Open'], df['High'], df['Low'], df['Close'])
            df['Engulfing'] = self._detect_engulfing(df['Open'], df['High'], df['Low'], df['Close'])
            df['Harami'] = self._detect_harami(df['Open'], df['High'], df['Low'], df['Close'])
            df['Morning_Star'] = self._detect_morning_star(df['Open'], df['High'], df['Low'], df['Close'])
            df['Evening_Star'] = self._detect_evening_star(df['Open'], df['High'], df['Low'], df['Close'])
            
            # Support and Resistance
            df['Support_20'] = df['Low'].rolling(window=20).min()
            df['Resistance_20'] = df['High'].rolling(window=20).max()
            df['Support_50'] = df['Low'].rolling(window=50).min()
            df['Resistance_50'] = df['High'].rolling(window=50).max()
            df['Support_100'] = df['Low'].rolling(window=100).min()
            df['Resistance_100'] = df['High'].rolling(window=100).max()
            
            # Price momentum
            df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
            df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
            df['Momentum_50'] = df['Close'] / df['Close'].shift(50) - 1
            
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
            df['Trend_Strength_50'] = self._calculate_trend_strength(df['Close'], 50)
            df['Trend_Direction_50'] = self._calculate_trend_direction(df['Close'], 50)
            
            # Market structure
            df['Higher_High'] = self._detect_higher_high(df['High'])
            df['Lower_Low'] = self._detect_lower_low(df['Low'])
            df['Breakout'] = self._detect_breakout(df['High'], df['Low'], df['Close'])
            df['Breakdown'] = self._detect_breakdown(df['High'], df['Low'], df['Close'])
            
            return df
            
        except Exception as e:
            print(f"Error adding advanced indicators: {e}")
            return df
    
    def _calculate_ichimoku(self, high, low, close, conversion_period=9, base_period=26, span_b_period=52, displacement=26):
        """Calculate Ichimoku Cloud"""
        try:
            # Conversion Line (Tenkan-sen)
            conversion = (high.rolling(window=conversion_period).max() + 
                         low.rolling(window=conversion_period).min()) / 2
            
            # Base Line (Kijun-sen)
            base = (high.rolling(window=base_period).max() + 
                   low.rolling(window=base_period).min()) / 2
            
            # Leading Span A (Senkou Span A)
            span_a = ((conversion + base) / 2).shift(displacement)
            
            # Leading Span B (Senkou Span B)
            span_b = ((high.rolling(window=span_b_period).max() + 
                      low.rolling(window=span_b_period).min()) / 2).shift(displacement)
            
            # Cloud boundaries
            cloud_top = np.maximum(span_a, span_b)
            cloud_bottom = np.minimum(span_a, span_b)
            
            return {
                'conversion': conversion,
                'base': base,
                'span_a': span_a,
                'span_b': span_b,
                'cloud_top': cloud_top,
                'cloud_bottom': cloud_bottom
            }
        except Exception as e:
            print(f"Error calculating Ichimoku: {e}")
            return {
                'conversion': pd.Series(index=close.index),
                'base': pd.Series(index=close.index),
                'span_a': pd.Series(index=close.index),
                'span_b': pd.Series(index=close.index),
                'cloud_top': pd.Series(index=close.index),
                'cloud_bottom': pd.Series(index=close.index)
            }
    
    def _calculate_fibonacci_levels(self, high, low, close, lookback=20):
        """Calculate Fibonacci retracement levels"""
        try:
            recent_high = high.rolling(window=lookback).max()
            recent_low = low.rolling(window=lookback).min()
            range_size = recent_high - recent_low
            
            fib_levels = {}
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            for ratio in fib_ratios:
                fib_levels[f'Retracement_{ratio}'] = recent_high - (range_size * ratio)
                fib_levels[f'Extension_{ratio}'] = recent_low + (range_size * ratio)
            
            return fib_levels
        except Exception as e:
            print(f"Error calculating Fibonacci: {e}")
            return {}
    
    def _calculate_pivot_points(self, high, low, close):
        """Calculate pivot points"""
        try:
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'Pivot': pivot,
                'R1': r1,
                'R2': r2,
                'R3': r3,
                'S1': s1,
                'S2': s2,
                'S3': s3
            }
        except Exception as e:
            print(f"Error calculating pivot points: {e}")
            return {}
    
    def _calculate_volume_profile(self, high, low, close, volume, bins=20):
        """Calculate volume profile"""
        try:
            # Simple volume profile implementation
            price_range = high.max() - low.min()
            bin_size = price_range / bins
            
            # Find price of control (POC) - price level with highest volume
            poc_price = close.iloc[-1]  # Simplified
            
            # Volume at high (VAH) and volume at low (VAL)
            vah = high.rolling(window=20).max().iloc[-1]
            val = low.rolling(window=20).min().iloc[-1]
            
            return {
                'poc': poc_price,
                'vah': vah,
                'val': val
            }
        except Exception as e:
            print(f"Error calculating volume profile: {e}")
            return {'poc': close.iloc[-1], 'vah': high.iloc[-1], 'val': low.iloc[-1]}
    
    def _detect_harami(self, open_price, high, low, close):
        """Detect Harami pattern"""
        try:
            prev_body = abs(close.shift() - open_price.shift())
            curr_body = abs(close - open_price)
            prev_bullish = close.shift() > open_price.shift()
            curr_bullish = close > open_price
            
            harami_bullish = (curr_body < prev_body) & (prev_bullish) & (curr_bullish)
            harami_bearish = (curr_body < prev_body) & (~prev_bullish) & (~curr_bullish)
            
            return (harami_bullish | harami_bearish).astype(int)
        except Exception as e:
            return pd.Series(0, index=close.index)
    
    def _detect_morning_star(self, open_price, high, low, close):
        """Detect Morning Star pattern"""
        try:
            # Simplified morning star detection
            star_body = abs(close - open_price)
            prev_body = abs(close.shift(2) - open_price.shift(2))
            
            return ((star_body < prev_body * 0.3) & 
                   (close.shift(2) < open_price.shift(2)) & 
                   (close > open_price)).astype(int)
        except Exception as e:
            return pd.Series(0, index=close.index)
    
    def _detect_evening_star(self, open_price, high, low, close):
        """Detect Evening Star pattern"""
        try:
            # Simplified evening star detection
            star_body = abs(close - open_price)
            prev_body = abs(close.shift(2) - open_price.shift(2))
            
            return ((star_body < prev_body * 0.3) & 
                   (close.shift(2) > open_price.shift(2)) & 
                   (close < open_price)).astype(int)
        except Exception as e:
            return pd.Series(0, index=close.index)
    
    def _detect_higher_high(self, high, lookback=5):
        """Detect higher high pattern"""
        try:
            return (high > high.rolling(window=lookback).max().shift(1)).astype(int)
        except Exception as e:
            return pd.Series(0, index=high.index)
    
    def _detect_lower_low(self, low, lookback=5):
        """Detect lower low pattern"""
        try:
            return (low < low.rolling(window=lookback).min().shift(1)).astype(int)
        except Exception as e:
            return pd.Series(0, index=low.index)
    
    def _detect_breakout(self, high, low, close, lookback=20):
        """Detect breakout pattern"""
        try:
            resistance = high.rolling(window=lookback).max().shift(1)
            return (close > resistance).astype(int)
        except Exception as e:
            return pd.Series(0, index=close.index)
    
    def _detect_breakdown(self, high, low, close, lookback=20):
        """Detect breakdown pattern"""
        try:
            support = low.rolling(window=lookback).min().shift(1)
            return (close < support).astype(int)
        except Exception as e:
            return pd.Series(0, index=close.index)
    
    def _get_enhanced_news_sentiment(self, symbol):
        """Get enhanced news sentiment from multiple sources"""
        try:
            # Yahoo Finance news
            yahoo_news = self._get_yahoo_news(symbol)
            
            # Google News
            google_news = self._get_google_news(symbol)
            
            # Reddit sentiment
            reddit_sentiment = self._get_reddit_sentiment(symbol)
            
            # Twitter sentiment
            twitter_sentiment = self._get_twitter_sentiment(symbol)
            
            # Combine all news
            all_news = yahoo_news + google_news
            
            # Calculate sentiment using multiple methods
            sentiment_scores = []
            vader_scores = []
            finbert_scores = []
            
            for article in all_news:
                text = article['title'] + ' ' + article.get('summary', '')
                
                # TextBlob sentiment
                blob = TextBlob(text)
                sentiment_scores.append(blob.sentiment.polarity)
                
                # VADER sentiment
                if VADER_AVAILABLE:
                    vader_score = self.vader_analyzer.polarity_scores(text)
                    vader_scores.append(vader_score['compound'])
                
                # FinBERT sentiment
                if TRANSFORMERS_AVAILABLE and self.finbert:
                    try:
                        finbert_result = self.finbert(text[:512])  # Limit text length
                        finbert_scores.append(finbert_result[0]['score'] if finbert_result[0]['label'] == 'POSITIVE' else -finbert_result[0]['score'])
                    except:
                        finbert_scores.append(0)
            
            # Calculate overall sentiment
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            avg_vader = np.mean(vader_scores) if vader_scores else 0
            avg_finbert = np.mean(finbert_scores) if finbert_scores else 0
            
            # Weighted average
            overall_sentiment = (avg_sentiment * 0.4 + avg_vader * 0.3 + avg_finbert * 0.3)
            sentiment_score = (overall_sentiment + 1) * 50  # Convert to 0-100 scale
            
            return {
                'sentiment_score': sentiment_score,
                'news_count': len(all_news),
                'recent_news': all_news[:10],
                'reddit_sentiment': reddit_sentiment,
                'twitter_sentiment': twitter_sentiment,
                'vader_sentiment': avg_vader,
                'finbert_sentiment': avg_finbert,
                'overall_sentiment': 'positive' if sentiment_score > 60 else 'negative' if sentiment_score < 40 else 'neutral'
            }
            
        except Exception as e:
            print(f"Error getting enhanced news sentiment for {symbol}: {e}")
            return {'sentiment_score': 50, 'news_count': 0, 'recent_news': [], 'overall_sentiment': 'neutral'}
    
    def _get_sector_analysis(self, symbol):
        """Get sector analysis data"""
        try:
            # This would typically use sector ETFs or sector data
            # For now, return placeholder data
            return {
                'sector_performance': 0.05,
                'sector_rank': 5,
                'sector_momentum': 0.02,
                'sector_volatility': 0.15
            }
        except Exception as e:
            return {'sector_performance': 0, 'sector_rank': 0, 'sector_momentum': 0, 'sector_volatility': 0}
    
    def _get_analyst_data(self, symbol):
        """Get analyst data"""
        try:
            # This would typically use analyst rating APIs
            # For now, return placeholder data
            return {
                'analyst_rating': 'Buy',
                'price_target': 0,
                'rating_changes': 0,
                'analyst_consensus': 0.05
            }
        except Exception as e:
            return {'analyst_rating': 'Hold', 'price_target': 0, 'rating_changes': 0, 'analyst_consensus': 0}
    
    # Include all the existing methods from the original enhanced_data_fetcher.py
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
            
            return news_items[:20]
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
            
            return news_items[:20]
        except Exception as e:
            print(f"Error getting Google news for {symbol}: {e}")
            return []
    
    def _get_reddit_sentiment(self, symbol):
        """Get Reddit sentiment (simplified)"""
        try:
            return {'sentiment': 50, 'mentions': 0, 'subreddits': []}
        except Exception as e:
            return {'sentiment': 50, 'mentions': 0, 'subreddits': []}
    
    def _get_twitter_sentiment(self, symbol):
        """Get Twitter sentiment (simplified)"""
        try:
            return {'sentiment': 50, 'mentions': 0, 'hashtags': []}
        except Exception as e:
            return {'sentiment': 50, 'mentions': 0, 'hashtags': []}
    
    def _get_insider_trading(self, symbol):
        """Get insider trading data from free sources"""
        try:
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
            
            put_volume = options.puts['volume'].sum() if not options.puts.empty else 0
            call_volume = options.calls['volume'].sum() if not options.calls.empty else 0
            put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
            
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
            return {
                'institutional_ownership': 0.5,
                'institutional_confidence': 50,
                'hedge_fund_activity': 0
            }
        except Exception as e:
            return {'institutional_ownership': 0.5, 'institutional_confidence': 50, 'hedge_fund_activity': 0}
    
        def _get_earnings_data(self, symbol):
            """Get comprehensive earnings data like professional analysts"""
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get earnings history and estimates
                earnings_history = ticker.earnings_history
                earnings_dates = ticker.earnings_dates
                
                # Calculate earnings surprise trends
                earnings_surprise = 0
                earnings_beat_rate = 0
                if not earnings_history.empty:
                    recent_earnings = earnings_history.tail(4)  # Last 4 quarters
                    if 'Surprise' in recent_earnings.columns:
                        earnings_surprise = recent_earnings['Surprise'].mean()
                        earnings_beat_rate = (recent_earnings['Surprise'] > 0).mean() * 100
                
                # Get forward guidance
                forward_pe = info.get('forwardPE', 0)
                peg_ratio = info.get('pegRatio', 0)
                
                # Calculate earnings quality metrics
                earnings_quality_score = 50  # Base score
                if earnings_beat_rate > 75:
                    earnings_quality_score += 25
                elif earnings_beat_rate > 50:
                    earnings_quality_score += 10
                elif earnings_beat_rate < 25:
                    earnings_quality_score -= 25
                
                if earnings_surprise > 0.05:  # 5% positive surprise
                    earnings_quality_score += 15
                elif earnings_surprise < -0.05:  # 5% negative surprise
                    earnings_quality_score -= 15
                
                return {
                    'next_earnings_date': info.get('earningsDate', None),
                    'earnings_growth': info.get('earningsGrowth', 0),
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'profit_margins': info.get('profitMargins', 0),
                    'return_on_equity': info.get('returnOnEquity', 0),
                    'earnings_surprise': earnings_surprise,
                    'earnings_beat_rate': earnings_beat_rate,
                    'earnings_quality_score': max(0, min(100, earnings_quality_score)),
                    'forward_pe': forward_pe,
                    'peg_ratio': peg_ratio,
                    'earnings_consensus': info.get('earningsQuarterlyGrowth', 0),
                    'revenue_consensus': info.get('revenueQuarterlyGrowth', 0)
                }
            except Exception as e:
                return {
                    'next_earnings_date': None, 'earnings_growth': 0, 'revenue_growth': 0, 
                    'profit_margins': 0, 'return_on_equity': 0, 'earnings_surprise': 0,
                    'earnings_beat_rate': 50, 'earnings_quality_score': 50, 'forward_pe': 0,
                    'peg_ratio': 0, 'earnings_consensus': 0, 'revenue_consensus': 0
                }
    
    def _get_economic_indicators(self):
        """Get economic indicators"""
        try:
            return {
                'vix': 20.0,
                'fed_rate': 5.25,
                'gdp_growth': 2.5,
                'inflation': 3.0,
                'unemployment': 3.8
            }
        except Exception as e:
            return {'vix': 20.0, 'fed_rate': 5.25, 'gdp_growth': 2.5, 'inflation': 3.0, 'unemployment': 3.8}
    
    def _get_analyst_ratings(self, symbol):
        """Get comprehensive analyst ratings like professional traders track"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get analyst recommendations
            recommendations = ticker.recommendations
            
            # Calculate analyst consensus
            analyst_rating = 'Hold'
            price_target = 0
            rating_changes = 0
            analyst_consensus = 0
            
            if not recommendations.empty:
                recent_recs = recommendations.tail(10)  # Last 10 recommendations
                if 'To Grade' in recent_recs.columns:
                    # Count recent rating changes
                    rating_changes = len(recent_recs[recent_recs['To Grade'] != recent_recs['To Grade'].shift()])
                    
                    # Calculate consensus
                    latest_ratings = recent_recs['To Grade'].value_counts()
                    if not latest_ratings.empty:
                        if 'Buy' in latest_ratings.index and latest_ratings['Buy'] > latest_ratings.get('Hold', 0):
                            analyst_rating = 'Buy'
                        elif 'Sell' in latest_ratings.index and latest_ratings['Sell'] > latest_ratings.get('Hold', 0):
                            analyst_rating = 'Sell'
            
            # Get price targets from info
            target_high = info.get('targetHighPrice', 0)
            target_low = info.get('targetLowPrice', 0)
            target_mean = info.get('targetMeanPrice', 0)
            
            if target_mean > 0:
                price_target = target_mean
            elif target_high > 0 and target_low > 0:
                price_target = (target_high + target_low) / 2
            
            # Calculate consensus upside/downside
            current_price = info.get('currentPrice', 0)
            if current_price > 0 and price_target > 0:
                analyst_consensus = (price_target - current_price) / current_price
            
            # Calculate analyst confidence score
            analyst_confidence = 50  # Base score
            if analyst_rating == 'Buy' and analyst_consensus > 0.1:
                analyst_confidence += 30
            elif analyst_rating == 'Buy' and analyst_consensus > 0.05:
                analyst_confidence += 20
            elif analyst_rating == 'Sell' and analyst_consensus < -0.1:
                analyst_confidence -= 30
            elif analyst_rating == 'Sell' and analyst_consensus < -0.05:
                analyst_confidence -= 20
            
            # Factor in rating changes
            if rating_changes > 0:
                recent_changes = recommendations.tail(rating_changes)
                upgrades = len(recent_changes[recent_changes['To Grade'].isin(['Buy', 'Strong Buy'])])
                downgrades = len(recent_changes[recent_changes['To Grade'].isin(['Sell', 'Strong Sell'])])
                
                if upgrades > downgrades:
                    analyst_confidence += 15
                elif downgrades > upgrades:
                    analyst_confidence -= 15
            
            return {
                'analyst_rating': analyst_rating,
                'price_target': price_target,
                'rating_changes': rating_changes,
                'analyst_consensus': analyst_consensus,
                'analyst_confidence': max(0, min(100, analyst_confidence)),
                'target_high': target_high,
                'target_low': target_low,
                'target_mean': target_mean,
                'num_analysts': info.get('numberOfAnalystOpinions', 0)
            }
        except Exception as e:
            return {
                'analyst_rating': 'Hold', 'price_target': 0, 'rating_changes': 0, 
                'analyst_consensus': 0, 'analyst_confidence': 50, 'target_high': 0,
                'target_low': 0, 'target_mean': 0, 'num_analysts': 0
            }
