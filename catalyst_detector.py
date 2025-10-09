"""
Catalyst Detection System - Explosive Opportunity Hunter
Detects market catalysts for 100-10000%+ gains using free data sources

Detects:
- News-driven spikes (AMD/OpenAI type events) 
- Earnings surprises
- Unusual volume/price breakouts
- Insider activity spikes
- Partnership announcements
- Technical breakout patterns
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*No timezone found.*')
warnings.filterwarnings('ignore', message='.*No data found.*')
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
import sys
import bs4
import json
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# Add Finnhub API (free tier)
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
FINNHUB_BASE = 'https://finnhub.io/api/v1'

def fetch_finnhub_price_volume(symbol):
    if not FINNHUB_API_KEY:
        return None
    try:
        url = f"{FINNHUB_BASE}/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'current_price': data.get('c'),
                'previous_close': data.get('pc'),
                'volume': data.get('v'),
            }
    except Exception:
        pass
    return None

def fetch_finnhub_news(symbol):
    if not FINNHUB_API_KEY:
        return []
    try:
        url = f"{FINNHUB_BASE}/company-news?symbol={symbol}&from={datetime.now().date()}&to={datetime.now().date()}&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []

@contextmanager
def suppress_stdout_stderr():
    """Safely suppress stdout and stderr to reduce noise"""
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    except Exception as e:
        # If suppression fails, just continue without it
        yield
    finally:
        # Always restore original streams
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

class CatalystDetector:
    """Advanced catalyst detection system for explosive trading opportunities"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Initialize sentiment analyzers
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.news_sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Rate limiting and caching
        self._last_request = 0
        self._request_delay = 0.1  # 100ms between requests to avoid rate limits
        self._company_names_cache = {}  # Cache company names to avoid repeated lookups
        
        # Catalyst keywords for different types of events
        self.catalyst_keywords = {
            'partnership': ['partnership', 'collaboration', 'deal', 'agreement', 'joint venture', 'alliance', 'merger', 'acquisition'],
            'earnings': ['earnings', 'revenue', 'profit', 'guidance', 'outlook', 'beat', 'miss', 'surprise'],
            'product': ['product', 'launch', 'release', 'approval', 'patent', 'innovation', 'breakthrough'],
            'upgrade': ['upgrade', 'raised', 'target', 'buy', 'overweight', 'outperform'],
            'ai_tech': ['artificial intelligence', 'AI', 'machine learning', 'openai', 'chatgpt', 'automation'],
            'regulatory': ['FDA', 'approval', 'cleared', 'authorized', 'regulation'],
            'insider': ['insider', 'bought', 'sold', 'filing', 'SEC']
        }
        
    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last = current_time - self._last_request
        if time_since_last < self._request_delay:
            time.sleep(self._request_delay - time_since_last)
        self._last_request = time.time()
    
    def get_company_name(self, symbol: str) -> str:
        """Get the full company name for a stock symbol"""
        if symbol in self._company_names_cache:
            return self._company_names_cache[symbol]
        
        # Known company names for common symbols
        known_names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'NVDA': 'NVIDIA Corporation',
            'AMD': 'Advanced Micro Devices Inc',
            'TSLA': 'Tesla Inc',
            'META': 'Meta Platforms Inc',
            'GOOGL': 'Alphabet Inc Class A',
            'AMZN': 'Amazon.com Inc',
            'NFLX': 'Netflix Inc',
            'JPM': 'JPMorgan Chase & Co',
            'BAC': 'Bank of America Corp',
            'KO': 'The Coca-Cola Company',
            'PG': 'Procter & Gamble Co',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc',
            'MA': 'Mastercard Inc',
            'DIS': 'The Walt Disney Company',
            'ABBV': 'AbbVie Inc',
            'PLTR': 'Palantir Technologies Inc',
            'CRWD': 'CrowdStrike Holdings Inc',
            'SNOW': 'Snowflake Inc',
            'BIDU': 'Baidu Inc',
            'BBIG': 'Vinco Ventures Inc',
            'BENE': 'Benessere Capital Acquisition Corp',
            'BFRI': 'Biofrontera Inc'
        }
        
        # First try known names
        if symbol in known_names:
            company_name = known_names[symbol]
            self._company_names_cache[symbol] = company_name
            return company_name
        
        # Then try yfinance
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different fields for company name
            company_name = (info.get('longName') or 
                          info.get('shortName') or 
                          info.get('companyName') or 
                          symbol)
            
            # Clean up the name
            if company_name and company_name != symbol:
                # Remove common suffixes to shorten names
                company_name = company_name.replace(' Inc.', ' Inc')
                company_name = company_name.replace(' Corporation', ' Corp')
                company_name = company_name.replace(' Company', ' Co')
                
            self._company_names_cache[symbol] = company_name
            return company_name
            
        except Exception as e:
            # Fallback to symbol if we can't get the name
            self._company_names_cache[symbol] = symbol
            return symbol
    
    def detect_explosive_momentum(self, symbol: str, days_back: int = 5) -> dict:
        """
        Detect explosive price/volume momentum that suggests a catalyst
        Returns momentum indicators and breakout signals
        """
        try:
            self._rate_limit()
            
            # Get most recent data with multiple approaches for current prices
            hist = None
            current_price = None
            
            # Method 1: Get current/real-time price first
            try:
                ticker = yf.Ticker(symbol)
                # Get current price from info (most recent)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            except:
                pass
            
            # Method 2: Try recent periods for historical data (ensuring current data)
            periods = ["5d", "1mo", "3mo"]  # Start with shorter periods for more recent data
            
            for period in periods:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period, interval="1d", prepost=True, actions=False)
                    if len(hist) >= 5:  # Need at least 5 days of data
                        # Use the most recent close price if we didn't get current price
                        if current_price is None:
                            current_price = hist['Close'].iloc[-1]
                        break
                except:
                    continue
            
            # Method 3: Fallback with specific date range (ensuring today's data)
            if hist is None or len(hist) < 5:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)  # Get last 30 days
                    hist = yf.download(symbol, start=start_date, end=end_date, progress=False, prepost=True)
                    if current_price is None and hist is not None and len(hist) > 0:
                        current_price = hist['Close'].iloc[-1]
                except:
                    pass
            
            # Final fallback: try Finnhub if yfinance fails
            if hist is None or len(hist) < 5:
                finnhub_data = fetch_finnhub_price_volume(symbol)
                if finnhub_data and finnhub_data['current_price']:
                    current_price = finnhub_data['current_price']
                    previous_close = finnhub_data['previous_close']
                    volume = finnhub_data['volume']
                    price_change_1d = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
                    # Only basic metrics if no full history
                    return {
                        'symbol': symbol,
                        'current_price': current_price,
                        'price_change_1d': price_change_1d,
                        'price_change_5d': 0,
                        'volume_spike': volume,
                        'current_volume': volume,
                        'avg_volume_20d': volume,
                        'volatility_20d': 0,
                        'breakout_high': False,
                        'breakdown_low': False,
                        'momentum_score': price_change_1d,
                        'catalyst_likely': price_change_1d > 10,
                        'explosive_move': price_change_1d > 10,
                        'source': 'finnhub'
                    }
            
            # Final fallback: skip analysis if no real data
            if hist is None or len(hist) < 5:
                return {'error': 'No reliable price/volume data available for this symbol. Skipping analysis.'}
            
            # Use the most recent/current price we obtained
            if current_price is None:
                current_price = hist['Close'].iloc[-1]
            recent_prices = hist['Close'].iloc[-days_back:]
            recent_volume = hist['Volume'].iloc[-days_back:]
            
            # Calculate momentum metrics
            price_change_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            price_change_5d = ((current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]) * 100 if len(hist) >= 6 else 0
            
            # Volume analysis
            avg_volume_20d = hist['Volume'].iloc[-20:].mean() if len(hist) >= 20 else hist['Volume'].mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_spike = (current_volume / avg_volume_20d) if avg_volume_20d > 0 else 1
            
            # Price volatility
            volatility_20d = hist['Close'].pct_change().iloc[-20:].std() * np.sqrt(252) if len(hist) >= 20 else 0
            
            # Breakout detection
            high_20d = hist['High'].iloc[-20:].max() if len(hist) >= 20 else hist['High'].max()
            low_20d = hist['Low'].iloc[-20:].min() if len(hist) >= 20 else hist['Low'].min()
            breakout_high = current_price >= high_20d * 0.98  # Within 2% of 20-day high
            breakdown_low = current_price <= low_20d * 1.02   # Within 2% of 20-day low
            
            # Momentum score calculation
            momentum_score = 0
            
            # Price momentum (weighted heavily for explosive moves)
            if abs(price_change_1d) > 20:
                momentum_score += 50
            elif abs(price_change_1d) > 10:
                momentum_score += 30
            elif abs(price_change_1d) > 5:
                momentum_score += 15
                
            # Volume spike (critical for catalyst validation)
            if volume_spike > 5:
                momentum_score += 40
            elif volume_spike > 3:
                momentum_score += 25
            elif volume_spike > 2:
                momentum_score += 15
                
            # Breakout confirmation
            if breakout_high and price_change_1d > 0:
                momentum_score += 20
            elif breakdown_low and price_change_1d < 0:
                momentum_score += 15
                
            # Multi-day acceleration
            if abs(price_change_5d) > abs(price_change_1d) * 2:
                momentum_score += 15
                
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_1d': price_change_1d,
                'price_change_5d': price_change_5d,
                'volume_spike': volume_spike,
                'current_volume': current_volume,
                'avg_volume_20d': avg_volume_20d,
                'volatility_20d': volatility_20d,
                'breakout_high': breakout_high,
                'breakdown_low': breakdown_low,
                'momentum_score': momentum_score,
                'catalyst_likely': momentum_score >= 50,
                'explosive_move': abs(price_change_1d) > 10 and volume_spike > 3
            }
            
        except Exception as e:
            return {'error': f'Error analyzing {symbol}: {str(e)}'}
    
    def get_news_sentiment(self, symbol: str, limit: int = 10) -> dict:
        """
        Get recent news and analyze sentiment for catalyst detection
        Uses free news sources to avoid rate limits
        """
        news_data = {
            'articles': [],
            'sentiment_score': 0,
            'catalyst_detected': False,
            'catalyst_type': None,
            'confidence': 0
        }
        
        try:
            self._rate_limit()
            
            # Method 1: Yahoo Finance news (most reliable)
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
            except:
                # Fallback to synthetic news simulation for testing
                news = []
            
            # Fallback: try Finnhub news if yfinance fails
            if not news:
                finnhub_news = fetch_finnhub_news(symbol)
                if finnhub_news:
                    for article in finnhub_news[:limit]:
                        title = article.get('headline', '')
                        summary = article.get('summary', '')
                        text = f"{title} {summary}".lower()
                        sentiment_score = 0
                        if VADER_AVAILABLE:
                            sentiment = self.sentiment_analyzer.polarity_scores(text)
                            sentiment_score = sentiment['compound']
                        news_data['articles'].append({
                            'title': title,
                            'summary': summary,
                            'sentiment': sentiment_score,
                            'catalyst_type': self._detect_catalyst_type(text),
                            'url': article.get('url', '')
                        })
                    if news_data['articles']:
                        news_data['sentiment_score'] = sum(a['sentiment'] for a in news_data['articles']) / len(news_data['articles'])
            
            # Fallback: skip analysis if no real news data
            if not news and not news_data['articles']:
                return {
                    'articles': [],
                    'sentiment_score': 0,
                    'catalyst_detected': False,
                    'catalyst_type': None,
                    'confidence': 0,
                    'error': 'No reliable news data available for this symbol. Skipping analysis.'
                }

            if news:
                articles_processed = 0
                total_sentiment = 0
                catalyst_signals = []
                
                for article in news[:limit]:
                    try:
                        title = article.get('title', '')
                        summary = article.get('summary', '')
                        text = f"{title} {summary}".lower()
                        
                        # Sentiment analysis
                        if VADER_AVAILABLE:
                            sentiment = self.sentiment_analyzer.polarity_scores(text)
                            sentiment_score = sentiment['compound']
                        else:
                            # Simple keyword-based sentiment fallback
                            positive_words = ['beat', 'up', 'surge', 'gain', 'win', 'approval', 'partnership', 'growth']
                            negative_words = ['miss', 'down', 'fall', 'loss', 'decline', 'warning', 'cut']
                            
                            pos_count = sum(1 for word in positive_words if word in text)
                            neg_count = sum(1 for word in negative_words if word in text)
                            sentiment_score = (pos_count - neg_count) / max(1, pos_count + neg_count)
                        
                        total_sentiment += sentiment_score
                        articles_processed += 1
                        
                        # Catalyst detection
                        catalyst_type = self._detect_catalyst_type(text)
                        if catalyst_type:
                            catalyst_signals.append({
                                'type': catalyst_type,
                                'title': title,
                                'sentiment': sentiment_score,
                                'published': article.get('providerPublishTime', 0)
                            })
                        
                        news_data['articles'].append({
                            'title': title,
                            'summary': summary,
                            'sentiment': sentiment_score,
                            'catalyst_type': catalyst_type,
                            'url': article.get('link', '')
                        })
                        
                    except Exception as e:
                        continue
                
                # Calculate overall sentiment and catalyst probability
                if articles_processed > 0:
                    news_data['sentiment_score'] = total_sentiment / articles_processed
                    
                    # Catalyst detection logic
                    if catalyst_signals:
                        # Sort by recency and sentiment
                        catalyst_signals.sort(key=lambda x: (x['published'], abs(x['sentiment'])), reverse=True)
                        
                        strongest_catalyst = catalyst_signals[0]
                        news_data['catalyst_detected'] = True
                        news_data['catalyst_type'] = strongest_catalyst['type']
                        news_data['confidence'] = min(95, abs(strongest_catalyst['sentiment']) * 100 + len(catalyst_signals) * 10)
            
            return news_data
            
        except Exception as e:
            news_data['error'] = f'Error fetching news for {symbol}: {str(e)}'
            return news_data
    
    def _detect_catalyst_type(self, text: str) -> str:
        """Detect the type of catalyst from news text"""
        text_lower = text.lower()
        
        # Score each catalyst type
        catalyst_scores = {}
        for catalyst_type, keywords in self.catalyst_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                catalyst_scores[catalyst_type] = score
        
        # Return the highest scoring catalyst type
        if catalyst_scores:
            return max(catalyst_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def detect_earnings_surprise(self, symbol: str) -> dict:
        """
        Detect potential earnings surprises and upcoming catalysts
        """
        try:
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            earnings_data = {
                'symbol': symbol,
                'next_earnings': None,
                'surprise_probability': 0,
                'analyst_sentiment': 'neutral',
                'estimate_revision': 0
            }
            
            # Check for upcoming earnings
            calendar = ticker.calendar
            if calendar is not None and not calendar.empty:
                earnings_data['next_earnings'] = calendar.index[0] if len(calendar.index) > 0 else None
            
            # Get analyst recommendations if available
            if hasattr(ticker, 'recommendations') and ticker.recommendations is not None:
                recommendations = ticker.recommendations
                if not recommendations.empty:
                    recent_recs = recommendations.tail(5)  # Last 5 recommendations
                    
                    # Calculate sentiment from recommendations
                    rec_mapping = {'Strong Buy': 5, 'Buy': 4, 'Hold': 3, 'Sell': 2, 'Strong Sell': 1}
                    
                    sentiment_scores = []
                    for _, row in recent_recs.iterrows():
                        for col in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']:
                            if col in row and pd.notna(row[col]):
                                count = row[col]
                                if col == 'strongBuy':
                                    sentiment_scores.extend([5] * int(count))
                                elif col == 'buy':
                                    sentiment_scores.extend([4] * int(count))
                                elif col == 'hold':
                                    sentiment_scores.extend([3] * int(count))
                                elif col == 'sell':
                                    sentiment_scores.extend([2] * int(count))
                                elif col == 'strongSell':
                                    sentiment_scores.extend([1] * int(count))
                    
                    if sentiment_scores:
                        avg_sentiment = np.mean(sentiment_scores)
                        if avg_sentiment >= 4:
                            earnings_data['analyst_sentiment'] = 'bullish'
                        elif avg_sentiment <= 2:
                            earnings_data['analyst_sentiment'] = 'bearish'
                        
                        # Calculate surprise probability based on sentiment
                        earnings_data['surprise_probability'] = max(0, (avg_sentiment - 3) * 25)
            
            return earnings_data
            
        except Exception as e:
            return {'error': f'Error analyzing earnings for {symbol}: {str(e)}'}
    
    def scan_for_catalysts(self, symbols: list, max_workers: int = 10) -> list:
        """
        Scan multiple symbols for potential catalysts
        Returns ranked list of opportunities
        """
        results = []
        
        # Use threading for parallel analysis
        # Optimize: Increase max_workers for speed, early skip unreliable stocks, cache company names
        max_workers = min(max_workers, 32)  # Cap for system stability
        company_name_cache = {}
        def analyze_symbol(symbol):
            try:
                # Early skip: check cache for company name
                if symbol in company_name_cache:
                    company_name = company_name_cache[symbol]
                else:
                    company_name = self.get_company_name(symbol)
                    company_name_cache[symbol] = company_name
                # Get momentum analysis
                momentum = self.detect_explosive_momentum(symbol)
                if 'error' in momentum:
                    return None
                # Get news sentiment
                news = self.get_news_sentiment(symbol)
                # Get earnings data
                earnings = self.detect_earnings_surprise(symbol)
                # Calculate composite catalyst score
                catalyst_score = 0
                # Momentum contribution (40%)
                catalyst_score += momentum.get('momentum_score', 0) * 0.4
                # News sentiment contribution (35%)
                if news.get('catalyst_detected'):
                    catalyst_score += news.get('confidence', 0) * 0.35
                elif abs(news.get('sentiment_score', 0)) > 0.5:
                    catalyst_score += abs(news.get('sentiment_score', 0)) * 20 * 0.35
                # Earnings surprise contribution (25%)
                catalyst_score += earnings.get('surprise_probability', 0) * 0.25
                # Get company name for display
                return {
                    'symbol': symbol,
                    'company_name': company_name,
                    'catalyst_score': catalyst_score,
                    'momentum_data': momentum,
                    'news_data': news,
                    'earnings_data': earnings,
                    'explosive_potential': momentum.get('explosive_move', False),
                    'catalyst_type': news.get('catalyst_type'),
                    'recommendation': 'STRONG BUY' if catalyst_score > 70 else 'BUY' if catalyst_score > 50 else 'WATCH'
                }
            except Exception as e:
                if "No timezone found" not in str(e) and "No data found" not in str(e):
                    print(f"Error analyzing {symbol}: {e}")
                return None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(analyze_symbol, symbol): symbol for symbol in symbols}
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result and result.get('catalyst_score', 0) > 30:
                    results.append(result)
        
        # Sort by catalyst score
        results.sort(key=lambda x: x['catalyst_score'], reverse=True)
        
        return results
    
    def get_breakout_candidates(self, symbols: list, min_momentum_score: int = 40) -> list:
        """
        Get stocks showing technical breakout patterns with catalyst potential
        """
        breakout_candidates = []
        
        for symbol in symbols:
            try:
                momentum = self.detect_explosive_momentum(symbol)
                
                if (momentum.get('momentum_score', 0) >= min_momentum_score and 
                    (momentum.get('breakout_high') or momentum.get('explosive_move'))):
                    
                    breakout_candidates.append({
                        'symbol': symbol,
                        'momentum_score': momentum.get('momentum_score', 0),
                        'price_change_1d': momentum.get('price_change_1d', 0),
                        'volume_spike': momentum.get('volume_spike', 0),
                        'breakout_type': 'explosive_move' if momentum.get('explosive_move') else 'breakout_high',
                        'current_price': momentum.get('current_price', 0)
                    })
                    
            except Exception as e:
                continue
        
        return sorted(breakout_candidates, key=lambda x: x['momentum_score'], reverse=True)

    def detect_unusual_activity(self, symbol: str) -> dict:
        """
        Detect unusual trading activity that might indicate insider knowledge or upcoming catalysts
        """
        try:
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="60d", interval="1d")
            
            if len(hist) < 30:
                return {'error': f'Insufficient data for {symbol}'}
            
            # Calculate unusual activity metrics
            current_volume = hist['Volume'].iloc[-1]
            avg_volume_30d = hist['Volume'].iloc[-30:].mean()
            volume_percentile = (hist['Volume'].iloc[-1] > hist['Volume'].quantile(0.95))
            
            # Price action analysis
            current_price = hist['Close'].iloc[-1]
            price_change = hist['Close'].pct_change().iloc[-1]
            price_volatility = hist['Close'].pct_change().std()
            
            # Look for unusual patterns
            unusual_patterns = []
            
            # Volume spike with price movement
            if current_volume > avg_volume_30d * 3 and abs(price_change) > 0.05:
                unusual_patterns.append('volume_price_spike')
            
            # Sustained volume increase
            recent_volume_avg = hist['Volume'].iloc[-5:].mean()
            if recent_volume_avg > avg_volume_30d * 2:
                unusual_patterns.append('sustained_volume_increase')
            
            # Price compression before breakout
            recent_volatility = hist['Close'].iloc[-10:].pct_change().std()
            if recent_volatility < price_volatility * 0.5 and abs(price_change) > recent_volatility * 2:
                unusual_patterns.append('compression_breakout')
            
            # Calculate unusual activity score
            activity_score = 0
            if current_volume > avg_volume_30d * 5:
                activity_score += 40
            elif current_volume > avg_volume_30d * 3:
                activity_score += 25
            elif current_volume > avg_volume_30d * 2:
                activity_score += 15
            
            if abs(price_change) > 0.1:  # 10%+ move
                activity_score += 30
            elif abs(price_change) > 0.05:  # 5%+ move
                activity_score += 20
            
            if volume_percentile:
                activity_score += 15
            
            if len(unusual_patterns) > 0:
                activity_score += len(unusual_patterns) * 10
            
            return {
                'symbol': symbol,
                'current_volume': current_volume,
                'avg_volume_30d': avg_volume_30d,
                'volume_ratio': current_volume / avg_volume_30d if avg_volume_30d > 0 else 1,
                'price_change': price_change * 100,
                'volume_percentile': volume_percentile,
                'unusual_patterns': unusual_patterns,
                'activity_score': activity_score,
                'unusual_activity_detected': activity_score >= 50
            }
            
        except Exception as e:
            return {'error': f'Error detecting unusual activity for {symbol}: {str(e)}'}
    
    def _generate_synthetic_momentum_data(self, symbol: str) -> dict:
        # REMOVED: Synthetic data generation for production reliability
        return {'error': 'No reliable data available for this symbol. Skipping analysis.'}

    def _generate_synthetic_news(self, symbol: str) -> list:
        # REMOVED: Synthetic news generation for production reliability
        return []
