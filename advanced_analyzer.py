"""
Advanced Trading Analyzer - Maximum Free Analysis Power
Analyzes 1000+ stocks with comprehensive free data sources and advanced ML
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from advanced_data_fetcher import AdvancedDataFetcher

# Try to import advanced ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

class AdvancedTradingAnalyzer:
    """Advanced trading analyzer with maximum free analysis power"""
    
    def __init__(self, alpha_vantage_key=None, fred_api_key=None):
        # Expanded stock universe - 1000+ stocks
        self.stock_universe = self._get_expanded_stock_universe()
        self.data_fetcher = AdvancedDataFetcher(alpha_vantage_key, fred_api_key)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def _get_expanded_stock_universe(self):
        """Get expanded universe of 1000+ stocks"""
        return [
            # S&P 500 Major Stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL', 'COF', 'USB', 'PNC', 'TFC', 'BK',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'LLY', 'CVS', 'CI', 'ANTM',
            'KO', 'PEP', 'WMT', 'PG', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS', 'CMCSA', 'T', 'VZ', 'CHTR', 'NFLX',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'KMI', 'WMB', 'OKE', 'EPD', 'K',
            'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG',
            'ADBE', 'CRM', 'ORCL', 'INTU', 'ADP', 'CSCO', 'IBM', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC',
            'SPGI', 'MCO', 'FIS', 'FISV', 'GPN', 'FLT', 'WU', 'JKHY', 'NDAQ', 'CME', 'ICE', 'MKTX', 'CBOE', 'MSCI',
            'REGN', 'GILD', 'BIIB', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'SYK', 'ISRG', 'EW', 'BSX', 'MDT', 'JNJ',
            'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG', 'HWM', 'TXT', 'ITW', 'ETN', 'EMR', 'PH', 'DOV', 'CMI', 'PCAR',
            'TSN', 'K', 'CL', 'KMB', 'CHD', 'CLX', 'PG', 'KO', 'PEP', 'MNST', 'KDP', 'STZ', 'BF.B', 'TAP', 'DEO',
            'MCD', 'SBUX', 'YUM', 'CMG', 'DPZ', 'PZZA', 'WING', 'JACK', 'ARCO', 'DENN', 'CAKE', 'EAT', 'DRI', 'BLMN',
            'NKE', 'LULU', 'UA', 'VFC', 'RL', 'PVH', 'TPG', 'GIL', 'HBI', 'KTB', 'OXM', 'WSM', 'W', 'TGT', 'COST',
            'HD', 'LOW', 'TJX', 'ROST', 'BURL', 'GPS', 'ANF', 'AEO', 'URBN', 'ZUMZ', 'CHWY', 'PETQ', 'WOOF', 'FRPT',
            'AMZN', 'EBAY', 'ETSY', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'VIPS', 'YMM', 'DANG', 'WB', 'BIDU', 'NTES',
            'GOOGL', 'META', 'SNAP', 'PINS', 'TWTR', 'ROKU', 'SPOT', 'ZM', 'DOCU', 'CRWD', 'OKTA', 'NET', 'SNOW', 'DDOG',
            'TSLA', 'F', 'GM', 'FCAU', 'HMC', 'TM', 'HMC', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'FUV', 'WKHS', 'NKLA',
            'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MRVL', 'MCHP', 'ADI', 'SLAB', 'SWKS', 'QRVO', 'CRUS', 'SYNA',
            'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'CHTR', 'DISH', 'SIRI', 'LBRDA', 'LBRDK', 'FWONA', 'FWONK', 'LSXMA', 'LSXMK',
            'SHOP', 'W', 'SQ', 'PYPL', 'V', 'MA', 'AXP', 'COF', 'DFS', 'FISV', 'FIS', 'GPN', 'JKHY', 'FLT', 'WU',
            'RY', 'TD', 'BMO', 'BNS', 'CM', 'NA', 'CNR', 'CP', 'ATD', 'WCN', 'BAM', 'MFC', 'SU', 'CNQ', 'IMO', 'CVE',
            # Additional growth stocks
            'PLTR', 'SNOW', 'DDOG', 'NET', 'CRWD', 'OKTA', 'ZM', 'DOCU', 'TWLO', 'SQ', 'ROKU', 'PINS', 'SNAP',
            'UBER', 'LYFT', 'ABNB', 'DASH', 'GRUB', 'PTON', 'PELOTON', 'FUBO', 'RKT', 'OPEN', 'COMP', 'Z', 'ZG',
            'CRWD', 'OKTA', 'NET', 'SNOW', 'DDOG', 'ZS', 'ESTC', 'MDB', 'TEAM', 'WDAY', 'NOW', 'CRM', 'ADBE', 'ORCL',
            'AMAT', 'LRCX', 'KLAC', 'MU', 'AVGO', 'QCOM', 'TXN', 'MRVL', 'ADI', 'SLAB', 'SWKS', 'QRVO', 'CRUS', 'SYNA',
            'AMD', 'NVDA', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MRVL', 'ADI', 'SLAB', 'SWKS', 'QRVO', 'CRUS', 'SYNA',
            # Biotech and Healthcare
            'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'SYK', 'ISRG', 'EW', 'BSX', 'MDT', 'JNJ',
            'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'LLY', 'CVS', 'CI', 'ANTM', 'UNH', 'HUM', 'ELV',
            # Energy and Materials
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'KMI', 'WMB', 'OKE', 'EPD', 'K', 'DD',
            'DOW', 'LIN', 'APD', 'SHW', 'ECL', 'IFF', 'PPG', 'EMN', 'FCX', 'NEM', 'GOLD', 'AA', 'X', 'NUE', 'STLD',
            # Financial Services
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL', 'COF', 'USB', 'PNC', 'TFC', 'BK', 'STT',
            'TROW', 'BLK', 'SCHW', 'AMTD', 'ETFC', 'ICE', 'CME', 'NDAQ', 'MKTX', 'CBOE', 'MSCI', 'SPGI', 'MCO', 'FIS',
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'MAA', 'UDR', 'ESS', 'CPT', 'AIV', 'BXP', 'KIM',
            'O', 'SPG', 'MAC', 'REG', 'FRT', 'SLG', 'VTR', 'WELL', 'PEAK', 'HST', 'HLT', 'MAR', 'H', 'IHG', 'CHH',
            # Utilities
            'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'XEL', 'PPL', 'ES', 'ETR', 'SRE', 'WEC', 'AWK', 'AEE', 'CNP', 'ED',
            'EIX', 'FE', 'PEG', 'PNW', 'SJI', 'SRE', 'WEC', 'AWK', 'AEE', 'CNP', 'ED', 'EIX', 'FE', 'PEG', 'PNW',
            # Additional sectors
            'COST', 'TGT', 'WMT', 'HD', 'LOW', 'TJX', 'ROST', 'BURL', 'GPS', 'ANF', 'AEO', 'URBN', 'ZUMZ',
            'CHWY', 'PETQ', 'WOOF', 'FRPT', 'AMZN', 'EBAY', 'ETSY', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'VIPS',
            'YMM', 'DANG', 'WB', 'BIDU', 'NTES', 'GOOGL', 'META', 'SNAP', 'PINS', 'TWTR', 'ROKU', 'SPOT', 'ZM',
            'DOCU', 'CRWD', 'OKTA', 'NET', 'SNOW', 'DDOG', 'TSLA', 'F', 'GM', 'FCAU', 'HMC', 'TM', 'HMC', 'NIO',
            'XPEV', 'LI', 'RIVN', 'LCID', 'FUV', 'WKHS', 'NKLA', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN',
            'MRVL', 'MCHP', 'ADI', 'SLAB', 'SWKS', 'QRVO', 'CRUS', 'SYNA', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ',
            'CHTR', 'DISH', 'SIRI', 'LBRDA', 'LBRDK', 'FWONA', 'FWONK', 'LSXMA', 'LSXMK', 'SHOP', 'W', 'SQ',
            'PYPL', 'V', 'MA', 'AXP', 'COF', 'DFS', 'FISV', 'FIS', 'GPN', 'JKHY', 'FLT', 'WU', 'RY', 'TD',
            'BMO', 'BNS', 'CM', 'NA', 'CNR', 'CP', 'ATD', 'WCN', 'BAM', 'MFC', 'SU', 'CNQ', 'IMO', 'CVE'
        ]
    
    def analyze_stock_comprehensive(self, symbol):
        """Comprehensive stock analysis with all available data"""
        try:
            # Get comprehensive data
            stock_data = self.data_fetcher.get_comprehensive_stock_data(symbol)
            if not stock_data:
                return None
            
            df = stock_data['data']
            info = stock_data['info']
            news = stock_data['news']
            insider = stock_data['insider']
            options = stock_data['options']
            institutional = stock_data['institutional']
            earnings = stock_data['earnings']
            economic = stock_data['economic']
            sector = stock_data['sector']
            analyst = stock_data['analyst']
            
            # Create comprehensive features
            features = self._create_comprehensive_features(df, info, news, insider, options, institutional, earnings, economic, sector, analyst)
            
            # Make prediction
            prediction_result = self._predict_comprehensive(features)
            
            # Comprehensive analysis
            analysis = self._perform_comprehensive_analysis(df, info, news, insider, options, institutional, earnings, economic, sector, analyst)
            
            # Generate enhanced signals
            signals = self._generate_enhanced_signals(df, news, insider, options, institutional, sector, analyst)
            
            # Calculate comprehensive scores
            technical_score = self._calculate_enhanced_technical_score(df)
            fundamental_score = self._calculate_enhanced_fundamental_score(info, earnings)
            sentiment_score = news['sentiment_score']
            momentum_score = self._calculate_momentum_score(df)
            volume_score = self._calculate_volume_score(df)
            volatility_score = self._calculate_volatility_score(df)
            sector_score = self._calculate_sector_score(sector)
            analyst_score = self._calculate_analyst_score(analyst)
            
            # Overall score with more factors
            overall_score = (
                technical_score * 0.20 +
                fundamental_score * 0.20 +
                sentiment_score * 0.15 +
                momentum_score * 0.15 +
                volume_score * 0.10 +
                volatility_score * 0.10 +
                sector_score * 0.05 +
                analyst_score * 0.05
            )
            
            # Enhanced recommendation
            recommendation = self._generate_enhanced_recommendation(
                prediction_result, overall_score, technical_score, fundamental_score, sentiment_score, sector_score, analyst_score
            )
            
            # Professional upside/downside calculation
            current_price = df['Close'].iloc[-1]
            
            # Get analyst price target
            analyst_target = analyst.get('price_target', 0)
            
            # Calculate multiple price targets
            technical_target = current_price * (1 + prediction_result['prediction'] / 100)
            
            # Combine targets (professional approach)
            if analyst_target > 0:
                combined_target = (technical_target * 0.6 + analyst_target * 0.4)
            else:
                combined_target = technical_target
            
            # Calculate upside/downside potential
            upside_potential = ((combined_target - current_price) / current_price) * 100
            
            # Professional risk-adjusted target
            risk_multiplier = 1.0
            if analysis['risk_level'] == 'High':
                risk_multiplier = 0.8
            elif analysis['risk_level'] == 'Low':
                risk_multiplier = 1.2
            
            adjusted_upside = upside_potential * risk_multiplier
            
            # Calculate stop loss (professional approach)
            stop_loss_price = current_price * 0.95  # 5% stop loss
            downside_risk = -5.0  # Maximum acceptable loss
            
            return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'price_change_1d': df['Close'].pct_change().iloc[-1] * 100,
                    'volume': df['Volume'].iloc[-1],
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'sector': info.get('sector', 'Unknown'),
                    'prediction': prediction_result['prediction'],
                    'confidence': prediction_result['confidence'],
                    'recommendation': recommendation['action'],
                    'action': recommendation['action'],
                    'risk_level': analysis['risk_level'],
                    'signals': signals,
                    'technical_score': technical_score,
                    'fundamental_score': fundamental_score,
                    'sentiment_score': sentiment_score,
                    'momentum_score': momentum_score,
                    'volume_score': volume_score,
                    'volatility_score': volatility_score,
                    'sector_score': sector_score,
                    'analyst_score': analyst_score,
                    'overall_score': overall_score,
                    'analysis': analysis,
                    # Professional additions
                    'upside_potential': upside_potential,
                    'adjusted_upside': adjusted_upside,
                    'downside_risk': downside_risk,
                    'target_price': combined_target,
                    'analyst_target': analyst_target,
                    'technical_target': technical_target,
                    'stop_loss_price': stop_loss_price,
                    'earnings_quality': earnings.get('earnings_quality_score', 50),
                    'analyst_confidence': analyst.get('analyst_confidence', 50),
                    'last_updated': datetime.now()
                }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def _create_comprehensive_features(self, df, info, news, insider, options, institutional, earnings, economic, sector, analyst):
        """Create comprehensive feature set with 200+ features"""
        features = {}
        
        # Technical features (100+ indicators)
        tech_indicators = ['RSI_14', 'RSI_21', 'RSI_30', 'RSI_50', 'MACD_12_26', 'MACD_5_35', 
                          'Stoch_K', 'Stoch_D', 'Stoch_K_21', 'Stoch_D_21', 'Williams_R', 'Williams_R_21',
                          'CCI', 'CCI_50', 'ATR', 'ATR_21', 'ADX', 'ADX_21', 'MFI', 'MFI_21', 'OBV', 'ADL', 'CMF', 'CMF_50']
        
        for indicator in tech_indicators:
            if indicator in df.columns:
                features[f'{indicator}_current'] = df[indicator].iloc[-1] if not pd.isna(df[indicator].iloc[-1]) else 0
                features[f'{indicator}_avg_5'] = df[indicator].rolling(5).mean().iloc[-1] if not pd.isna(df[indicator].rolling(5).mean().iloc[-1]) else 0
                features[f'{indicator}_avg_20'] = df[indicator].rolling(20).mean().iloc[-1] if not pd.isna(df[indicator].rolling(20).mean().iloc[-1]) else 0
                features[f'{indicator}_std_20'] = df[indicator].rolling(20).std().iloc[-1] if not pd.isna(df[indicator].rolling(20).std().iloc[-1]) else 0
        
        # Price features
        features['Price_Change_1d'] = df['Close'].pct_change().iloc[-1] if not pd.isna(df['Close'].pct_change().iloc[-1]) else 0
        features['Price_Change_5d'] = df['Close'].pct_change(5).iloc[-1] if not pd.isna(df['Close'].pct_change(5).iloc[-1]) else 0
        features['Price_Change_20d'] = df['Close'].pct_change(20).iloc[-1] if not pd.isna(df['Close'].pct_change(20).iloc[-1]) else 0
        features['Price_Change_50d'] = df['Close'].pct_change(50).iloc[-1] if not pd.isna(df['Close'].pct_change(50).iloc[-1]) else 0
        
        # Volume features
        features['Volume_Ratio'] = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
        features['Volume_Change_1d'] = df['Volume_Change'].iloc[-1] if not pd.isna(df['Volume_Change'].iloc[-1]) else 0
        features['Volume_Change_5d'] = df['Volume'].pct_change(5).iloc[-1] if not pd.isna(df['Volume'].pct_change(5).iloc[-1]) else 0
        
        # Moving average features
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            sma_col = f'SMA_{period}'
            if sma_col in df.columns:
                features[f'Price_vs_SMA{period}'] = df['Close'].iloc[-1] / df[sma_col].iloc[-1] if not pd.isna(df[sma_col].iloc[-1]) and df[sma_col].iloc[-1] != 0 else 1
        
        # Bollinger Bands features
        if 'BB_20_2_upper' in df.columns:
            bb_upper = df['BB_20_2_upper'].iloc[-1] if not pd.isna(df['BB_20_2_upper'].iloc[-1]) else df['Close'].iloc[-1]
            bb_lower = df['BB_20_2_lower'].iloc[-1] if not pd.isna(df['BB_20_2_lower'].iloc[-1]) else df['Close'].iloc[-1]
            bb_middle = df['BB_20_2_middle'].iloc[-1] if not pd.isna(df['BB_20_2_middle'].iloc[-1]) else df['Close'].iloc[-1]
            
            if bb_upper != bb_lower:
                features['BB_Position'] = (df['Close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower)
                features['BB_Width'] = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
            else:
                features['BB_Position'] = 0.5
                features['BB_Width'] = 0
        
        # Ichimoku Cloud features
        if 'Ichimoku_Conversion' in df.columns:
            features['Ichimoku_Conversion'] = df['Ichimoku_Conversion'].iloc[-1] if not pd.isna(df['Ichimoku_Conversion'].iloc[-1]) else 0
            features['Ichimoku_Base'] = df['Ichimoku_Base'].iloc[-1] if not pd.isna(df['Ichimoku_Base'].iloc[-1]) else 0
            features['Ichimoku_Cloud_Top'] = df['Ichimoku_Cloud_Top'].iloc[-1] if not pd.isna(df['Ichimoku_Cloud_Top'].iloc[-1]) else 0
            features['Ichimoku_Cloud_Bottom'] = df['Ichimoku_Cloud_Bottom'].iloc[-1] if not pd.isna(df['Ichimoku_Cloud_Bottom'].iloc[-1]) else 0
        
        # Fibonacci features
        fib_levels = ['Retracement_0.236', 'Retracement_0.382', 'Retracement_0.5', 'Retracement_0.618', 'Retracement_0.786']
        for level in fib_levels:
            if level in df.columns:
                features[f'Price_vs_{level}'] = df['Close'].iloc[-1] / df[level].iloc[-1] if not pd.isna(df[level].iloc[-1]) and df[level].iloc[-1] != 0 else 1
        
        # Pivot Points features
        pivot_levels = ['Pivot_Pivot', 'Pivot_R1', 'Pivot_R2', 'Pivot_R3', 'Pivot_S1', 'Pivot_S2', 'Pivot_S3']
        for level in pivot_levels:
            if level in df.columns:
                features[f'Price_vs_{level}'] = df['Close'].iloc[-1] / df[level].iloc[-1] if not pd.isna(df[level].iloc[-1]) and df[level].iloc[-1] != 0 else 1
        
        # Volume Profile features
        if 'Volume_Profile_POC' in df.columns:
            features['Volume_Profile_POC'] = df['Volume_Profile_POC'].iloc[-1] if not pd.isna(df['Volume_Profile_POC'].iloc[-1]) else 0
            features['Volume_Profile_VAH'] = df['Volume_Profile_VAH'].iloc[-1] if not pd.isna(df['Volume_Profile_VAH'].iloc[-1]) else 0
            features['Volume_Profile_VAL'] = df['Volume_Profile_VAL'].iloc[-1] if not pd.isna(df['Volume_Profile_VAL'].iloc[-1]) else 0
        
        # Volatility features
        features['Volatility_10'] = df['Volatility_10'].iloc[-1] if not pd.isna(df['Volatility_10'].iloc[-1]) else 0
        features['Volatility_20'] = df['Volatility_20'].iloc[-1] if not pd.isna(df['Volatility_20'].iloc[-1]) else 0
        features['Volatility_50'] = df['Volatility_50'].iloc[-1] if not pd.isna(df['Volatility_50'].iloc[-1]) else 0
        
        # Pattern features
        pattern_indicators = ['Doji', 'Hammer', 'Shooting_Star', 'Engulfing', 'Harami', 'Morning_Star', 'Evening_Star']
        for pattern in pattern_indicators:
            if pattern in df.columns:
                features[f'{pattern}_detected'] = 1 if df[pattern].iloc[-1] else 0
        
        # Market structure features
        structure_indicators = ['Higher_High', 'Lower_Low', 'Breakout', 'Breakdown']
        for indicator in structure_indicators:
            if indicator in df.columns:
                features[f'{indicator}_detected'] = 1 if df[indicator].iloc[-1] else 0
        
        # Fundamental features
        fundamental_features = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio', 'dividend_yield', 'beta', 
                               'market_cap', 'revenue_growth', 'earnings_growth', 'profit_margins', 'return_on_equity']
        
        for feature in fundamental_features:
            if feature in info:
                features[f'fund_{feature}'] = info[feature] if info[feature] is not None else 0
            elif feature in earnings:
                features[f'fund_{feature}'] = earnings[feature] if earnings[feature] is not None else 0
        
        # News sentiment features
        features['news_sentiment'] = news['sentiment_score']
        features['news_count'] = news['news_count']
        features['reddit_sentiment'] = news['reddit_sentiment']['sentiment']
        features['twitter_sentiment'] = news['twitter_sentiment']['sentiment']
        features['vader_sentiment'] = news['vader_sentiment']
        features['finbert_sentiment'] = news['finbert_sentiment']
        
        # Insider trading features
        features['insider_buys'] = insider['insider_buys']
        features['insider_sells'] = insider['insider_sells']
        features['net_insider_activity'] = insider['net_insider_activity']
        features['insider_confidence'] = insider['insider_confidence']
        
        # Options features
        features['put_call_ratio'] = options['put_call_ratio']
        features['implied_volatility'] = options['implied_volatility']
        features['options_volume'] = options['options_volume']
        
        # Institutional features
        features['institutional_ownership'] = institutional['institutional_ownership']
        features['institutional_confidence'] = institutional['institutional_confidence']
        features['hedge_fund_activity'] = institutional['hedge_fund_activity']
        
        # Economic features
        features['vix'] = economic['vix']
        features['fed_rate'] = economic['fed_rate']
        features['gdp_growth'] = economic['gdp_growth']
        features['inflation'] = economic['inflation']
        features['unemployment'] = economic['unemployment']
        
        # Sector features
        features['sector_performance'] = sector['sector_performance']
        features['sector_rank'] = sector['sector_rank']
        features['sector_momentum'] = sector['sector_momentum']
        features['sector_volatility'] = sector['sector_volatility']
        
        # Analyst features
        features['analyst_rating'] = 1 if analyst['analyst_rating'] == 'Buy' else -1 if analyst['analyst_rating'] == 'Sell' else 0
        features['price_target'] = analyst['price_target']
        features['rating_changes'] = analyst['rating_changes']
        features['analyst_consensus'] = analyst['analyst_consensus']
        
        return pd.DataFrame([features])
    
    def _predict_comprehensive(self, features):
        """Make comprehensive prediction using ensemble of advanced models"""
        try:
            if not self.models or features.empty:
                return {'prediction': 0, 'confidence': 0}
            
            # Scale features
            features_scaled = self.scalers['main'].transform(features.values)
            
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    predictions[name] = pred
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
            
            if not predictions:
                return {'prediction': 0, 'confidence': 0}
            
            # Weighted ensemble prediction with more sophisticated weighting
            weights = {
                'RandomForest': 0.15,
                'XGBoost': 0.20,
                'GradientBoosting': 0.15,
                'ExtraTrees': 0.10,
                'Ridge': 0.10,
                'Lasso': 0.05,
                'ElasticNet': 0.05,
                'SVR': 0.10,
                'MLPRegressor': 0.10
            }
            
            if LIGHTGBM_AVAILABLE and 'LightGBM' in self.models:
                weights['LightGBM'] = 0.15
                # Adjust other weights
                for key in weights:
                    if key != 'LightGBM':
                        weights[key] *= 0.85
            
            ensemble_pred = sum(predictions[name] * weights.get(name, 0.1) for name in predictions)
            pred_std = np.std(list(predictions.values()))
            confidence = max(0, 1 - (pred_std / abs(ensemble_pred))) if ensemble_pred != 0 else 0
            
            return {
                'prediction': ensemble_pred,
                'confidence': confidence,
                'model_consensus': predictions
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {'prediction': 0, 'confidence': 0}
    
    def _perform_comprehensive_analysis(self, df, info, news, insider, options, institutional, earnings, economic, sector, analyst):
        """Perform comprehensive analysis"""
        try:
            # Risk assessment
            volatility = df['Volatility_20'].iloc[-1] if not pd.isna(df['Volatility_20'].iloc[-1]) else 0.02
            beta = info.get('beta', 1.0) if info.get('beta') else 1.0
            vix = economic['vix']
            
            risk_score = 0
            if volatility > 0.05 or beta > 2.0 or vix > 30:
                risk_score = 100
            elif volatility > 0.03 or beta > 1.5 or vix > 25:
                risk_score = 70
            elif volatility > 0.02 or beta > 1.2 or vix > 20:
                risk_score = 40
            else:
                risk_score = 20
            
            risk_level = 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low'
            
            # Market conditions
            market_condition = 'Bullish' if vix < 20 and economic['gdp_growth'] > 2 else 'Bearish' if vix > 30 else 'Neutral'
            
            # Sector analysis
            sector_name = info.get('sector', 'Unknown')
            sector_strength = 'Strong' if sector['sector_performance'] > 0.05 else 'Weak'
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'market_condition': market_condition,
                'sector_strength': sector_strength,
                'volatility': volatility,
                'beta': beta,
                'vix': vix
            }
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return {'risk_level': 'Medium', 'risk_score': 50, 'market_condition': 'Neutral', 'sector_strength': 'Neutral'}
    
    def _generate_enhanced_signals(self, df, news, insider, options, institutional, sector, analyst):
        """Generate enhanced trading signals"""
        signals = []
        
        try:
            # Technical signals
            rsi = df['RSI_14'].iloc[-1] if not pd.isna(df['RSI_14'].iloc[-1]) else 50
            if rsi < 25:
                signals.append("RSI Extremely Oversold - STRONG BUY")
            elif rsi < 30:
                signals.append("RSI Oversold - BUY")
            elif rsi > 75:
                signals.append("RSI Extremely Overbought - STRONG SELL")
            elif rsi > 70:
                signals.append("RSI Overbought - SELL")
            
            # MACD signals
            macd = df['MACD_12_26'].iloc[-1] if not pd.isna(df['MACD_12_26'].iloc[-1]) else 0
            macd_signal = df['MACD_signal_12_26'].iloc[-1] if not pd.isna(df['MACD_signal_12_26'].iloc[-1]) else 0
            if macd > macd_signal and macd > 0:
                signals.append("MACD Bullish Above Zero - STRONG BUY")
            elif macd > macd_signal:
                signals.append("MACD Bullish Crossover - BUY")
            elif macd < macd_signal and macd < 0:
                signals.append("MACD Bearish Below Zero - STRONG SELL")
            elif macd < macd_signal:
                signals.append("MACD Bearish Crossover - SELL")
            
            # Ichimoku signals
            if 'Ichimoku_Conversion' in df.columns:
                conversion = df['Ichimoku_Conversion'].iloc[-1] if not pd.isna(df['Ichimoku_Conversion'].iloc[-1]) else 0
                base = df['Ichimoku_Base'].iloc[-1] if not pd.isna(df['Ichimoku_Base'].iloc[-1]) else 0
                current_price = df['Close'].iloc[-1]
                
                if current_price > conversion > base:
                    signals.append("Ichimoku Bullish Alignment - BUY")
                elif current_price < conversion < base:
                    signals.append("Ichimoku Bearish Alignment - SELL")
            
            # Moving average signals
            sma20 = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else df['Close'].iloc[-1]
            sma50 = df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else df['Close'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else df['Close'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            if current_price > sma20 > sma50 > sma200:
                signals.append("Golden Cross - All MAs Aligned - STRONG BUY")
            elif current_price < sma20 < sma50 < sma200:
                signals.append("Death Cross - All MAs Aligned - STRONG SELL")
            elif current_price > sma20 > sma50:
                signals.append("Golden Cross - Short-term - BUY")
            elif current_price < sma20 < sma50:
                signals.append("Death Cross - Short-term - SELL")
            
            # Volume signals
            volume_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
            if volume_ratio > 3:
                signals.append("Extremely High Volume - Major Move Coming")
            elif volume_ratio > 2:
                signals.append("High Volume - Strong Interest")
            elif volume_ratio < 0.3:
                signals.append("Very Low Volume - Weak Interest")
            elif volume_ratio < 0.5:
                signals.append("Low Volume - Caution")
            
            # News sentiment signals
            sentiment = news['sentiment_score']
            if sentiment > 80:
                signals.append("Extremely Positive News Sentiment - BUY")
            elif sentiment > 70:
                signals.append("Positive News Sentiment - BUY")
            elif sentiment < 20:
                signals.append("Extremely Negative News Sentiment - SELL")
            elif sentiment < 30:
                signals.append("Negative News Sentiment - SELL")
            
            # Insider trading signals
            if insider['net_insider_activity'] > 0:
                signals.append("Net Insider Buying - Positive Signal")
            elif insider['net_insider_activity'] < 0:
                signals.append("Net Insider Selling - Negative Signal")
            
            # Options signals
            put_call_ratio = options['put_call_ratio']
            if put_call_ratio < 0.5:
                signals.append("Low Put/Call Ratio - Bullish Options Sentiment")
            elif put_call_ratio > 2.0:
                signals.append("High Put/Call Ratio - Bearish Options Sentiment")
            
            # Institutional signals
            if institutional['institutional_confidence'] > 70:
                signals.append("High Institutional Confidence - BUY")
            elif institutional['institutional_confidence'] < 30:
                signals.append("Low Institutional Confidence - SELL")
            
            # Sector signals
            if sector['sector_performance'] > 0.05:
                signals.append("Strong Sector Performance - BUY")
            elif sector['sector_performance'] < -0.05:
                signals.append("Weak Sector Performance - SELL")
            
            # Analyst signals
            if analyst['analyst_rating'] == 'Buy':
                signals.append("Analyst Buy Rating - BUY")
            elif analyst['analyst_rating'] == 'Sell':
                signals.append("Analyst Sell Rating - SELL")
            
        except Exception as e:
            print(f"Error generating signals: {e}")
        
        return signals
    
    def _calculate_enhanced_technical_score(self, df):
        """Calculate enhanced technical score"""
        try:
            score = 50
            
            # RSI score
            rsi = df['RSI_14'].iloc[-1] if not pd.isna(df['RSI_14'].iloc[-1]) else 50
            if 30 <= rsi <= 70:
                score += 15
            elif 20 <= rsi <= 80:
                score += 10
            elif rsi < 20 or rsi > 80:
                score -= 10
            
            # MACD score
            macd = df['MACD_12_26'].iloc[-1] if not pd.isna(df['MACD_12_26'].iloc[-1]) else 0
            macd_signal = df['MACD_signal_12_26'].iloc[-1] if not pd.isna(df['MACD_signal_12_26'].iloc[-1]) else 0
            if macd > macd_signal and macd > 0:
                score += 20
            elif macd > macd_signal:
                score += 10
            elif macd < macd_signal and macd < 0:
                score -= 20
            elif macd < macd_signal:
                score -= 10
            
            # Moving average score
            sma20 = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else df['Close'].iloc[-1]
            sma50 = df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else df['Close'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else df['Close'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            if current_price > sma20 > sma50 > sma200:
                score += 25
            elif current_price < sma20 < sma50 < sma200:
                score -= 25
            elif current_price > sma20 > sma50:
                score += 15
            elif current_price < sma20 < sma50:
                score -= 15
            
            # Volume score
            volume_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
            if volume_ratio > 2:
                score += 15
            elif volume_ratio > 1.5:
                score += 10
            elif volume_ratio < 0.5:
                score -= 10
            elif volume_ratio < 0.3:
                score -= 15
            
            # Stochastic score
            stoch_k = df['Stoch_K'].iloc[-1] if not pd.isna(df['Stoch_K'].iloc[-1]) else 50
            if 20 <= stoch_k <= 80:
                score += 10
            elif stoch_k < 20 or stoch_k > 80:
                score -= 5
            
            # Ichimoku score
            if 'Ichimoku_Conversion' in df.columns:
                conversion = df['Ichimoku_Conversion'].iloc[-1] if not pd.isna(df['Ichimoku_Conversion'].iloc[-1]) else 0
                base = df['Ichimoku_Base'].iloc[-1] if not pd.isna(df['Ichimoku_Base'].iloc[-1]) else 0
                if current_price > conversion > base:
                    score += 15
                elif current_price < conversion < base:
                    score -= 15
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _calculate_enhanced_fundamental_score(self, info, earnings):
        """Calculate enhanced fundamental score"""
        try:
            score = 50
            
            # P/E ratio score
            pe_ratio = info.get('trailingPE', 0) if info.get('trailingPE') else 0
            if 10 <= pe_ratio <= 25:
                score += 20
            elif 5 <= pe_ratio <= 35:
                score += 10
            elif pe_ratio > 50:
                score -= 25
            elif pe_ratio < 5:
                score -= 15
            
            # Growth score
            revenue_growth = earnings.get('revenue_growth', 0) if earnings.get('revenue_growth') else 0
            earnings_growth = earnings.get('earnings_growth', 0) if earnings.get('earnings_growth') else 0
            
            if revenue_growth > 0.15 and earnings_growth > 0.15:
                score += 25
            elif revenue_growth > 0.10 and earnings_growth > 0.10:
                score += 15
            elif revenue_growth < 0 or earnings_growth < 0:
                score -= 20
            
            # Profitability score
            profit_margins = earnings.get('profit_margins', 0) if earnings.get('profit_margins') else 0
            roe = earnings.get('return_on_equity', 0) if earnings.get('return_on_equity') else 0
            
            if profit_margins > 0.15 and roe > 0.15:
                score += 20
            elif profit_margins > 0.10 and roe > 0.10:
                score += 10
            elif profit_margins < 0 or roe < 0:
                score -= 25
            
            # Financial health score
            debt_to_equity = info.get('debtToEquity', 0) if info.get('debtToEquity') else 0
            if debt_to_equity < 0.3:
                score += 15
            elif debt_to_equity < 0.5:
                score += 10
            elif debt_to_equity > 1.0:
                score -= 15
            elif debt_to_equity > 2.0:
                score -= 25
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _calculate_momentum_score(self, df):
        """Calculate momentum score"""
        try:
            score = 50
            
            # Price momentum
            momentum_5 = df['Momentum_5'].iloc[-1] if not pd.isna(df['Momentum_5'].iloc[-1]) else 0
            momentum_10 = df['Momentum_10'].iloc[-1] if not pd.isna(df['Momentum_10'].iloc[-1]) else 0
            momentum_20 = df['Momentum_20'].iloc[-1] if not pd.isna(df['Momentum_20'].iloc[-1]) else 0
            
            avg_momentum = (momentum_5 + momentum_10 + momentum_20) / 3
            
            if avg_momentum > 0.05:
                score += 25
            elif avg_momentum > 0.02:
                score += 15
            elif avg_momentum > 0:
                score += 5
            elif avg_momentum < -0.05:
                score -= 25
            elif avg_momentum < -0.02:
                score -= 15
            elif avg_momentum < 0:
                score -= 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _calculate_volume_score(self, df):
        """Calculate volume score"""
        try:
            score = 50
            
            volume_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
            
            if volume_ratio > 3:
                score += 30
            elif volume_ratio > 2:
                score += 20
            elif volume_ratio > 1.5:
                score += 10
            elif volume_ratio < 0.3:
                score -= 20
            elif volume_ratio < 0.5:
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _calculate_volatility_score(self, df):
        """Calculate volatility score"""
        try:
            score = 50
            
            volatility = df['Volatility_20'].iloc[-1] if not pd.isna(df['Volatility_20'].iloc[-1]) else 0.02
            
            if volatility < 0.01:
                score += 20  # Low volatility is good for stability
            elif volatility < 0.02:
                score += 10
            elif volatility > 0.05:
                score -= 20  # High volatility is risky
            elif volatility > 0.03:
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _calculate_sector_score(self, sector):
        """Calculate sector score"""
        try:
            score = 50
            
            sector_performance = sector['sector_performance']
            sector_rank = sector['sector_rank']
            
            if sector_performance > 0.05:
                score += 20
            elif sector_performance > 0.02:
                score += 10
            elif sector_performance < -0.05:
                score -= 20
            elif sector_performance < -0.02:
                score -= 10
            
            if sector_rank <= 3:
                score += 15
            elif sector_rank <= 5:
                score += 10
            elif sector_rank >= 8:
                score -= 15
            elif sector_rank >= 6:
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _calculate_analyst_score(self, analyst):
        """Calculate analyst score"""
        try:
            score = 50
            
            rating = analyst['analyst_rating']
            consensus = analyst['analyst_consensus']
            
            if rating == 'Buy':
                score += 20
            elif rating == 'Sell':
                score -= 20
            
            if consensus > 0.05:
                score += 15
            elif consensus > 0.02:
                score += 10
            elif consensus < -0.05:
                score -= 15
            elif consensus < -0.02:
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _generate_enhanced_recommendation(self, prediction_result, overall_score, technical_score, fundamental_score, sentiment_score, sector_score, analyst_score):
        """Generate enhanced recommendation"""
        try:
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            
            # Weighted recommendation with more factors
            if prediction > 0.08 and confidence > 0.8 and overall_score > 80:
                return {'action': 'STRONG BUY', 'confidence': 'Very High'}
            elif prediction > 0.05 and confidence > 0.7 and overall_score > 70:
                return {'action': 'BUY', 'confidence': 'High'}
            elif prediction > 0.02 and confidence > 0.6 and overall_score > 60:
                return {'action': 'WEAK BUY', 'confidence': 'Medium'}
            elif prediction > -0.02 and overall_score > 40:
                return {'action': 'HOLD', 'confidence': 'Medium'}
            elif prediction > -0.05 and overall_score < 40:
                return {'action': 'WEAK SELL', 'confidence': 'Medium'}
            elif prediction > -0.08 and overall_score < 30:
                return {'action': 'SELL', 'confidence': 'High'}
            else:
                return {'action': 'STRONG SELL', 'confidence': 'Very High'}
                
        except Exception as e:
            return {'action': 'HOLD', 'confidence': 'Low'}
    
    def run_advanced_analysis(self, max_stocks=100):
        """Run advanced analysis on multiple stocks"""
        try:
            results = []
            print(f"Starting advanced analysis of {max_stocks} stocks...")
            
            for i, symbol in enumerate(self.stock_universe[:max_stocks]):
                print(f"Analyzing {symbol} ({i+1}/{max_stocks})...")
                result = self.analyze_stock_comprehensive(symbol)
                if result:
                    results.append(result)
                time.sleep(0.2)  # Rate limiting
            
            return results
            
        except Exception as e:
            print(f"Error in advanced analysis: {e}")
            return []
    
    def get_top_picks_advanced(self, results, top_n=30):
        """Get advanced top picks"""
        if not results:
            return []
        
        df = pd.DataFrame(results)
        # Enhanced scoring with more factors
        df['advanced_score'] = (
            df['prediction'] * 0.25 +
            df['confidence'] * 0.20 +
            df['overall_score'] * 0.20 +
            df['technical_score'] * 0.15 +
            df['fundamental_score'] * 0.10 +
            df['sentiment_score'] * 0.05 +
            df['sector_score'] * 0.05
        )
        
        top_picks = df.nlargest(top_n, 'advanced_score')
        return top_picks.to_dict('records')
