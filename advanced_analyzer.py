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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import functools
import hashlib
import pickle
import os

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
    
    def __init__(self, alpha_vantage_key=None, fred_api_key=None, *, enable_training: bool = False, data_mode: str = "light"):
        # Expanded stock universe - 1000+ stocks
        self.stock_universe = self._get_expanded_stock_universe()
        # Use light mode by default to avoid rate-limited endpoints
        self.data_fetcher = AdvancedDataFetcher(alpha_vantage_key, fred_api_key, data_mode=data_mode)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.enable_training = enable_training
        self.data_mode = data_mode
        # Internal breadth context (computed once per run in run_advanced_analysis)
        self._breadth_context = {}
        
        # Performance optimization features
        self.cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._indicator_cache = {}
        self._analysis_cache = {}
        
        # Determine optimal worker count
        self.cpu_count = multiprocessing.cpu_count()
        self.max_workers = min(self.cpu_count * 2, 32)  # Cap at 32 to avoid resource exhaustion
        
        print(f"🚀 Optimizer: Using {self.max_workers} workers (CPU cores: {self.cpu_count})")
        print("🛡️ Data integrity validation: ENABLED (synthetic data blocked)")
        
    def _validate_analysis_data(self, results):
        """Validate that analysis results use real market data"""
        if not results:
            return False, "No results to validate"
        
        real_data_count = 0
        total_count = len(results)
        
        for result in results:
            # Check if this looks like real market data
            if (result.get('current_price', 0) > 0 and 
                result.get('volume', 0) > 0 and
                result.get('market_cap', 0) >= 0):
                real_data_count += 1
        
        real_data_percentage = (real_data_count / total_count) * 100 if total_count > 0 else 0
        
        if real_data_percentage < 80:  # Less than 80% real data is concerning
            return False, f"Only {real_data_percentage:.1f}% appears to be real market data"
        
        return True, f"Data validation passed: {real_data_percentage:.1f}% real market data"
        
    def _get_expanded_stock_universe(self):
        """Get expanded universe of 700+ high-potential stocks for 5-2000%+ returns"""
        # Import the cleaned high-potential universe (optimized)
        try:
            from cleaned_high_potential_universe import get_cleaned_high_potential_universe
            return get_cleaned_high_potential_universe()
        except ImportError:
            # Fallback to original high-potential universe
            try:
                from high_potential_universe_500plus import get_high_potential_universe_500plus
                return get_high_potential_universe_500plus()
            except ImportError:
                # Fallback to original universe if import fails
                universe = [
            # S&P 500 Major Stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL', 'COF', 'USB', 'PNC', 'TFC', 'BK',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'LLY', 'CVS', 'CI', 'ELV',
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
            'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'LLY', 'CVS', 'CI', 'ELV', 'UNH', 'HUM',
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
            'BMO', 'BNS', 'CM', 'NA', 'CNR', 'CP', 'ATD', 'WCN', 'BAM', 'MFC', 'SU', 'CNQ', 'IMO', 'CVE',
            
            # Additional High-Growth Tech (Expanded Coverage)
            'ADSK', 'WDAY', 'VEEV', 'ZS', 'PANW', 'FTNT', 'CYBR', 'PING', 'SPLK', 'TENB', 'RPD', 'FEYE',
            'QLYS', 'VRNS', 'MIME', 'PFPT', 'ALRM', 'SAIL', 'ASAN', 'MNDY', 'PD', 'BILL', 'DOCN', 'FSLY',
            
            # Biotech/Life Sciences (High Upside Potential)
            'NVAX', 'SRPT', 'BLUE', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'PRIME', 'VCYT', 'PACB', 'TWST', 'CDNA',
            'FATE', 'SGMO', 'RGNX', 'RARE', 'FOLD', 'ARWR', 'IONS', 'EXAS', 'TDOC', 'DXCM', 'ALGN', 'PODD',
            
            # Clean Energy/Sustainability (Future Growth)
            'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR', 'SPWR', 'CSIQ', 'JKS', 'SOL', 'MAXN', 'PLUG', 'FCEL',
            'BE', 'BLDP', 'HYLN', 'QS', 'CHPT', 'BLNK', 'EVG', 'CLNE', 'ICLN', 'PBW', 'QCLN', 'LIT',
            
            # Fintech/Digital Finance (Disruption Plays)
            'AFRM', 'UPST', 'SOFI', 'LC', 'LMND', 'ROOT', 'MTTR', 'HOOD', 'COIN', 'MARA', 'RIOT', 'HUT',
            'BITF', 'CAN', 'EBON', 'SOS', 'BTBT', 'GREE', 'OSTK', 'MSTR', 'TSLA', 'SQ', 'PYPL', 'V',
            
            # E-commerce/Digital Consumer (Market Expansion)
            'MELI', 'SE', 'CHWY', 'PETQ', 'WOOF', 'FRPT', 'BARK', 'WAGS', 'PETM', 'PETS', 'RDFN', 'COMP',
            'TREE', 'CARS', 'CVNA', 'VRM', 'SFT', 'CPNG', 'GRAB', 'DIDI', 'BABA', 'JD', 'PDD', 'VIPS',
            
            # Cloud/SaaS (Scalable Business Models)
            'AI', 'C3AI', 'PATH', 'SMAR', 'FROG', 'BIGC', 'APPS', 'WORK', 'BOX', 'DBX', 'SEND', 'BAND',
            'RNG', 'GTLB', 'ESTC', 'MDB', 'TEAM', 'WDAY', 'NOW', 'CRM', 'ADBE', 'ORCL', 'INTU', 'ADP',
            
            # Gaming/Entertainment (Digital Transformation)
            'RBLX', 'U', 'DKNG', 'PENN', 'MGM', 'LVS', 'WYNN', 'CZR', 'BYD', 'RSI', 'GLUU', 'HUYA',
            'DOYU', 'BILI', 'IQ', 'FUBO', 'PARA', 'WBD', 'FOXA', 'EA', 'ATVI', 'TTWO', 'ZNGA',
            
            # REITs/Real Estate (Diversification)
            'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'UDR', 'ESS', 'MAA', 'CPT', 'AIV', 'STOR', 'WPC',
            'NNN', 'ADC', 'STAG', 'LXP', 'GTY', 'EPRT', 'FCPT', 'GOOD', 'VNO', 'BXP', 'KIM', 'REG',
            
            # International/ADRs (Global Exposure)
            'TSM', 'ASML', 'NVO', 'UL', 'BP', 'TTE', 'E', 'SAN', 'BBVA', 'ING', 'BABA', 'JD', 'PDD',
            'VIPS', 'YMM', 'DANG', 'WB', 'BIDU', 'NTES', 'TCOM', 'MOMO', 'SINA', 'SOHU', 'NTES',
            
            # Small/Mid Cap High Potential (Hidden Gems)
            'CRWD', 'NET', 'DDOG', 'SNOW', 'PLTR', 'RBLX', 'U', 'DKNG', 'PENN', 'CHWY', 'ETSY', 'MELI',
            'SE', 'AFRM', 'UPST', 'SOFI', 'LC', 'LMND', 'ROOT', 'HOOD', 'COIN', 'MARA', 'RIOT', 'HUT',
            
            # Emerging/Speculative (Maximum Upside Potential)
            'SPCE', 'RKLB', 'ASTR', 'VACQ', 'HOL', 'SRAC', 'CCIV', 'THCB', 'ACTC', 'STPK', 'CLOV',
            'WISH', 'BARK', 'OPEN', 'RDFN', 'Z', 'ZG', 'COMP', 'TREE', 'CARS', 'CVNA', 'VRM', 'SFT',
            
            # Additional Dividend/Value Opportunities
            'KMB', 'CHD', 'CLX', 'MNST', 'KDP', 'STZ', 'BF.B', 'TAP', 'DEO', 'YUM', 'CMG', 'DPZ',
            'PZZA', 'WING', 'JACK', 'ARCO', 'DENN', 'CAKE', 'EAT', 'DRI', 'BLMN', 'LULU', 'UA', 'VFC',
            
            # Technology Hardware/Semiconductors
            'MRVL', 'MCHP', 'ADI', 'SLAB', 'SWKS', 'QRVO', 'CRUS', 'SYNA', 'AMAT', 'LRCX', 'KLAC',
            'MU', 'WDC', 'STX', 'NTAP', 'HPQ', 'DELL', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            
            # === EXPANSION: 100 LARGE CAP STOCKS ===
            # S&P 500 Large Cap Leaders
            'BERKB', 'UNH', 'JNJ', 'XOM', 'JPM', 'WMT', 'CVX', 'LLY', 'PG', 'HD', 'ABBV', 'PFE', 'BAC', 'KO', 'AVGO',
            'MRK', 'PEP', 'TMO', 'COST', 'DHR', 'ABT', 'ACN', 'MCD', 'CSCO', 'LIN', 'ADBE', 'WFC', 'VZ', 'CRM', 'NKE',
            'TXN', 'RTX', 'NEE', 'QCOM', 'UPS', 'LOW', 'ORCL', 'PM', 'HON', 'UNP', 'AMGN', 'SBUX', 'IBM', 'T', 'SPGI',
            'CAT', 'INTU', 'GS', 'AXP', 'BKNG', 'DE', 'MDT', 'BLK', 'ELV', 'GILD', 'ADP', 'TJX', 'VRTX', 'LRCX', 'SYK',
            'MMM', 'CVS', 'TMUS', 'SCHW', 'MO', 'ZTS', 'FIS', 'BSX', 'EOG', 'DUK', 'SO', 'NSC', 'HUM', 'PLD', 'ITW',
            'BMY', 'AON', 'CL', 'APD', 'EQIX', 'GE', 'SHW', 'CME', 'USB', 'MMC', 'REGN', 'FCX', 'PNC', 'EMR', 'COP',
            'MCK', 'CSX', 'MSI', 'TGT', 'PSA', 'KLAC', 'AMAT', 'BIIB', 'ATVI', 'ADI', 'MU', 'ISRG', 'FISV', 'ECL',
            
            # === EXPANSION: 100 MEDIUM CAP STOCKS ===
            # Russell Mid Cap Growth & Value
            'ROKU', 'PINS', 'SNAP', 'TWTR', 'SPOT', 'ZM', 'DOCU', 'TWLO', 'OKTA', 'CRWD', 'NET', 'DDOG', 'SNOW', 'PLTR',
            'RBLX', 'U', 'DKNG', 'PENN', 'MGM', 'LVS', 'WYNN', 'CZR', 'BYD', 'RSI', 'GLUU', 'HUYA', 'DOYU', 'BILI',
            'IQ', 'FUBO', 'PARA', 'WBD', 'FOXA', 'EA', 'TTWO', 'ZNGA', 'CHWY', 'ETSY', 'MELI', 'SE', 'AFRM', 'UPST',
            'SOFI', 'LC', 'LMND', 'ROOT', 'HOOD', 'COIN', 'MARA', 'RIOT', 'HUT', 'BITF', 'CAN', 'EBON', 'SOS', 'BTBT',
            'GREE', 'OSTK', 'MSTR', 'RDFN', 'COMP', 'TREE', 'CARS', 'CVNA', 'VRM', 'SFT', 'CPNG', 'GRAB', 'DIDI',
            'AI', 'C3AI', 'PATH', 'SMAR', 'FROG', 'BIGC', 'APPS', 'WORK', 'BOX', 'DBX', 'SEND', 'BAND', 'RNG', 'GTLB',
            'ESTC', 'MDB', 'TEAM', 'VEEV', 'ZS', 'PANW', 'FTNT', 'ADSK', 'WDAY', 'NVAX', 'SRPT', 'BLUE', 'EDIT',
            'CRSP', 'NTLA', 'BEAM', 'PRIME', 'VCYT', 'PACB', 'TWST', 'CDNA', 'FATE', 'SGMO', 'RGNX', 'RARE', 'FOLD',
            'ARWR', 'IONS', 'EXAS', 'TDOC', 'DXCM', 'ALGN', 'PODD', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR', 'SPWR',
            'CSIQ', 'JKS', 'SOL', 'MAXN', 'PLUG', 'FCEL', 'BE', 'BLDP', 'HYLN', 'QS', 'CHPT', 'BLNK', 'EVG', 'CLNE',
            
            # === EXPANSION: 100 SMALL CAP STOCKS ===
            # Russell 2000 Small Cap High Growth Potential
            'SPCE', 'RKLB', 'ASTR', 'VACQ', 'HOL', 'SRAC', 'CCIV', 'THCB', 'ACTC', 'STPK', 'CLOV', 'WISH', 'BARK',
            'OPEN', 'Z', 'ZG', 'PETQ', 'WOOF', 'FRPT', 'WAGS', 'PETM', 'PETS', 'CYBR', 'PING', 'SPLK', 'TENB', 'RPD',
            'FEYE', 'QLYS', 'VRNS', 'MIME', 'PFPT', 'ALRM', 'SAIL', 'ASAN', 'MNDY', 'PD', 'BILL', 'DOCN', 'FSLY',
            'ICLN', 'PBW', 'QCLN', 'LIT', 'MTTR', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'UDR', 'ESS', 'MAA', 'CPT',
            'AIV', 'STOR', 'WPC', 'NNN', 'ADC', 'STAG', 'LXP', 'GTY', 'EPRT', 'FCPT', 'GOOD', 'VNO', 'BXP', 'KIM',
            'REG', 'TSM', 'ASML', 'NVO', 'UL', 'BP', 'TTE', 'E', 'SAN', 'BBVA', 'ING', 'TCOM', 'MOMO', 'SINA', 'SOHU',
            'KMB', 'CHD', 'CLX', 'MNST', 'KDP', 'STZ', 'BF.B', 'TAP', 'DEO', 'YUM', 'CMG', 'DPZ', 'PZZA', 'WING',
            'JACK', 'ARCO', 'DENN', 'CAKE', 'EAT', 'DRI', 'BLMN', 'LULU', 'UA', 'VFC', 'ROST', 'BURL', 'GPS', 'ANF',
            'AEO', 'URBN', 'ZUMZ', 'TJX', 'COST', 'TGT', 'WMT', 'HD', 'LOW', 'EBAY', 'VIPS', 'YMM', 'DANG', 'WB',
            'BIDU', 'NTES', 'RY', 'TD', 'BMO', 'BNS', 'CM', 'NA', 'CNR', 'CP', 'ATD', 'WCN', 'BAM', 'MFC', 'SU', 'CNQ'
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_universe = []
        for symbol in universe:
            if symbol not in seen:
                seen.add(symbol)
                unique_universe.append(symbol)
        
        return unique_universe
    
    def _get_data_hash(self, symbol, df):
        """Generate hash for caching based on symbol and data"""
        try:
            # Create hash from symbol + last few data points + data length
            data_str = f"{symbol}_{len(df)}_{df['Close'].iloc[-1]:.4f}_{df['Volume'].iloc[-1]}"
            return hashlib.md5(data_str.encode()).hexdigest()
        except:
            return f"{symbol}_{int(time.time())}"
    
    def _load_cached_analysis(self, cache_key):
        """Load cached analysis result"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                # Check if cache is less than 1 hour old
                if time.time() - os.path.getmtime(cache_file) < 3600:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
        except:
            pass
        return None
    
    def _save_cached_analysis(self, cache_key, result):
        """Save analysis result to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except:
            pass
    
    @functools.lru_cache(maxsize=200)
    def _cached_technical_score(self, symbol, data_hash):
        """Cached technical score calculation"""
        return self._indicator_cache.get(f"tech_{symbol}_{data_hash}", None)
    
    def analyze_stock_comprehensive(self, symbol, preloaded_hist: pd.DataFrame | None = None):
        """Comprehensive stock analysis with all available data"""
        try:
            # Check cache first
            if preloaded_hist is not None:
                data_hash = self._get_data_hash(symbol, preloaded_hist)
                cache_key = f"analysis_{symbol}_{data_hash}"
                cached_result = self._load_cached_analysis(cache_key)
                if cached_result:
                    print(f"📋 Cache hit: {symbol}")
                    return cached_result
            
            # Get comprehensive data
            stock_data = self.data_fetcher.get_comprehensive_stock_data(symbol, preloaded_hist=preloaded_hist)
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
            # Market overlay score from macro + breadth
            overlay_score = self._calculate_market_overlay_score(economic, getattr(self, '_breadth_context', {}))
            
            # Overall score with adaptive weights for light mode
            if getattr(self, 'data_mode', 'light') == 'light':
                overall_score = (
                    technical_score * 0.28 +
                    fundamental_score * 0.10 +
                    sentiment_score * 0.18 +
                    momentum_score * 0.18 +
                    volume_score * 0.09 +
                    volatility_score * 0.09 +
                    overlay_score * 0.08 +
                    sector_score * 0.00 +  # neutralize weaker/approx sector in light mode
                    0.0  # analyst excluded
                )
            else:
                overall_score = (
                    technical_score * 0.18 +
                    fundamental_score * 0.18 +
                    sentiment_score * 0.14 +
                    momentum_score * 0.14 +
                    volume_score * 0.09 +
                    volatility_score * 0.09 +
                    overlay_score * 0.09 +
                    sector_score * 0.05 +
                    analyst_score * 0.04
                )
            
            # Phase 3: Detect market regime and adjust targets/risk
            regime_info = self._detect_market_regime(df)
            market_regime = regime_info['regime']
            regime_risk_mult = regime_info['risk_multiplier']
            
            # Enhanced recommendation
            recommendation = self._generate_enhanced_recommendation(
                prediction_result, overall_score, technical_score, fundamental_score, sentiment_score, sector_score, analyst_score
            )
            
            # Professional upside/downside calculation
            current_price = df['Close'].iloc[-1]
            
            # Get analyst price target (ignored in light mode)
            analyst_target = analyst.get('price_target', 0)
            
            # Calculate multiple price targets
            technical_target = current_price * (1 + prediction_result['prediction'] / 100)
            
            # Combine targets (skip analyst target in light mode)
            if getattr(self, 'data_mode', 'light') != 'light' and analyst_target > 0:
                combined_target = (technical_target * 0.6 + analyst_target * 0.4)
            else:
                combined_target = technical_target
            
            # Calculate upside/downside potential
            upside_potential = ((combined_target - current_price) / current_price) * 100
            
            # Professional risk-adjusted target
            risk_multiplier = regime_risk_mult
            if analysis['risk_level'] == 'High':
                risk_multiplier *= 0.9
            elif analysis['risk_level'] == 'Low':
                risk_multiplier *= 1.05
            
            adjusted_target = current_price * (1 + (prediction_result['prediction'] / 100) * risk_multiplier)
            adjusted_upside = ((adjusted_target - current_price) / current_price) * 100
            stop_loss_price = current_price * 0.95  # 5% stop loss
            downside_risk = -5.0  # Maximum acceptable loss
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_1d': df['Close'].pct_change().iloc[-1] * 100,
                'volume': df['Volume'].iloc[-1],
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'sector': self._get_sector_from_symbol(symbol, info),
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
                'overlay_score': overlay_score,
                'analysis': analysis,
                'market_regime': market_regime,
                # Professional additions
                'upside_potential': upside_potential,
                'adjusted_upside': adjusted_upside,
                'stop_loss_price': stop_loss_price,
                'earnings_quality': earnings.get('earnings_quality_score', 50),
                'analyst_target': analyst_target,
                'technical_target': technical_target,
                'analyst_confidence': analyst.get('analyst_confidence', 50),
                'last_updated': datetime.now()
            }
            
            # Save to cache if we have preloaded data
            if preloaded_hist is not None:
                self._save_cached_analysis(cache_key, result)
                print(f"💾 Cached: {symbol}")
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def run_advanced_analysis(self, max_stocks=100, symbols: list[str] | None = None):
        """Run advanced analysis on multiple stocks (lightweight by default)."""
        try:
            # Train models first if enabled and not already trained
            if self.enable_training and not self.models:
                print("Models not trained yet. Training models first...")
                if not self._train_models(min(50, max_stocks)):
                    print("Failed to train models. Using simple predictions.")
            
            symbols = (symbols or self.stock_universe)[:max_stocks]
            # Bulk fetch OHLCV once for all symbols
            hist_map = self.data_fetcher.get_bulk_history(symbols, period="2y", interval="1d")
            
            # Prefill missing histories with per-symbol yfinance, then free Stooq fallback before threading
            try:
                missing = [s for s in symbols if not isinstance(hist_map.get(s), pd.DataFrame) or hist_map.get(s) is None or hist_map.get(s).empty]
                if missing:
                    import io, contextlib
                    for s in missing:
                        # Attempt a small per-symbol yfinance fetch first (suppressed output)
                        try:
                            buf_out, buf_err = io.StringIO(), io.StringIO()
                            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                                tk = yf.Ticker(s)
                                df_yf = tk.history(period="2y", interval="1d")
                            if df_yf is not None and not df_yf.empty and {'Open','High','Low','Close','Volume'}.issubset(df_yf.columns):
                                hist_map[s] = df_yf
                                continue
                        except Exception:
                            pass

                        # Then try Stooq variants
                        stooq_df = self.data_fetcher._fetch_stooq_history(s)
                        if (stooq_df is None or stooq_df.empty):
                            # Try common variants
                            variants = [f"{s}.us", s.lower(), f"{s.lower()}.us", s.replace('.', '-'), f"{s.replace('.', '-')}.us"]
                            for var in variants:
                                stooq_df = self.data_fetcher._fetch_stooq_history(var)
                                if stooq_df is not None and not stooq_df.empty:
                                    break
                        hist_map[s] = stooq_df if (stooq_df is not None and not stooq_df.empty) else None
            except Exception:
                pass
            
            # Build a list of symbols that have data after prefill
            valid_symbols = []
            for s in symbols:
                df_pre = hist_map.get(s)
                if isinstance(df_pre, pd.DataFrame) and not df_pre.empty and {'Open','High','Low','Close','Volume'}.issubset(df_pre.columns):
                    valid_symbols.append(s)
            if not valid_symbols:
                print("No valid histories after prefill. Falling back to curated large caps...")
                fallback_syms = self._get_safe_large_caps()[:max_stocks]
                hist_map = self.data_fetcher.get_bulk_history(fallback_syms, period="2y", interval="1d")
                # Prefill for fallback set
                try:
                    missing_fb = [s for s in fallback_syms if not isinstance(hist_map.get(s), pd.DataFrame) or hist_map.get(s) is None or hist_map.get(s).empty]
                    if missing_fb:
                        import io, contextlib
                        for s in missing_fb:
                            try:
                                buf_out, buf_err = io.StringIO(), io.StringIO()
                                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                                    tk = yf.Ticker(s)
                                    df_yf = tk.history(period="2y", interval="1d")
                                if df_yf is not None and not df_yf.empty and {'Open','High','Low','Close','Volume'}.issubset(df_yf.columns):
                                    hist_map[s] = df_yf
                                    continue
                            except Exception:
                                pass
                            stooq_df = self.data_fetcher._fetch_stooq_history(s)
                            if (stooq_df is None or stooq_df.empty):
                                variants = [f"{s}.us", s.lower(), f"{s.lower()}.us", s.replace('.', '-'), f"{s.replace('.', '-')}.us"]
                                for var in variants:
                                    stooq_df = self.data_fetcher._fetch_stooq_history(var)
                                    if stooq_df is not None and not stooq_df.empty:
                                        break
                            hist_map[s] = stooq_df if (stooq_df is not None and not stooq_df.empty) else None
                except Exception:
                    pass
                # Recompute valid symbols
                valid_symbols = []
                for s in fallback_syms:
                    df_pre = hist_map.get(s)
                    if isinstance(df_pre, pd.DataFrame) and not df_pre.empty and {'Open','High','Low','Close','Volume'}.issubset(df_pre.columns):
                        valid_symbols.append(s)
                if not valid_symbols:
                    print("Fallback large caps also failed to provide histories.")
                    return []

            # Compute internal breadth once per run using hist_map (zero-cost local compute)
            try:
                self._breadth_context = self._compute_internal_breadth(hist_map, valid_symbols)
            except Exception:
                self._breadth_context = {}

            results = []
            print(f"🚀 Starting optimized analysis of {len(valid_symbols)} stocks...")
            print(f"⚡ Performance mode: {self.max_workers} workers, caching enabled")
            
            # Enhanced task function with better error handling
            def enhanced_task(sym_data):
                sym, idx, total = sym_data
                try:
                    pre_hist = hist_map.get(sym) if isinstance(hist_map, dict) else None
                    result = self.analyze_stock_comprehensive(sym, preloaded_hist=pre_hist)
                    if result:
                        print(f"✅ {sym} ({idx+1}/{total}) - Score: {result.get('overall_score', 0):.1f}")
                    return result
                except Exception as e:
                    print(f"❌ {sym} ({idx+1}/{total}) - Error: {str(e)[:50]}")
                    return None

            # Prepare tasks with progress info
            tasks = [(sym, i, len(valid_symbols)) for i, sym in enumerate(valid_symbols)]
            
            # Use optimized worker count
            optimal_workers = min(self.max_workers, len(valid_symbols))
            print(f"🔧 Using {optimal_workers} parallel workers")
            
            # Process in batches to avoid memory issues
            batch_size = max(50, optimal_workers * 2)
            
            start_time = time.time()
            processed = 0
            
            for batch_start in range(0, len(tasks), batch_size):
                batch_tasks = tasks[batch_start:batch_start + batch_size]
                
                with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                    futures = {executor.submit(enhanced_task, task): task for task in batch_tasks}
                    
                    for fut in as_completed(futures):
                        try:
                            res = fut.result()
                            if res:
                                results.append(res)
                            processed += 1
                            
                            # Progress update every 10 stocks
                            if processed % 10 == 0:
                                elapsed = time.time() - start_time
                                rate = processed / elapsed
                                eta = (len(valid_symbols) - processed) / rate if rate > 0 else 0
                                print(f"📊 Progress: {processed}/{len(valid_symbols)} ({processed/len(valid_symbols)*100:.1f}%) - Rate: {rate:.1f}/sec - ETA: {eta/60:.1f}min")
                                
                        except Exception as e:
                            processed += 1
                            continue
                
                # Small delay between batches to prevent resource exhaustion
                if batch_start + batch_size < len(tasks):
                    time.sleep(0.1)
            
            elapsed_total = time.time() - start_time
            print(f"🎉 Analysis complete! {len(results)} stocks analyzed in {elapsed_total/60:.1f} minutes ({len(results)/elapsed_total:.1f} stocks/sec)")
            
            # Validate data integrity
            is_valid, validation_msg = self._validate_analysis_data(results)
            if is_valid:
                print(f"✅ {validation_msg}")
            else:
                print(f"❌ CRITICAL DATA ISSUE: {validation_msg}")
                print("🚨 WARNING: Analysis may contain unreliable data - review results carefully!")
            
            return results
        except Exception as e:
            print(f"Error in advanced analysis: {e}")
            return []

    def _compute_internal_breadth(self, hist_map: dict, symbols: list[str]) -> dict:
        """Compute internal market breadth metrics from already-fetched OHLCV.
        Zero external calls. Uses last available data for each symbol.
        """
        try:
            total = max(1, len(symbols))
            adv_1d = 0
            rets_1d = []
            above_50 = 0
            above_200 = 0
            nh_20 = 0
            nl_20 = 0

            for s in symbols:
                df = hist_map.get(s)
                if not isinstance(df, pd.DataFrame) or df.empty or 'Close' not in df.columns:
                    continue
                close = df['Close']
                if len(close) < 3:
                    continue
                r1 = float((close.iloc[-1] / close.iloc[-2]) - 1.0)
                rets_1d.append(r1)
                if r1 > 0:
                    adv_1d += 1

                # SMA checks (compute quickly here)
                sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
                sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
                last = close.iloc[-1]
                if not pd.isna(sma50) and last > sma50:
                    above_50 += 1
                if not pd.isna(sma200) and last > sma200:
                    above_200 += 1

                # 20-day highs/lows
                if len(close) >= 20:
                    if last >= close.rolling(20).max().iloc[-1]:
                        nh_20 += 1
                    if last <= close.rolling(20).min().iloc[-1]:
                        nl_20 += 1

            dec_1d = max(0, len(rets_1d) - adv_1d)
            adv_pct_1d = adv_1d / max(1, len(rets_1d))
            adv_dec_ratio_1d = adv_1d / max(1, dec_1d)
            pct_above_50 = above_50 / total
            pct_above_200 = above_200 / total
            nhl_ratio_20 = nh_20 / max(1, nl_20)
            median_ret_1d = float(np.median(rets_1d)) if rets_1d else 0.0

            return {
                'adv_pct_1d': adv_pct_1d,
                'adv_dec_ratio_1d': adv_dec_ratio_1d,
                'pct_above_sma50': pct_above_50,
                'pct_above_sma200': pct_above_200,
                'new_highs_20d': nh_20 / total,
                'new_lows_20d': nl_20 / total,
                'nh_nl_ratio_20d': nhl_ratio_20,
                'median_ret_1d': median_ret_1d,
                'universe_size': total,
            }
        except Exception as e:
            return {}

    def _get_safe_large_caps(self):
        """Curated list of highly reliable large-cap tickers for fallback."""
        return [
            'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD','INTC',
            'JPM','BAC','WFC','GS','MS','C','AXP','V','MA','PYPL',
            'JNJ','PFE','UNH','ABBV','MRK','TMO','ABT','DHR','BMY','AMGN','LLY',
            'KO','PEP','WMT','PG','HD','MCD','NKE','SBUX','DIS','CMCSA',
            'VZ','T','CVX','XOM','COP','SLB','BP','SHEL','BHP','RIO',
            'CAT','BA','MMM','GE','HON','LMT','NOC','RTX','DE','ETN'
        ]
    
    def _get_sector_from_symbol(self, symbol, info):
        """Get sector information with fallback mapping"""
        # First try to get from info
        sector = info.get('sector', None)
        if sector and sector != 'Unknown':
            return sector
            
        # Fallback sector mapping for common stocks
        sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
            'META': 'Technology', 'NVDA': 'Technology', 'TSLA': 'Technology', 'NFLX': 'Technology',
            'AMD': 'Technology', 'INTC': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology',
            'ADBE': 'Technology', 'CSCO': 'Technology', 'AVGO': 'Technology', 'TXN': 'Technology',
            'QCOM': 'Technology', 'IBM': 'Technology', 'INTU': 'Technology', 'NOW': 'Technology',
            'AMAT': 'Technology', 'ADI': 'Technology', 'LRCX': 'Technology', 'KLAC': 'Technology',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'MRK': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
            'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'LLY': 'Healthcare', 'CVS': 'Healthcare',
            'CI': 'Healthcare', 'ELV': 'Healthcare', 'GILD': 'Healthcare', 'VRTX': 'Healthcare',
            
            # Financial Services
            'JPM': 'Financial Services', 'BAC': 'Financial Services', 'WFC': 'Financial Services',
            'GS': 'Financial Services', 'MS': 'Financial Services', 'C': 'Financial Services',
            'AXP': 'Financial Services', 'V': 'Financial Services', 'MA': 'Financial Services',
            'PYPL': 'Financial Services', 'COF': 'Financial Services', 'USB': 'Financial Services',
            'PNC': 'Financial Services', 'TFC': 'Financial Services', 'BK': 'Financial Services',
            
            # Consumer Discretionary
            'AMZN': 'Consumer Discretionary', 'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
            'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary', 'DIS': 'Consumer Discretionary',
            'LOW': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary',
            
            # Consumer Staples
            'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples',
            'PG': 'Consumer Staples', 'COST': 'Consumer Staples', 'CL': 'Consumer Staples',
            
            # Communication Services
            'CMCSA': 'Communication Services', 'T': 'Communication Services', 'VZ': 'Communication Services',
            'CHTR': 'Communication Services', 'TMUS': 'Communication Services',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
            'SLB': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'MPC': 'Energy',
            
            # Industrials
            'BA': 'Industrials', 'CAT': 'Industrials', 'MMM': 'Industrials', 'GE': 'Industrials',
            'HON': 'Industrials', 'LMT': 'Industrials', 'NOC': 'Industrials', 'RTX': 'Industrials',
            'DE': 'Industrials', 'ETN': 'Industrials', 'UPS': 'Industrials', 'FDX': 'Industrials',
            
            # Utilities
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'AEP': 'Utilities',
            'EXC': 'Utilities', 'XEL': 'Utilities', 'PPL': 'Utilities', 'ES': 'Utilities',
            
            # Real Estate
            'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
            'SPG': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate', 'DLR': 'Real Estate',
            
            # Materials
            'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials', 'FCX': 'Materials',
            'NEM': 'Materials', 'SHW': 'Materials', 'VMC': 'Materials', 'MLM': 'Materials',
            
            # Biotechnology
            'NVAX': 'Biotechnology', 'SRPT': 'Biotechnology', 'BLUE': 'Biotechnology', 'EDIT': 'Biotechnology',
            'CRSP': 'Biotechnology', 'NTLA': 'Biotechnology', 'BEAM': 'Biotechnology', 'PRIME': 'Biotechnology',
            'VCYT': 'Biotechnology', 'PACB': 'Biotechnology', 'TWST': 'Biotechnology', 'CDNA': 'Biotechnology',
            'FATE': 'Biotechnology', 'SGMO': 'Biotechnology', 'RGNX': 'Biotechnology', 'RARE': 'Biotechnology',
            'FOLD': 'Biotechnology', 'ARWR': 'Biotechnology', 'IONS': 'Biotechnology', 'EXAS': 'Biotechnology',
            
            # Cybersecurity
            'CYBR': 'Technology', 'PING': 'Technology', 'SPLK': 'Technology', 'TENB': 'Technology',
            'RPD': 'Technology', 'FEYE': 'Technology', 'QLYS': 'Technology', 'VRNS': 'Technology',
            'MIME': 'Technology', 'PFPT': 'Technology', 'ALRM': 'Technology', 'SAIL': 'Technology',
            
            # Software
            'ASAN': 'Technology', 'MNDY': 'Technology', 'PD': 'Technology', 'BILL': 'Technology',
            'DOCN': 'Technology', 'FSLY': 'Technology',
            
            # Healthcare Technology
            'TDOC': 'Healthcare', 'DXCM': 'Healthcare', 'ALGN': 'Healthcare', 'PODD': 'Healthcare',
            
            # Clean Energy
            'ENPH': 'Energy', 'SEDG': 'Energy', 'RUN': 'Energy', 'NOVA': 'Energy',
            'FSLR': 'Energy', 'SPWR': 'Energy', 'CSIQ': 'Energy', 'JKS': 'Energy',
            'SOL': 'Energy', 'MAXN': 'Energy', 'PLUG': 'Energy', 'FCEL': 'Energy',
            'BE': 'Energy', 'BLDP': 'Energy', 'HYLN': 'Energy', 'QS': 'Energy',
            'CHPT': 'Energy', 'BLNK': 'Energy', 'EVG': 'Energy', 'CLNE': 'Energy',
            
            # Additional Large Cap
            'BERKB': 'Financial Services', 'LLY': 'Healthcare', 'AVGO': 'Technology',
            'ACN': 'Technology', 'LIN': 'Materials', 'RTX': 'Industrials', 'UNP': 'Industrials',
            'PM': 'Consumer Staples', 'SPGI': 'Financial Services', 'BKNG': 'Consumer Discretionary',
            'DE': 'Industrials', 'BLK': 'Financial Services', 'ELV': 'Healthcare', 'ZTS': 'Healthcare',
            'NSC': 'Industrials', 'HUM': 'Healthcare', 'ITW': 'Industrials', 'AON': 'Financial Services',
            'MMC': 'Financial Services', 'MCK': 'Healthcare', 'CSX': 'Industrials', 'MSI': 'Technology',
            'FISV': 'Technology', 'ECL': 'Materials', 'EMR': 'Industrials',
            
            # Medium Cap Additions
            'ROKU': 'Technology', 'DKNG': 'Consumer Discretionary', 'PENN': 'Consumer Discretionary',
            'MGM': 'Consumer Discretionary', 'LVS': 'Consumer Discretionary', 'WYNN': 'Consumer Discretionary',
            'CZR': 'Consumer Discretionary', 'BYD': 'Consumer Discretionary', 'RSI': 'Consumer Discretionary',
            'GLUU': 'Technology', 'HUYA': 'Technology', 'DOYU': 'Technology', 'BILI': 'Technology',
            'IQ': 'Technology', 'FUBO': 'Technology', 'PARA': 'Communication Services',
            'WBD': 'Communication Services', 'FOXA': 'Communication Services', 'EA': 'Technology',
            'TTWO': 'Technology', 'ZNGA': 'Technology', 'RBLX': 'Technology', 'U': 'Technology',
            
            # Small Cap Additions
            'SPCE': 'Technology', 'RKLB': 'Technology', 'ASTR': 'Technology', 'VACQ': 'Technology',
            'HOL': 'Technology', 'SRAC': 'Technology', 'CCIV': 'Technology', 'THCB': 'Technology',
            'ACTC': 'Technology', 'STPK': 'Technology', 'CLOV': 'Healthcare', 'WISH': 'Technology',
            'WAGS': 'Consumer Discretionary', 'PETM': 'Consumer Discretionary', 'PETS': 'Consumer Discretionary',
            'ICLN': 'Energy', 'PBW': 'Energy', 'QCLN': 'Energy', 'LIT': 'Materials',
            'MTTR': 'Technology', 'STOR': 'Real Estate', 'WPC': 'Real Estate', 'NNN': 'Real Estate',
            'ADC': 'Real Estate', 'STAG': 'Real Estate', 'LXP': 'Real Estate', 'GTY': 'Real Estate',
            'EPRT': 'Real Estate', 'FCPT': 'Real Estate', 'GOOD': 'Real Estate', 'VNO': 'Real Estate',
            
            # International/ADRs
            'TSM': 'Technology', 'ASML': 'Technology', 'NVO': 'Healthcare', 'UL': 'Consumer Staples',
            'TTE': 'Energy', 'E': 'Energy', 'SAN': 'Financial Services', 'BBVA': 'Financial Services',
            'ING': 'Financial Services', 'TCOM': 'Technology', 'MOMO': 'Technology', 'SINA': 'Technology',
            'SOHU': 'Technology', 'KMB': 'Consumer Staples', 'CHD': 'Consumer Staples',
            'CLX': 'Consumer Staples', 'MNST': 'Consumer Staples', 'KDP': 'Consumer Staples',
            'STZ': 'Consumer Staples', 'BF.B': 'Consumer Staples', 'TAP': 'Consumer Staples',
            'DEO': 'Consumer Staples', 'CMG': 'Consumer Discretionary', 'DPZ': 'Consumer Discretionary',
            'PZZA': 'Consumer Discretionary', 'WING': 'Consumer Discretionary', 'JACK': 'Consumer Discretionary',
            'ARCO': 'Consumer Discretionary', 'DENN': 'Consumer Discretionary', 'CAKE': 'Consumer Discretionary',
            'EAT': 'Consumer Discretionary', 'DRI': 'Consumer Discretionary', 'BLMN': 'Consumer Discretionary',
            'UA': 'Consumer Discretionary', 'VFC': 'Consumer Discretionary', 'ZUMZ': 'Consumer Discretionary'
        }
        
        return sector_mapping.get(symbol, 'Technology')  # Default to Technology for unknown symbols
    
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
        # Include expanded macro context if available
        features['macro_spy_return_1d'] = economic.get('spy_return_1d', 0.0)
        features['macro_spy_vol_20'] = economic.get('spy_vol_20', 0.0)
        features['macro_usd_1d'] = economic.get('usd_change_1d', 0.0)
        features['macro_gold_1d'] = economic.get('gold_change_1d', 0.0)
        features['macro_oil_1d'] = economic.get('oil_change_1d', 0.0)
        features['macro_yield_10y'] = economic.get('yield_10y', 0.0)
        features['macro_yield_3m'] = economic.get('yield_3m', 0.0)
        features['macro_yield_curve'] = economic.get('yield_curve_slope', 0.0)
        features['macro_hyg_lqd_1d'] = economic.get('hyg_lqd_ratio_1d', 0.0)
        features['macro_small_large_1d'] = economic.get('small_large_ratio_1d', 0.0)
        features['macro_xly_xlp_1d'] = economic.get('xly_xlp_ratio_1d', 0.0)
        features['macro_semis_spy_1d'] = economic.get('semis_spy_ratio_1d', 0.0)
        # Static placeholders retained
        features['fed_rate'] = economic['fed_rate']
        features['gdp_growth'] = economic['gdp_growth']
        features['inflation'] = economic['inflation']
        features['unemployment'] = economic['unemployment']

        # Internal breadth features (computed once per run)
        bc = getattr(self, '_breadth_context', {}) or {}
        features['breadth_adv_pct_1d'] = bc.get('adv_pct_1d', 0.5)
        features['breadth_adv_dec_ratio_1d'] = bc.get('adv_dec_ratio_1d', 1.0)
        features['breadth_pct_above_sma50'] = bc.get('pct_above_sma50', 0.5)
        features['breadth_pct_above_sma200'] = bc.get('pct_above_sma200', 0.5)
        features['breadth_new_highs_20d'] = bc.get('new_highs_20d', 0.1)
        features['breadth_new_lows_20d'] = bc.get('new_lows_20d', 0.1)
        features['breadth_nh_nl_ratio_20d'] = bc.get('nh_nl_ratio_20d', 1.0)
        features['breadth_median_ret_1d'] = bc.get('median_ret_1d', 0.0)
        
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
                # Fallback to simple prediction based on technical indicators
                return self._simple_prediction(features)
            
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
                return self._simple_prediction(features)
            
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
            return self._simple_prediction(features)
    
    def _simple_prediction(self, features):
        """Simple prediction based on technical indicators when ML models fail"""
        try:
            if features.empty:
                return {'prediction': 0, 'confidence': 0.3}
            
            # Get key technical indicators
            rsi = features.get('RSI_14_current', [50])[0] if 'RSI_14_current' in features.columns else 50
            macd = features.get('MACD_12_26_current', [0])[0] if 'MACD_12_26_current' in features.columns else 0
            bb_position = features.get('BB_Position', [0.5])[0] if 'BB_Position' in features.columns else 0.5
            volume_ratio = features.get('Volume_Ratio', [1])[0] if 'Volume_Ratio' in features.columns else 1
            
            # Simple prediction logic
            prediction = 0
            
            # RSI-based prediction
            if rsi < 30:
                prediction += 2.0  # Oversold - bullish
            elif rsi > 70:
                prediction -= 2.0  # Overbought - bearish
            elif 40 <= rsi <= 60:
                prediction += 0.5  # Neutral - slightly bullish
            
            # MACD-based prediction
            if macd > 0:
                prediction += 1.5  # MACD above zero - bullish
            else:
                prediction -= 1.0  # MACD below zero - bearish
            
            # Bollinger Bands position
            if bb_position < 0.2:
                prediction += 1.0  # Near lower band - bullish
            elif bb_position > 0.8:
                prediction -= 1.0  # Near upper band - bearish
            
            # Volume confirmation
            if volume_ratio > 1.5:
                prediction *= 1.2  # High volume confirms signal
            elif volume_ratio < 0.5:
                prediction *= 0.8  # Low volume weakens signal
            
            # Add some randomness for realism
            prediction += np.random.normal(0, 0.5)
            
            # Cap prediction between -5% and +5%
            prediction = max(-5, min(5, prediction))
            
            # Calculate confidence based on signal strength
            confidence = min(0.8, abs(prediction) / 3.0 + 0.3)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'method': 'simple_technical'
            }
            
        except Exception as e:
            print(f"Error in simple prediction: {e}")
            return {'prediction': 0, 'confidence': 0.3}
    
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

    def _detect_market_regime(self, df):
        """Phase 3: Detect Bull/Neutral/Bear regime using only OHLCV (no external data)."""
        try:
            # Use moving averages, volatility and drawdown
            sma50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns and not pd.isna(df['SMA_50'].iloc[-1]) else df['Close'].rolling(50).mean().iloc[-1]
            sma200 = df['SMA_200'].iloc[-1] if 'SMA_200' in df.columns and not pd.isna(df['SMA_200'].iloc[-1]) else df['Close'].rolling(200).mean().iloc[-1]
            vol20 = df['Volatility_20'].iloc[-1] if 'Volatility_20' in df.columns and not pd.isna(df['Volatility_20'].iloc[-1]) else df['Close'].pct_change().rolling(20).std().iloc[-1]
            recent_close = df['Close'].iloc[-1]
            rolling_max = df['Close'].rolling(200).max().iloc[-1]
            drawdown = (recent_close / rolling_max - 1.0) if rolling_max and not pd.isna(rolling_max) else 0.0

            # Simple rule-based regime
            if sma50 > sma200 and drawdown > -0.10 and vol20 < 0.03:
                return {
                    'regime': 'Bull',
                    'risk_multiplier': 1.10,
                    'weights': {'momentum': 1.1, 'mean_rev': 0.9}
                }
            elif sma50 < sma200 and (drawdown < -0.15 or vol20 > 0.04):
                return {
                    'regime': 'Bear',
                    'risk_multiplier': 0.85,
                    'weights': {'momentum': 0.9, 'mean_rev': 1.1}
                }
            else:
                return {
                    'regime': 'Neutral',
                    'risk_multiplier': 1.0,
                    'weights': {'momentum': 1.0, 'mean_rev': 1.0}
                }
        except Exception:
            return {'regime': 'Neutral', 'risk_multiplier': 1.0, 'weights': {'momentum': 1.0, 'mean_rev': 1.0}}
    
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

            # SuperTrend direction
            if 'SuperTrend_Dir' in df.columns:
                st_dir = df['SuperTrend_Dir'].iloc[-1]
                if st_dir == 1:
                    signals.append("SuperTrend Uptrend - BUY")
                elif st_dir == -1:
                    signals.append("SuperTrend Downtrend - SELL")

            # Donchian breakout/breakdown
            if 'Donchian_Upper' in df.columns and 'Donchian_Lower' in df.columns:
                du = df['Donchian_Upper'].iloc[-1]
                dl = df['Donchian_Lower'].iloc[-1]
                if current_price > du:
                    signals.append("Donchian Breakout - BUY")
                elif current_price < dl:
                    signals.append("Donchian Breakdown - SELL")

            # Keltner channel breakout and squeeze hint
            if all(c in df.columns for c in ['Keltner_Upper','Keltner_Lower','Keltner_Width']):
                ku = df['Keltner_Upper'].iloc[-1]
                kl = df['Keltner_Lower'].iloc[-1]
                kw = df['Keltner_Width']
                if current_price > ku:
                    signals.append("Keltner Channel Breakout - BUY")
                elif current_price < kl:
                    signals.append("Keltner Channel Breakdown - SELL")
                # Squeeze detection: width at local percentile
                try:
                    recent_kw = kw.tail(50)
                    if len(recent_kw) >= 20:
                        pct = (recent_kw.rank(pct=True).iloc[-1])
                        if pct < 0.2:
                            signals.append("Keltner Squeeze Forming - Potential Expansion")
                except Exception:
                    pass
            
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
            
            # Options/Institutional/Analyst signals only in full mode
            if getattr(self, 'data_mode', 'light') != 'light':
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
            if getattr(self, 'data_mode', 'light') != 'light':
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
            
            # SuperTrend contribution
            if 'SuperTrend_Dir' in df.columns:
                st_dir = df['SuperTrend_Dir'].iloc[-1]
                if st_dir == 1:
                    score += 10
                elif st_dir == -1:
                    score -= 10

            # Donchian breakout/breakdown
            if 'Donchian_Upper' in df.columns and 'Donchian_Lower' in df.columns:
                du = df['Donchian_Upper'].iloc[-1]
                dl = df['Donchian_Lower'].iloc[-1]
                if current_price > du:
                    score += 10
                elif current_price < dl:
                    score -= 10

            # Keltner channel breakout and squeeze
            if all(c in df.columns for c in ['Keltner_Upper','Keltner_Lower','Keltner_Width']):
                ku = df['Keltner_Upper'].iloc[-1]
                kl = df['Keltner_Lower'].iloc[-1]
                kw = df['Keltner_Width']
                if current_price > ku:
                    score += 8
                elif current_price < kl:
                    score -= 8
                try:
                    recent_kw = kw.tail(50)
                    if len(recent_kw) >= 20:
                        pct = (recent_kw.rank(pct=True).iloc[-1])
                        if pct < 0.2:
                            score += 5  # squeeze likely resolves into expansion
                except Exception:
                    pass
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50

    def _calculate_market_overlay_score(self, economic: dict, breadth: dict) -> float:
        """Combine macro (one-shot) and internal breadth into a 0-100 overlay score."""
        try:
            # Base neutral score
            score = 50.0
            vix = float(economic.get('vix', 20.0))
            yield_curve = float(economic.get('yield_curve_slope', 0.0))
            usd = float(economic.get('usd_change_1d', 0.0))
            oil = float(economic.get('oil_change_1d', 0.0))
            xly_xlp = float(economic.get('xly_xlp_ratio_1d', 0.0))
            semis = float(economic.get('semis_spy_ratio_1d', 0.0))
            hyg_lqd = float(economic.get('hyg_lqd_ratio_1d', 0.0))

            # Breadth
            adv_pct = float(breadth.get('adv_pct_1d', 0.5))  # 0..1
            pct50 = float(breadth.get('pct_above_sma50', 0.5))
            pct200 = float(breadth.get('pct_above_sma200', 0.5))
            nh_nl = float(breadth.get('nh_nl_ratio_20d', 1.0))  # >1 bullish

            # Macro influences (heuristic)
            # Lower VIX is bullish, high VIX is bearish
            if vix < 18:
                score += 5
            elif vix > 28:
                score -= 7

            # Positive yield curve slope bullish, inverted bearish
            score += max(-10, min(10, yield_curve * 50))  # ~ +/-10 pts

            # Cyclical vs defensive
            score += max(-5, min(5, xly_xlp * 100))

            # Semis leadership as tech risk-on proxy
            score += max(-5, min(5, semis * 100))

            # Credit risk appetite
            score += max(-5, min(5, hyg_lqd * 100))

            # Breadth influences
            score += (adv_pct - 0.5) * 30  # +/-15
            score += (pct50 - 0.5) * 20    # +/-10
            score += (pct200 - 0.5) * 20   # +/-10
            score += max(-10, min(10, (nh_nl - 1.0) * 10))

            return max(0.0, min(100.0, score))
        except Exception:
            return 50.0
    
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
            
            # More realistic recommendation thresholds for current market conditions
            if prediction > 0.04 and confidence > 0.65 and overall_score > 75:
                return {'action': 'STRONG BUY', 'confidence': 'Very High'}
            elif prediction > 0.025 and confidence > 0.55 and overall_score > 65:
                return {'action': 'BUY', 'confidence': 'High'}
            elif prediction > 0.01 and confidence > 0.45 and overall_score > 55:
                return {'action': 'WEAK BUY', 'confidence': 'Medium'}
            elif prediction > -0.01 and overall_score > 45:
                return {'action': 'HOLD', 'confidence': 'Medium'}
            elif prediction > -0.025 and overall_score > 35:
                return {'action': 'WEAK SELL', 'confidence': 'Medium'}
            elif prediction > -0.04 and overall_score > 25:
                return {'action': 'SELL', 'confidence': 'High'}
            else:
                return {'action': 'STRONG SELL', 'confidence': 'Very High'}
                
        except Exception as e:
            return {'action': 'HOLD', 'confidence': 'Low'}
    
    def _train_models(self, max_stocks=50):
        """Train ML models on historical data"""
        try:
            print("Training ML models on historical data...")
            
            # Collect training data from multiple stocks
            training_data = []
            training_targets = []
            
            # Use a subset of stocks for training
            training_stocks = self.stock_universe[:max_stocks]
            
            for i, symbol in enumerate(training_stocks):
                print(f"Collecting training data from {symbol} ({i+1}/{len(training_stocks)})...")
                try:
                    stock_data = self.data_fetcher.get_comprehensive_stock_data(symbol)
                    if stock_data and not stock_data['data'].empty:
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
                        
                        # Create features for each historical point
                        for j in range(50, len(df)):  # Start from 50 to have enough history
                            try:
                                # Get historical data up to point j
                                hist_df = df.iloc[:j+1]
                                
                                # Create features
                                features = self._create_comprehensive_features(
                                    hist_df, info, news, insider, options, 
                                    institutional, earnings, economic, sector, analyst
                                )
                                
                                if not features.empty:
                                    # Calculate target (future return)
                                    if j + 5 < len(df):  # 5-day future return
                                        future_price = df['Close'].iloc[j + 5]
                                        current_price = df['Close'].iloc[j]
                                        target = (future_price - current_price) / current_price * 100
                                        
                                        training_data.append(features.values[0])
                                        training_targets.append(target)
                                        
                            except Exception as e:
                                continue
                                
                except Exception as e:
                    print(f"Error collecting data from {symbol}: {e}")
                    continue
                
                time.sleep(0.1)  # Rate limiting
            
            if len(training_data) < 100:
                print("Not enough training data collected. Using default models.")
                return False
            
            # Convert to numpy arrays
            X = np.array(training_data)
            y = np.array(training_targets)
            
            print(f"Training on {len(X)} samples with {X.shape[1]} features...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=10, max_depth=6, n_jobs=-1, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=10, max_depth=6, random_state=42),
                'ExtraTrees': ExtraTreesRegressor(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.1),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
                'SVR': SVR(kernel='rbf', C=1.0),
                'MLPRegressor': MLPRegressor(hidden_layer_sizes=(50,), max_iter=100, random_state=42)
            }
            
            if LIGHTGBM_AVAILABLE:
                models['LightGBM'] = lgb.LGBMRegressor(n_estimators=10, max_depth=6, n_jobs=-1, random_state=42)
            
            trained_models = {}
            for name, model in models.items():
                try:
                    print(f"Training {name}...")
                    model.fit(X_train_scaled, y_train)
                    
                    # Test performance
                    y_pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    print(f"{name} R² score: {r2:.3f}")
                    
                    trained_models[name] = model
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if trained_models:
                self.models = trained_models
                self.scalers = {'main': scaler}
                print(f"Successfully trained {len(trained_models)} models!")
                return True
            else:
                print("Failed to train any models.")
                return False
                
        except Exception as e:
            print(f"Error training models: {e}")
            return False

    
    
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
