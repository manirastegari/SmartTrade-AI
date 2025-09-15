"""
Simple Automated Trading Analysis System
Works with minimal dependencies - no pandas-ta or lightgbm
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb

class SimpleTradingAnalyzer:
    """Simple automated trading analysis system"""
    
    def __init__(self):
        self.stock_universe = [
            # Major US Stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
            'KO', 'PEP', 'WMT', 'PG', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'KMI',
            # Canadian Stocks
            'SHOP', 'RY', 'TD', 'CNR', 'CP', 'ATD', 'WCN', 'BAM', 'MFC', 'SU'
        ]
        
        self.models = {}
        self.scalers = {}
        self.last_analysis = {}
        
    def get_stock_data_auto(self, symbol):
        """Automatically fetch comprehensive stock data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period="2y")
            info = ticker.info
            
            if hist.empty:
                return None
            
            # Add technical indicators manually (without pandas-ta)
            hist = self._add_technical_indicators(hist)
            
            return {
                'symbol': symbol,
                'data': hist,
                'info': info,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df):
        """Add technical indicators manually"""
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD calculation
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Price change indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5d'] = df['Close'].pct_change(5)
            df['Price_Change_20d'] = df['Close'].pct_change(20)
            
            # Volatility
            df['Volatility'] = df['Price_Change'].rolling(window=20).std()
            
            # Support and Resistance
            df['Support'] = df['Low'].rolling(window=20).min()
            df['Resistance'] = df['High'].rolling(window=20).max()
            
            return df
            
        except Exception as e:
            print(f"Error adding indicators: {e}")
            return df
    
    def create_ml_features(self, df, info):
        """Create ML features"""
        try:
            features = {}
            
            # Technical features
            tech_cols = ['RSI', 'MACD', 'MACD_signal', 'Volatility']
            for col in tech_cols:
                if col in df.columns:
                    features[f'{col}_current'] = df[col].iloc[-1] if not pd.isna(df[col].iloc[-1]) else 0
                    features[f'{col}_avg_5'] = df[col].rolling(5).mean().iloc[-1] if not pd.isna(df[col].rolling(5).mean().iloc[-1]) else 0
                    features[f'{col}_avg_20'] = df[col].rolling(20).mean().iloc[-1] if not pd.isna(df[col].rolling(20).mean().iloc[-1]) else 0
            
            # Price features
            features['Price_Change_1d'] = df['Price_Change'].iloc[-1] if not pd.isna(df['Price_Change'].iloc[-1]) else 0
            features['Price_Change_5d'] = df['Price_Change_5d'].iloc[-1] if not pd.isna(df['Price_Change_5d'].iloc[-1]) else 0
            features['Price_Change_20d'] = df['Price_Change_20d'].iloc[-1] if not pd.isna(df['Price_Change_20d'].iloc[-1]) else 0
            
            # Volume features
            features['Volume_Ratio'] = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
            features['Volume_Change_1d'] = df['Volume'].pct_change().iloc[-1] if not pd.isna(df['Volume'].pct_change().iloc[-1]) else 0
            
            # Moving average features
            features['Price_vs_SMA20'] = df['Close'].iloc[-1] / df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) and df['SMA_20'].iloc[-1] != 0 else 1
            features['Price_vs_SMA50'] = df['Close'].iloc[-1] / df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) and df['SMA_50'].iloc[-1] != 0 else 1
            features['Price_vs_SMA200'] = df['Close'].iloc[-1] / df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) and df['SMA_200'].iloc[-1] != 0 else 1
            
            # Bollinger Bands features
            bb_upper = df['BB_upper'].iloc[-1] if not pd.isna(df['BB_upper'].iloc[-1]) else df['Close'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1] if not pd.isna(df['BB_lower'].iloc[-1]) else df['Close'].iloc[-1]
            
            if bb_upper != bb_lower:
                features['BB_Position'] = (df['Close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower)
            else:
                features['BB_Position'] = 0.5
            
            # Fundamental features
            features['pe_ratio'] = info.get('trailingPE', 0) if info.get('trailingPE') else 0
            features['pb_ratio'] = info.get('priceToBook', 0) if info.get('priceToBook') else 0
            features['ps_ratio'] = info.get('priceToSalesTrailing12Months', 0) if info.get('priceToSalesTrailing12Months') else 0
            features['dividend_yield'] = info.get('dividendYield', 0) if info.get('dividendYield') else 0
            features['beta'] = info.get('beta', 1.0) if info.get('beta') else 1.0
            features['market_cap'] = np.log10(info.get('marketCap', 1)) if info.get('marketCap') else 0
            
            return pd.DataFrame([features])
            
        except Exception as e:
            print(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def train_models_auto(self, training_data):
        """Train ML models"""
        try:
            if training_data.empty:
                return {}
            
            # Prepare data
            X = training_data.drop(['target', 'symbol'], axis=1, errors='ignore')
            y = training_data['target']
            
            # Remove any columns with all NaN values
            X = X.dropna(axis=1, how='all')
            
            if X.empty or len(X) < 10:
                return {}
            
            # Fill NaN values
            X = X.fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
            }
            
            trained_models = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    trained_models[name] = model
                except Exception as e:
                    print(f"Error training {name}: {e}")
            
            self.models = trained_models
            self.scalers['main'] = scaler
            
            return trained_models
            
        except Exception as e:
            print(f"Error training models: {e}")
            return {}
    
    def predict_auto(self, features):
        """Make predictions"""
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
            
            # Ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()))
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
    
    def analyze_stock_auto(self, symbol):
        """Fully automated stock analysis"""
        try:
            # Get data
            stock_data = self.get_stock_data_auto(symbol)
            if not stock_data:
                return None
            
            df = stock_data['data']
            info = stock_data['info']
            
            # Create features
            features = self.create_ml_features(df, info)
            if features.empty:
                return None
            
            # Make prediction
            prediction_result = self.predict_auto(features)
            
            # Technical analysis
            current_price = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            macd = df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else 0
            macd_signal = df['MACD_signal'].iloc[-1] if not pd.isna(df['MACD_signal'].iloc[-1]) else 0
            
            # Generate signals automatically
            signals = []
            if rsi < 30:
                signals.append("RSI Oversold - BUY Signal")
            elif rsi > 70:
                signals.append("RSI Overbought - SELL Signal")
            
            if macd > macd_signal:
                signals.append("MACD Bullish - BUY Signal")
            else:
                signals.append("MACD Bearish - SELL Signal")
            
            # Moving average signals
            sma20 = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else current_price
            sma50 = df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else current_price
            sma200 = df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else current_price
            
            if current_price > sma20 > sma50 > sma200:
                signals.append("Golden Cross - STRONG BUY")
            elif current_price < sma20 < sma50 < sma200:
                signals.append("Death Cross - STRONG SELL")
            
            # Volume signals
            volume_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
            if volume_ratio > 2:
                signals.append("High Volume - Momentum Building")
            elif volume_ratio < 0.5:
                signals.append("Low Volume - Weak Interest")
            
            # Risk assessment
            volatility = df['Volatility'].iloc[-1] if not pd.isna(df['Volatility'].iloc[-1]) else 0.02
            beta = info.get('beta', 1.0) if info.get('beta') else 1.0
            
            if volatility > 0.03 or beta > 1.5:
                risk_level = 'High'
            elif volatility > 0.02 or beta > 1.2:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Automatic recommendation
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            
            if prediction > 0.05 and confidence > 0.7:
                recommendation = 'STRONG BUY'
                action = 'BUY NOW'
            elif prediction > 0.02 and confidence > 0.6:
                recommendation = 'BUY'
                action = 'BUY'
            elif prediction > -0.02:
                recommendation = 'HOLD'
                action = 'HOLD'
            elif prediction > -0.05:
                recommendation = 'WEAK SELL'
                action = 'CONSIDER SELLING'
            else:
                recommendation = 'SELL'
                action = 'SELL'
            
            # Calculate scores
            technical_score = self._calculate_technical_score_auto(df)
            fundamental_score = self._calculate_fundamental_score_auto(info)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_1d': df['Close'].pct_change().iloc[-1] * 100 if not pd.isna(df['Close'].pct_change().iloc[-1]) else 0,
                'volume': df['Volume'].iloc[-1],
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'sector': info.get('sector', 'Unknown'),
                'prediction': prediction,
                'confidence': confidence,
                'recommendation': recommendation,
                'action': action,
                'risk_level': risk_level,
                'signals': signals,
                'technical_score': technical_score,
                'fundamental_score': fundamental_score,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_technical_score_auto(self, df):
        """Calculate technical score"""
        try:
            score = 50
            
            # RSI score
            rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            if 30 <= rsi <= 70:
                score += 10
            elif rsi < 30 or rsi > 70:
                score -= 5
            
            # MACD score
            macd = df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else 0
            macd_signal = df['MACD_signal'].iloc[-1] if not pd.isna(df['MACD_signal'].iloc[-1]) else 0
            if macd > macd_signal:
                score += 10
            else:
                score -= 5
            
            # Moving average score
            sma20 = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else df['Close'].iloc[-1]
            sma50 = df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else df['Close'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else df['Close'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            if current_price > sma20 > sma50 > sma200:
                score += 15
            elif current_price < sma20 < sma50 < sma200:
                score -= 15
            
            # Volume score
            volume_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
            if volume_ratio > 1.5:
                score += 10
            elif volume_ratio < 0.5:
                score -= 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def _calculate_fundamental_score_auto(self, info):
        """Calculate fundamental score"""
        try:
            score = 50
            
            # P/E ratio score
            pe_ratio = info.get('trailingPE', 0) if info.get('trailingPE') else 0
            if 10 <= pe_ratio <= 25:
                score += 15
            elif pe_ratio > 50:
                score -= 20
            elif pe_ratio < 5:
                score -= 10
            
            # Growth score
            revenue_growth = info.get('revenueGrowth', 0) if info.get('revenueGrowth') else 0
            if revenue_growth and revenue_growth > 0.1:
                score += 10
            
            # Profitability score
            profit_margins = info.get('profitMargins', 0) if info.get('profitMargins') else 0
            if profit_margins and profit_margins > 0.1:
                score += 10
            elif profit_margins and profit_margins < 0:
                score -= 15
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50
    
    def run_automated_analysis(self, max_stocks=20):
        """Run fully automated analysis on multiple stocks"""
        try:
            results = []
            print(f"Starting automated analysis of {max_stocks} stocks...")
            
            for i, symbol in enumerate(self.stock_universe[:max_stocks]):
                print(f"Analyzing {symbol} ({i+1}/{max_stocks})...")
                result = self.analyze_stock_auto(symbol)
                if result:
                    results.append(result)
                time.sleep(0.5)  # Rate limiting
            
            self.last_analysis = {
                'results': results,
                'timestamp': datetime.now(),
                'count': len(results)
            }
            
            return results
            
        except Exception as e:
            print(f"Error in automated analysis: {e}")
            return []
    
    def get_top_picks(self, results, top_n=10):
        """Get top stock picks"""
        if not results:
            return []
        
        df = pd.DataFrame(results)
        # Sort by prediction score and confidence
        df['combined_score'] = df['prediction'] * df['confidence']
        top_picks = df.nlargest(top_n, 'combined_score')
        
        return top_picks.to_dict('records')

def main():
    st.set_page_config(
        page_title="Simple Trading Analyzer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Simple Automated Trading Analyzer")
    st.markdown("**100% Automatic** - Fetches data, analyzes, and gives recommendations")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SimpleTradingAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar controls
    with st.sidebar:
        st.header("🤖 Automation Controls")
        
        # Analysis settings
        max_stocks = st.slider("Number of Stocks to Analyze", 5, 30, 15)
        
        if st.button("🚀 Start Automated Analysis", type="primary"):
            with st.spinner("Running automated analysis..."):
                results = analyzer.run_automated_analysis(max_stocks)
                st.session_state.results = results
        
        if st.button("🔄 Refresh Analysis"):
            with st.spinner("Refreshing analysis..."):
                results = analyzer.run_automated_analysis(max_stocks)
                st.session_state.results = results
    
    # Main dashboard
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stocks Analyzed", len(results))
            strong_buys = len([r for r in results if r['recommendation'] == 'STRONG BUY'])
            st.metric("Strong Buys", strong_buys)
        
        with col2:
            avg_prediction = np.mean([r['prediction'] for r in results])
            st.metric("Avg Prediction", f"{avg_prediction:.2%}")
            avg_confidence = np.mean([r['confidence'] for r in results])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            high_risk = len([r for r in results if r['risk_level'] == 'High'])
            st.metric("High Risk", high_risk)
            low_risk = len([r for r in results if r['risk_level'] == 'Low'])
            st.metric("Low Risk", low_risk)
        
        with col4:
            avg_tech = np.mean([r['technical_score'] for r in results])
            st.metric("Avg Tech Score", f"{avg_tech:.1f}")
            avg_fund = np.mean([r['fundamental_score'] for r in results])
            st.metric("Avg Fund Score", f"{avg_fund:.1f}")
        
        # Top picks
        st.subheader("🎯 Top Stock Picks (Automatically Selected)")
        top_picks = analyzer.get_top_picks(results, 10)
        
        if top_picks:
            for i, pick in enumerate(top_picks, 1):
                with st.expander(f"#{i} {pick['symbol']} - {pick['recommendation']} - {pick['action']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Price:** ${pick['current_price']:.2f}")
                        st.write(f"**Change:** {pick['price_change_1d']:.2f}%")
                        st.write(f"**Volume:** {pick['volume']:,.0f}")
                    
                    with col2:
                        st.write(f"**Prediction:** {pick['prediction']:.2%}")
                        st.write(f"**Confidence:** {pick['confidence']:.1%}")
                        st.write(f"**Risk:** {pick['risk_level']}")
                    
                    with col3:
                        st.write(f"**Tech Score:** {pick['technical_score']}/100")
                        st.write(f"**Fund Score:** {pick['fundamental_score']}/100")
                        st.write(f"**Sector:** {pick['sector']}")
                    
                    # Signals
                    st.write("**Trading Signals:**")
                    for signal in pick['signals']:
                        st.write(f"• {signal}")
        
        # Detailed results table
        st.subheader("📊 Complete Analysis Results")
        df = pd.DataFrame(results)
        display_df = df[['symbol', 'current_price', 'price_change_1d', 'recommendation', 
                        'action', 'prediction', 'confidence', 'risk_level', 'technical_score', 'fundamental_score']]
        st.dataframe(display_df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction vs Confidence scatter
            fig = px.scatter(df, x='prediction', y='confidence', 
                           color='recommendation', size='technical_score',
                           hover_data=['symbol', 'action'],
                           title="Prediction vs Confidence")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk distribution
            risk_counts = df['risk_level'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                        title="Risk Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation distribution
        rec_counts = df['recommendation'].value_counts()
        fig = px.bar(x=rec_counts.index, y=rec_counts.values, 
                    title="Recommendation Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👆 Click 'Start Automated Analysis' to begin!")
        
        # Show sample of what will be analyzed
        st.subheader("📋 Stocks to be Analyzed")
        sample_stocks = analyzer.stock_universe[:15]
        st.write(", ".join(sample_stocks))
        
        st.subheader("🔍 What the System Does Automatically:")
        st.write("""
        1. **Fetches Data** - Downloads real-time stock data from Yahoo Finance
        2. **Technical Analysis** - Calculates RSI, MACD, Moving Averages, Bollinger Bands
        3. **Fundamental Analysis** - Analyzes P/E ratios, growth rates, financial health
        4. **ML Predictions** - Uses Random Forest, XGBoost, Gradient Boosting
        5. **Signal Generation** - Creates buy/sell/hold signals automatically
        6. **Risk Assessment** - Evaluates risk levels for each stock
        7. **Ranking** - Ranks stocks by potential returns and confidence
        8. **Recommendations** - Provides specific actions (BUY NOW, HOLD, SELL)
        """)

if __name__ == "__main__":
    main()
