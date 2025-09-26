"""
Advanced Trading Analyzer App - Maximum Free Analysis Power
Analyzes 1000+ stocks with comprehensive free data sources and advanced ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from advanced_analyzer import AdvancedTradingAnalyzer

def main():
    st.set_page_config(
        page_title="Advanced Trading Analyzer",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Advanced Trading Analyzer - Maximum Free Analysis Power")
    st.markdown("**Analyzes 1000+ stocks with comprehensive free data sources and advanced ML**")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    
    analyzer = st.session_state.analyzer
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎯 Advanced Analysis Controls")
        
        # Advanced settings
        enable_ml_training = st.checkbox("Enable ML Training (longer, more accurate)", value=False)
        analyzer.enable_training = bool(enable_ml_training)

        # Analysis settings with dynamic upper bound
        max_available = max(50, min(len(analyzer.stock_universe), 1200))
        default_count = min(300, max_available)
        max_stocks = st.slider("Number of Stocks to Analyze", 20, max_available, default_count)
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive Analysis", "Quick Analysis", "Deep Dive Analysis", "Sector Analysis", "Risk Analysis"]
        )
        
        # Risk tolerance
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive", "Very Aggressive"]
        )
        
        # Market focus
        market_focus = st.multiselect(
            "Market Focus",
            ["US Large Cap", "US Mid Cap", "US Small Cap", "Canadian Stocks", "Growth Stocks", "Value Stocks", "Tech Stocks", "Healthcare Stocks", "Financial Stocks", "Energy Stocks"],
            default=["US Large Cap", "Canadian Stocks", "Tech Stocks"]
        )
        
        # Advanced settings
        st.subheader("🔧 Advanced Settings")
        use_ml_models = st.checkbox("Use Advanced ML Models", value=True)
        use_sentiment_analysis = st.checkbox("Use Sentiment Analysis", value=True)
        use_pattern_recognition = st.checkbox("Use Pattern Recognition", value=True)
        use_sector_analysis = st.checkbox("Use Sector Analysis", value=True)
        use_risk_management = st.checkbox("Use Risk Management", value=True)
        
        if st.button("🚀 Start Advanced Analysis", type="primary"):
            with st.spinner("Running advanced analysis..."):
                results = analyzer.run_advanced_analysis(max_stocks)
                st.session_state.results = results
        
        if st.button("🔄 Refresh Analysis"):
            with st.spinner("Refreshing analysis..."):
                results = analyzer.run_advanced_analysis(max_stocks)
                st.session_state.results = results
    
    # Main dashboard
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        # Summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Stocks Analyzed", len(results))
            strong_buys = len([r for r in results if 'STRONG BUY' in r['recommendation']])
            st.metric("Strong Buys", strong_buys)
        
        with col2:
            avg_prediction = np.mean([r['prediction'] for r in results])
            st.metric("Avg Prediction", f"{avg_prediction:.2f}%")
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
        
        with col5:
            avg_overall = np.mean([r['overall_score'] for r in results])
            st.metric("Avg Overall Score", f"{avg_overall:.1f}")
            avg_sentiment = np.mean([r['sentiment_score'] for r in results])
            st.metric("Avg Sentiment", f"{avg_sentiment:.1f}")
        
        with col6:
            avg_sector = np.mean([r['sector_score'] for r in results])
            st.metric("Avg Sector Score", f"{avg_sector:.1f}")
            avg_analyst = np.mean([r['analyst_score'] for r in results])
            st.metric("Avg Analyst Score", f"{avg_analyst:.1f}")
        
        # Top picks
        st.subheader("🏆 Top Stock Picks (Advanced Analysis)")
        top_picks = analyzer.get_top_picks_advanced(results, 30)
        
        if top_picks:
            for i, pick in enumerate(top_picks, 1):
                with st.expander(f"#{i} {pick['symbol']} - {pick['recommendation']} - {pick['action']} (Score: {pick['overall_score']:.1f})"):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.write(f"**Price:** ${pick['current_price']:.2f}")
                        st.write(f"**Change:** {pick['price_change_1d']:.2f}%")
                        st.write(f"**Volume:** {pick['volume']:,.0f}")
                        st.write(f"**Market Cap:** ${pick['market_cap']:,.0f}")
                    
                    with col2:
                        st.write(f"**Prediction:** {pick['prediction']:.2f}%")
                        st.write(f"**Confidence:** {pick['confidence']:.1%}")
                        st.write(f"**Risk:** {pick['risk_level']}")
                        st.write(f"**Sector:** {pick['sector']}")
                    
                    with col3:
                        st.write(f"**Tech Score:** {pick['technical_score']}/100")
                        st.write(f"**Fund Score:** {pick['fundamental_score']}/100")
                        st.write(f"**Sentiment:** {pick['sentiment_score']}/100")
                        st.write(f"**Momentum:** {pick['momentum_score']}/100")
                    
                    with col4:
                        st.write(f"**Volume Score:** {pick['volume_score']}/100")
                        st.write(f"**Volatility:** {pick['volatility_score']}/100")
                        st.write(f"**Sector Score:** {pick['sector_score']}/100")
                        st.write(f"**Analyst Score:** {pick['analyst_score']}/100")
                    
                    with col5:
                        st.write(f"**Overall:** {pick['overall_score']}/100")
                        st.write(f"**P/E Ratio:** {pick['pe_ratio']:.2f}")
                        st.write(f"**Last Updated:** {pick['last_updated'].strftime('%Y-%m-%d %H:%M')}")
                    
                    # Enhanced signals
                    st.write("**Advanced Trading Signals:**")
                    for signal in pick['signals'][:15]:  # Show top 15 signals
                        st.write(f"• {signal}")
        
        # Detailed results table
        st.subheader("📊 Complete Analysis Results")
        df = pd.DataFrame(results)
        display_df = df[['symbol', 'current_price', 'price_change_1d', 'recommendation', 
                        'action', 'prediction', 'confidence', 'risk_level', 'technical_score', 
                        'fundamental_score', 'sentiment_score', 'sector_score', 'analyst_score', 'overall_score']]
        st.dataframe(display_df, use_container_width=True)
        
        # Advanced charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction vs Confidence scatter
            fig = px.scatter(df, x='prediction', y='confidence', 
                           color='recommendation', size='overall_score',
                           hover_data=['symbol', 'action', 'technical_score', 'fundamental_score', 'sector_score'],
                           title="Prediction vs Confidence (Advanced)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk distribution
            risk_counts = df['risk_level'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                        title="Risk Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Score analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Technical vs Fundamental scores
            fig = px.scatter(df, x='technical_score', y='fundamental_score', 
                           color='recommendation', size='overall_score',
                           hover_data=['symbol', 'action'],
                           title="Technical vs Fundamental Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment vs Sector scores
            fig = px.scatter(df, x='sentiment_score', y='sector_score', 
                           color='recommendation', size='overall_score',
                           hover_data=['symbol', 'action'],
                           title="Sentiment vs Sector Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation distribution
        rec_counts = df['recommendation'].value_counts()
        fig = px.bar(x=rec_counts.index, y=rec_counts.values, 
                    title="Recommendation Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector analysis
        if 'sector' in df.columns:
            sector_analysis = df.groupby('sector').agg({
                'prediction': 'mean',
                'confidence': 'mean',
                'overall_score': 'mean',
                'technical_score': 'mean',
                'fundamental_score': 'mean',
                'sentiment_score': 'mean',
                'sector_score': 'mean',
                'analyst_score': 'mean',
                'symbol': 'count'
            }).round(2)
            sector_analysis.columns = ['Avg Prediction', 'Avg Confidence', 'Avg Overall Score', 
                                     'Avg Tech Score', 'Avg Fund Score', 'Avg Sentiment Score', 
                                     'Avg Sector Score', 'Avg Analyst Score', 'Count']
            sector_analysis = sector_analysis.sort_values('Avg Overall Score', ascending=False)
            
            st.subheader("📈 Advanced Sector Analysis")
            st.dataframe(sector_analysis)
        
        # Market overview
        st.subheader("🌍 Advanced Market Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Market Cap", f"${df['market_cap'].sum():,.0f}")
            st.metric("Avg P/E Ratio", f"{df['pe_ratio'].mean():.2f}")
        
        with col2:
            st.metric("High Confidence Picks", len(df[df['confidence'] > 0.7]))
            st.metric("Low Risk Picks", len(df[df['risk_level'] == 'Low']))
        
        with col3:
            st.metric("Positive Predictions", len(df[df['prediction'] > 0]))
            st.metric("Strong Fundamentals", len(df[df['fundamental_score'] > 70]))
        
        with col4:
            st.metric("High Sentiment", len(df[df['sentiment_score'] > 70]))
            st.metric("Strong Sectors", len(df[df['sector_score'] > 70]))
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Technical Score", f"{df['technical_score'].mean():.1f}")
            st.metric("Avg Fundamental Score", f"{df['fundamental_score'].mean():.1f}")
        
        with col2:
            st.metric("Avg Sentiment Score", f"{df['sentiment_score'].mean():.1f}")
            st.metric("Avg Sector Score", f"{df['sector_score'].mean():.1f}")
        
        with col3:
            st.metric("Avg Analyst Score", f"{df['analyst_score'].mean():.1f}")
            st.metric("Avg Overall Score", f"{df['overall_score'].mean():.1f}")
        
    else:
        st.info("👆 Click 'Start Advanced Analysis' to begin!")
        
        # Show sample of what will be analyzed
        st.subheader("📋 Advanced Analysis Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Advanced Technical Analysis (100+ Indicators):**")
            st.write("• RSI (14, 21, 30, 50 periods)")
            st.write("• MACD (multiple timeframes)")
            st.write("• Moving Averages (5, 10, 20, 50, 100, 200)")
            st.write("• Bollinger Bands (multiple periods)")
            st.write("• Stochastic Oscillator")
            st.write("• Williams %R")
            st.write("• Commodity Channel Index")
            st.write("• Average True Range")
            st.write("• Average Directional Index")
            st.write("• Money Flow Index")
            st.write("• On Balance Volume")
            st.write("• Accumulation/Distribution")
            st.write("• Chaikin Money Flow")
            st.write("• Ichimoku Cloud")
            st.write("• Fibonacci Retracements")
            st.write("• Pivot Points")
            st.write("• Volume Profile")
            st.write("• Candlestick Patterns")
            st.write("• Support/Resistance Levels")
            st.write("• Market Structure Analysis")
            st.write("• Volatility Indicators")
            st.write("• Trend Strength Analysis")
        
        with col2:
            st.write("**📰 Enhanced News & Sentiment Analysis:**")
            st.write("• Yahoo Finance News")
            st.write("• Google News Scraping")
            st.write("• Reddit Sentiment")
            st.write("• Twitter Sentiment")
            st.write("• VADER Sentiment Analysis")
            st.write("• FinBERT Financial Sentiment")
            st.write("• Overall Sentiment Score")
            st.write("• News Count Analysis")
            
            st.write("**🏢 Advanced Fundamental Analysis:**")
            st.write("• P/E, P/B, P/S Ratios")
            st.write("• PEG Ratio")
            st.write("• Dividend Yield")
            st.write("• Revenue Growth")
            st.write("• Earnings Growth")
            st.write("• Profit Margins")
            st.write("• Return on Equity")
            st.write("• Debt-to-Equity")
            st.write("• Market Cap Analysis")
            
            st.write("**🔍 Advanced Analysis:**")
            st.write("• Insider Trading Activity")
            st.write("• Options Data (Put/Call Ratio)")
            st.write("• Institutional Holdings")
            st.write("• Economic Indicators")
            st.write("• VIX Analysis")
            st.write("• Fed Rate Impact")
            st.write("• GDP Growth Analysis")
            st.write("• Inflation Impact")
            st.write("• Sector Rotation Analysis")
            st.write("• Analyst Ratings")
            st.write("• Price Targets")
        
        st.subheader("🎯 What Makes This Advanced:")
        st.write("""
        1. **1000+ Stocks** - Analyzes major US and Canadian stocks
        2. **100+ Technical Indicators** - Comprehensive technical analysis
        3. **Multi-Source News** - Yahoo, Google, Reddit, Twitter sentiment
        4. **Advanced Sentiment** - VADER, FinBERT, multi-method analysis
        5. **Insider Trading** - Tracks insider buying/selling activity
        6. **Options Analysis** - Put/call ratios and implied volatility
        7. **Institutional Data** - Tracks institutional holdings
        8. **Economic Indicators** - VIX, Fed rates, GDP, inflation
        9. **Advanced ML** - 9 different machine learning models
        10. **Risk Assessment** - Multi-factor risk analysis
        11. **Sector Analysis** - Sector rotation and performance
        12. **Analyst Analysis** - Analyst ratings and price targets
        13. **Pattern Recognition** - Candlestick and chart patterns
        14. **Market Structure** - Support/resistance, breakouts
        15. **Real-time Updates** - Fresh data every analysis
        """)

if __name__ == "__main__":
    main()
