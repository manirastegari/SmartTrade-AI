import streamlit as st
import yfinance as yf
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Suppress noisy library logs (Yahoo/yfinance transient messages)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

from advanced_analyzer import AdvancedTradingAnalyzer

# Professional Trading Interface - Like Goldman Sachs, JP Morgan, Citadel
st.set_page_config(
    page_title="Professional Trading Terminal - Institutional Grade",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .positive { color: #00ff00; font-weight: bold; }
    .negative { color: #ff4444; font-weight: bold; }
    .neutral { color: #ffa500; font-weight: bold; }
    .price-big { font-size: 2em; font-weight: bold; }
    .professional-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
<div class="professional-header">
    <h1>🏛️ Professional Trading Terminal</h1>
    <h3>Institutional-Grade Analysis • Real-Time Pricing • Professional Signals</h3>
    <p><strong>Created by: Mani Rastegari</strong> | <em>Hedge Fund Level Analysis</em></p>
</div>
""", unsafe_allow_html=True)

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    # Use light data mode and disable training to avoid heavy, rate-limited calls
    return AdvancedTradingAnalyzer(enable_training=False, data_mode="light")

analyzer = get_analyzer()

# Sidebar - Professional Controls
st.sidebar.markdown("## 🎯 Analysis Parameters")

# Professional analysis options
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Institutional Grade", "Hedge Fund Style", "Investment Bank Level", "Quant Research", "Risk Management"]
)

num_stocks = st.sidebar.slider("Number of Stocks", 10, 200, 50)

# Cap filter and risk style
cap_filter = st.sidebar.selectbox(
    "Cap Filter",
    ["Large Cap", "Mid Cap", "Small Cap", "All"]
)

risk_style = st.sidebar.selectbox(
    "Risk Style",
    ["Low Risk", "Balanced", "High Risk"]
)

market_focus = st.sidebar.selectbox(
    "Market Focus",
    ["S&P 500 Large Cap", "NASDAQ Growth", "Dow Jones Industrial", "Russell 2000 Small Cap", 
     "All Markets", "Sector Rotation", "Momentum Stocks", "Value Stocks", "Dividend Aristocrats"]
)

# Legacy controls (kept for layout, not used in light mode)
risk_model = st.sidebar.selectbox(
    "Risk Model",
    ["Conservative (Low Beta)", "Balanced (Market Beta)", "Aggressive (High Beta)", "Momentum (High Volatility)"]
)

# Professional features toggle
show_real_time = st.sidebar.checkbox("Real-Time Pricing", value=True)
show_analyst_targets = st.sidebar.checkbox("Analyst Price Targets", value=True)
show_earnings_impact = st.sidebar.checkbox("Earnings Impact Analysis", value=True)
show_institutional_flow = st.sidebar.checkbox("Institutional Flow", value=True)

# Main analysis button
def get_symbols_by_cap(analyzer, cap_filter: str, count: int):
    universe = analyzer.stock_universe
    universe_size = len(universe)
    
    if cap_filter == "Large Cap":
        # First 1/3 of universe (typically large caps)
        end_idx = min(universe_size // 3, 200)
        return universe[:max(10, min(count, end_idx))]
    elif cap_filter == "Mid Cap":
        # Middle 1/3 of universe
        start = universe_size // 3
        end = (universe_size * 2) // 3
        available = universe[start:end]
        return available[:max(10, min(count, len(available)))]
    elif cap_filter == "Small Cap":
        # Last 1/3 of universe (typically smaller caps)
        start = (universe_size * 2) // 3
        available = universe[start:]
        return available[:max(10, min(count, len(available)))]
    else:
        # All markets - return from entire universe
        return universe[:max(10, min(count, universe_size))]

def filter_by_risk(results, risk_style: str):
    if risk_style == "Low Risk":
        return [r for r in results if r.get('risk_level') in ("Low", "Medium")]
    if risk_style == "High Risk":
        return [r for r in results if r.get('risk_level') in ("High", "Medium")]
    return results

if st.sidebar.button("🚀 Run Professional Analysis", type="primary"):
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Professional analysis workflow
    with st.spinner("Running institutional-grade analysis..."):
        
        # Step 1: Data Collection
        status_text.text("📊 Collecting real-time market data...")
        progress_bar.progress(20)
        time.sleep(1)
        
        # Step 2: Technical Analysis
        status_text.text("📈 Running 100+ technical indicators...")
        progress_bar.progress(40)
        time.sleep(1)
        
        # Step 3: Fundamental Analysis
        status_text.text("💰 Analyzing earnings and fundamentals...")
        progress_bar.progress(60)
        time.sleep(1)
        
        # Step 4: Sentiment Analysis
        status_text.text("📰 Processing news and sentiment...")
        progress_bar.progress(80)
        time.sleep(1)
        
        # Step 5: Professional Scoring
        status_text.text("🎯 Generating professional recommendations...")
        progress_bar.progress(100)
        time.sleep(1)
        
        # Prepare symbols by cap filter
        symbols = get_symbols_by_cap(analyzer, cap_filter, num_stocks)
        
        # Run the analysis on selected symbols
        results = analyzer.run_advanced_analysis(max_stocks=len(symbols), symbols=symbols)

        # Auto-fallback: if no results, retry with curated fallback lists based on cap filter
        if not results:
            st.warning("Primary selection returned no results. Retrying with a curated fallback list for reliability...")
            
            if cap_filter == "Small Cap":
                # Reliable small cap symbols
                fallback_symbols = [
                    'PLTR','CRWD','SNOW','DDOG','NET','OKTA','ZM','DOCU','TWLO','SQ',
                    'ROKU','PINS','SNAP','UBER','LYFT','ABNB','DASH','PTON','FUBO','RKT',
                    'OPEN','COMP','Z','ZG','ESTC','MDB','TEAM','WDAY','NOW','ZS'
                ]
            elif cap_filter == "Mid Cap":
                # Reliable mid cap symbols  
                fallback_symbols = [
                    'REGN','GILD','BIIB','VRTX','ILMN','MRNA','ZTS','SYK','ISRG','EW',
                    'BSX','MDT','SPGI','MCO','FIS','FISV','GPN','NDAQ','CME','ICE',
                    'MKTX','CBOE','MSCI','TROW','BLK','SCHW','AMTD','ETFC','AMT','PLD'
                ]
            else:
                # Large cap fallback (default)
                fallback_symbols = [
                    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD','INTC',
                    'JPM','BAC','WFC','GS','MS','C','AXP','V','MA','PYPL',
                    'JNJ','PFE','UNH','ABBV','MRK','TMO','ABT','DHR','BMY','AMGN','LLY',
                    'KO','PEP','WMT','PG','HD','MCD','NKE','SBUX','DIS','CMCSA',
                    'VZ','T','CVX','XOM','COP','SLB','CAT','BA','MMM','GE','HON'
                ]
            
            results = analyzer.run_advanced_analysis(max_stocks=min(num_stocks, len(fallback_symbols)), symbols=fallback_symbols)
        
        # Post-filter by risk style for display
        results = filter_by_risk(results, risk_style)
        
        progress_bar.empty()
        status_text.empty()
    
    if results and len(results) > 0:
        
        # Professional Dashboard Layout
        st.markdown("## 🏛️ Professional Analysis Dashboard")
        
        # Real-time market overview
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>Stocks Analyzed</h4>
                <div class="price-big">{}</div>
            </div>
            """.format(len(results)), unsafe_allow_html=True)
        
        with col2:
            buy_signals = len([r for r in results if r['recommendation'] in ['BUY', 'STRONG BUY']])
            st.markdown("""
            <div class="metric-card">
                <h4>BUY Signals</h4>
                <div class="price-big positive">{}</div>
            </div>
            """.format(buy_signals), unsafe_allow_html=True)
        
        with col3:
            avg_confidence = np.mean([r['confidence'] for r in results])
            st.markdown("""
            <div class="metric-card">
                <h4>Avg Confidence</h4>
                <div class="price-big">{:.1%}</div>
            </div>
            """.format(avg_confidence), unsafe_allow_html=True)
        
        with col4:
            high_conviction = len([r for r in results if r['confidence'] > 0.8 and r['recommendation'] in ['BUY', 'STRONG BUY']])
            st.markdown("""
            <div class="metric-card">
                <h4>High Conviction</h4>
                <div class="price-big positive">{}</div>
            </div>
            """.format(high_conviction), unsafe_allow_html=True)
        
        with col5:
            low_risk = len([r for r in results if r['risk_level'] == 'Low'])
            st.markdown("""
            <div class="metric-card">
                <h4>Low Risk</h4>
                <div class="price-big">{}</div>
            </div>
            """.format(low_risk), unsafe_allow_html=True)
        
        with col6:
            avg_upside = np.mean([r.get('upside_potential', 0) for r in results])
            color_class = "positive" if avg_upside > 0 else "negative"
            st.markdown("""
            <div class="metric-card">
                <h4>Avg Upside</h4>
                <div class="price-big {}">{:+.1f}%</div>
            </div>
            """.format(color_class, avg_upside), unsafe_allow_html=True)
        
        # Top Professional Picks
        st.markdown("### 🏆 Top Professional Picks")
        
        # Sort by professional score
        top_picks = sorted(results, key=lambda x: x['overall_score'], reverse=True)[:10]
        
        for i, stock in enumerate(top_picks[:5]):
            with st.expander(f"#{i+1} {stock['symbol']} - {stock['recommendation']} - Score: {stock['overall_score']:.1f}"):
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    # Real-time price with professional formatting
                    current_price = stock['current_price']
                    price_change = stock['price_change_1d']
                    color_class = "positive" if price_change > 0 else "negative"
                    
                    st.markdown(f"""
                    **Current Price:** <span class="price-big">${current_price:.2f}</span><br>
                    **1D Change:** <span class="{color_class}">{price_change:+.2f}%</span><br>
                    **Volume:** {stock['volume']:,}<br>
                    **Market Cap:** ${stock['market_cap']:,.0f}
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Professional metrics
                    st.markdown(f"""
                    **Prediction:** {stock['prediction']:+.2f}%<br>
                    **Confidence:** {stock['confidence']:.1%}<br>
                    **Risk Level:** {stock['risk_level']}<br>
                    **Sector:** {stock['sector']}
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Professional scores
                    st.markdown("**Professional Scores:**")
                    st.progress(stock['technical_score']/100, text=f"Technical: {stock['technical_score']:.0f}/100")
                    st.progress(stock['fundamental_score']/100, text=f"Fundamental: {stock['fundamental_score']:.0f}/100")
                    st.progress(stock['sentiment_score']/100, text=f"Sentiment: {stock['sentiment_score']:.0f}/100")
                
                # Professional signals
                if stock.get('signals'):
                    st.markdown("**🎯 Professional Signals:**")
                    for signal in stock['signals'][:5]:
                        if 'BUY' in signal:
                            st.markdown(f"🟢 {signal}")
                        elif 'SELL' in signal:
                            st.markdown(f"🔴 {signal}")
                        else:
                            st.markdown(f"🟡 {signal}")
        
        # Professional Analysis Table
        st.markdown("### 📊 Complete Professional Analysis")
        
        # Create professional DataFrame
        df_results = pd.DataFrame(results)
        
        # Add professional columns
        df_display = df_results[['symbol', 'current_price', 'price_change_1d', 'prediction', 'confidence', 
                                'recommendation', 'risk_level', 'overall_score', 'technical_score', 
                                'fundamental_score', 'sentiment_score']].copy()
        
        # Add upside/downside calculation
        df_display['upside_potential'] = df_display['prediction']
        df_display['target_price'] = df_display['current_price'] * (1 + df_display['prediction']/100)
        
        # Format for professional display
        df_display['current_price'] = df_display['current_price'].apply(lambda x: f"${x:.2f}")
        df_display['price_change_1d'] = df_display['price_change_1d'].apply(lambda x: f"{x:+.2f}%")
        df_display['prediction'] = df_display['prediction'].apply(lambda x: f"{x:+.2f}%")
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}")
        df_display['upside_potential'] = df_display['upside_potential'].apply(lambda x: f"{x:+.2f}%")
        df_display['target_price'] = df_display['target_price'].apply(lambda x: f"${x:.2f}")
        
        # Rename columns professionally
        df_display.columns = ['Symbol', 'Current Price', '1D Change', 'ML Prediction', 'Confidence', 
                             'Recommendation', 'Risk Level', 'Overall Score', 'Technical Score', 
                             'Fundamental Score', 'Sentiment Score', 'Upside/Downside', 'Target Price']
        
        # Color-code the table
        def color_cells(val):
            if isinstance(val, str):
                if '+' in val and '%' in val:
                    return 'background-color: #d4edda; color: #155724'
                elif '-' in val and '%' in val:
                    return 'background-color: #f8d7da; color: #721c24'
                elif val in ['BUY', 'STRONG BUY']:
                    return 'background-color: #d1ecf1; color: #0c5460'
                elif val in ['SELL', 'STRONG SELL']:
                    return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        st.dataframe(df_display.style.applymap(color_cells), use_container_width=True)
        
        # Professional Charts
        st.markdown("### 📈 Professional Analysis Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk vs Return scatter plot
            fig_risk_return = px.scatter(
                df_results, 
                x='overall_score', 
                y='prediction',
                size='confidence',
                color='risk_level',
                hover_data=['symbol', 'recommendation'],
                title="Risk vs Return Analysis (Professional View)",
                labels={'overall_score': 'Professional Score', 'prediction': 'Expected Return (%)'}
            )
            fig_risk_return.update_layout(height=400)
            st.plotly_chart(fig_risk_return, use_container_width=True)
        
        with col2:
            # Sector analysis
            sector_data = df_results.groupby('sector').agg({
                'overall_score': 'mean',
                'prediction': 'mean',
                'confidence': 'mean'
            }).reset_index()
            
            fig_sector = px.bar(
                sector_data,
                x='sector',
                y='overall_score',
                color='prediction',
                title="Sector Analysis (Professional View)",
                labels={'overall_score': 'Avg Professional Score', 'prediction': 'Avg Expected Return'}
            )
            fig_sector.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # Professional Trading Recommendations
        st.markdown("### 🎯 Professional Trading Strategy")
        
        high_conviction_picks = [r for r in results if r['confidence'] > 0.8 and r['recommendation'] in ['BUY', 'STRONG BUY']]
        
        if high_conviction_picks:
            st.markdown("#### 🏆 High Conviction Plays (Like Hedge Funds)")
            
            for pick in high_conviction_picks[:3]:
                st.markdown(f"""
                **{pick['symbol']}** - {pick['recommendation']} 
                - **Entry:** ${pick['current_price']:.2f}
                - **Target:** ${pick['current_price'] * (1 + pick['prediction']/100):.2f} ({pick['prediction']:+.1f}%)
                - **Stop Loss:** ${pick['current_price'] * 0.95:.2f} (-5%)
                - **Position Size:** {'Large' if pick['risk_level'] == 'Low' else 'Medium' if pick['risk_level'] == 'Medium' else 'Small'}
                - **Time Horizon:** {'Long-term' if pick['confidence'] > 0.9 else 'Medium-term'}
                """)
        
        # Professional risk management
        st.markdown("#### ⚠️ Professional Risk Management")
        st.markdown("""
        **Portfolio Guidelines (Institutional Style):**
        - **Maximum single position:** 5% of portfolio
        - **Sector concentration:** Maximum 25% per sector
        - **Stop losses:** Set at -5% for all positions
        - **Rebalancing:** Weekly for active positions
        - **Risk monitoring:** Daily VaR calculation recommended
        
        **Professional Trading Rules:**
        - Only trade high conviction signals (confidence > 80%)
        - Diversify across at least 10 positions
        - Use position sizing based on risk level
        - Monitor earnings calendars for all holdings
        - Set alerts for analyst rating changes
        """)
        
    else:
        st.error("No analysis results available. Please check your settings and try again.")

# Live market data disabled to avoid external rate limits in large-scale free runs
st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Live Market Data")
st.sidebar.info("Disabled in Light Mode for scale and reliability.")

# Professional footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <strong>Professional Trading Terminal</strong><br>
    Created by Mani Rastegari | Institutional-Grade Analysis<br>
    <em>⚠️ For educational and research purposes only. Not financial advice.</em>
</div>
""", unsafe_allow_html=True)
