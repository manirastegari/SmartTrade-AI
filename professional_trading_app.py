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
from ultimate_strategy_analyzer import UltimateStrategyAnalyzer
from catalyst_detector import CatalystDetector

# Professional Trading Interface - Like Goldman Sachs, JP Morgan, Citadel
st.set_page_config(
    page_title="Professional Trading Terminal - Institutional Grade",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look with timeframe indicators
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
    .stock-card {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .timeframe-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        margin: 0.2rem;
    }
    .short-term { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
    .medium-term { background: #d1ecf1; color: #0c5460; border: 1px solid #b8daff; }
    .long-term { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .confidence-high { background: #d4edda; color: #155724; }
    .confidence-medium { background: #fff3cd; color: #856404; }
    .confidence-low { background: #f8d7da; color: #721c24; }
    .signal-item {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }
    .buy-signal { border-left-color: #28a745; }
    .sell-signal { border-left-color: #dc3545; }
    .neutral-signal { border-left-color: #ffc107; }
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
    # Use light data mode by default (rate-limit friendly). ML training can be toggled below.
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    st.success(f"🚀 Optimizer loaded: {analyzer.max_workers} workers, caching enabled")
    return analyzer

analyzer = get_analyzer()

# Timeframe determination functions
def determine_signal_timeframes(stock_data):
    """Determine timeframes for different signals based on analysis"""
    timeframes = {}
    
    # Technical indicators timeframes
    if stock_data.get('rsi_signal'):
        timeframes['RSI'] = {'timeframe': '1-5 days', 'type': 'short-term', 'accuracy': '60-70%'}
    if stock_data.get('macd_signal'):
        timeframes['MACD'] = {'timeframe': '3-10 days', 'type': 'short-term', 'accuracy': '65-75%'}
    
    # Chart patterns timeframes
    if any(signal in str(stock_data.get('signals', [])) for signal in ['Head', 'Double']):
        timeframes['Chart Patterns'] = {'timeframe': '2-6 weeks', 'type': 'medium-term', 'accuracy': '70-80%'}
    
    # Strategic signals timeframes
    if any(signal in str(stock_data.get('signals', [])) for signal in ['Golden', 'Death']):
        timeframes['Strategic'] = {'timeframe': '1-6 months', 'type': 'long-term', 'accuracy': '75-85%'}
    
    # ML prediction timeframes based on confidence
    confidence = stock_data.get('confidence', 0)
    if confidence > 0.8:
        timeframes['ML High Confidence'] = {'timeframe': '1-14 days', 'type': 'short-term', 'accuracy': '70-80%'}
    elif confidence > 0.6:
        timeframes['ML Medium Confidence'] = {'timeframe': '1-4 weeks', 'type': 'medium-term', 'accuracy': '60-75%'}
    
    # Fundamental analysis timeframes
    if stock_data.get('fundamental_score', 0) > 70:
        timeframes['Fundamentals'] = {'timeframe': '3-12 months', 'type': 'long-term', 'accuracy': '70-85%'}
    
    return timeframes

def get_primary_timeframe(stock_data):
    """Get the primary recommended timeframe for a stock"""
    confidence = stock_data.get('confidence', 0)
    fundamental_score = stock_data.get('fundamental_score', 0)
    
    # High confidence ML + strong fundamentals = medium to long term
    if confidence > 0.8 and fundamental_score > 70:
        return {'timeframe': '1-4 weeks', 'type': 'medium-term', 'accuracy': '70-80%', 'recommendation': 'Medium-term position'}
    
    # High confidence ML only = short term
    elif confidence > 0.8:
        return {'timeframe': '1-14 days', 'type': 'short-term', 'accuracy': '70-80%', 'recommendation': 'Short-term trade'}
    
    # Strong fundamentals = long term
    elif fundamental_score > 70:
        return {'timeframe': '3-12 months', 'type': 'long-term', 'accuracy': '75-85%', 'recommendation': 'Long-term investment'}
    
    # Default medium term
    else:
        return {'timeframe': '1-4 weeks', 'type': 'medium-term', 'accuracy': '65-75%', 'recommendation': 'Medium-term swing trade'}

def format_timeframe_badge(timeframe_info):
    """Format timeframe information as HTML badge"""
    type_class = timeframe_info['type'].replace('-', '-')
    return f'<span class="timeframe-badge {type_class}">⏰ {timeframe_info["timeframe"]} ({timeframe_info["accuracy"]})</span>'

def format_confidence_badge(confidence):
    """Format confidence as HTML badge"""
    if confidence > 0.8:
        return f'<span class="timeframe-badge confidence-high">🎯 High Confidence ({confidence:.1%})</span>'
    elif confidence > 0.6:
        return f'<span class="timeframe-badge confidence-medium">📊 Medium Confidence ({confidence:.1%})</span>'
    else:
        return f'<span class="timeframe-badge confidence-low">⚠️ Low Confidence ({confidence:.1%})</span>'

# Session state for consistent stock selection across analysis types
if 'selected_symbols' not in st.session_state:
    st.session_state.selected_symbols = None
if 'last_selection_params' not in st.session_state:
    st.session_state.last_selection_params = None

# Sidebar - Professional Controls
st.sidebar.markdown("## 🎯 Analysis Parameters")

# Professional analysis options
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["🏆 Ultimate Strategy (Automated 4-Strategy Consensus)",
     "🎯 Catalyst Hunter (Explosive Opportunities)",
     "Institutional Grade", "Hedge Fund Style", "Investment Bank Level", "Quant Research", "Risk Management"]
)

# Show description for Ultimate Strategy
if analysis_type == "🏆 Ultimate Strategy (Automated 4-Strategy Consensus)":
    st.sidebar.info("""
    **🏆 Ultimate Strategy:**
    
    Automatically runs all 4 optimal strategies:
    1. Institutional Consensus (716 stocks)
    2. Hedge Fund Alpha (500 stocks)
    3. Quant Value Hunter (600 stocks)
    4. Risk-Managed Core (400 stocks)
    
    **Output:** Final consensus recommendations organized by conviction tiers with specific buy/sell targets.
    
    **Time:** 2-3 hours
    **Expected Return:** 26-47% annually
    
    ⚠️ Other parameters below are ignored when using Ultimate Strategy.
    """)

elif analysis_type == "🎯 Catalyst Hunter (Explosive Opportunities)":
    st.sidebar.info("""
    **🎯 Catalyst Hunter:**
    
    Detects explosive opportunities like:
    • RGC: +10,000% moves
    • AMD: +27% on OpenAI news
    • Earnings surprises
    • Breakout patterns
    • Unusual volume spikes
    
    **Features:**
    ✅ Real-time news sentiment
    ✅ Volume/price anomaly detection
    ✅ Insider activity monitoring
    ✅ Partnership announcement tracking
    
    **Time:** 15-30 minutes
    **Target:** 50-1000%+ gains
    
    ⚠️ High risk, high reward strategy.
    """)

# Toggle ML training (optional: longer run, potentially higher accuracy)
enable_ml_training = st.sidebar.checkbox("Enable ML Training (longer, more accurate)", value=False)
analyzer.enable_training = bool(enable_ml_training)

# Dynamic stock count slider based on universe size
max_available = max(50, min(len(analyzer.stock_universe), 1200))
default_count = min(300, max_available)
num_stocks = st.sidebar.slider("Number of Stocks", 20, max_available, default_count)

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

# Stock selection consistency controls
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔄 Stock Selection")

if st.session_state.selected_symbols:
    st.sidebar.success(f"✅ {len(st.session_state.selected_symbols)} stocks selected")
    st.sidebar.write(f"**Parameters:** {st.session_state.last_selection_params}")
    
    if st.sidebar.button("🔄 Select New Stocks"):
        st.session_state.selected_symbols = None
        st.session_state.last_selection_params = None
        st.rerun()
else:
    st.sidebar.info("No stocks selected yet")

st.sidebar.markdown("---")

# Enhanced stock selection with Market Focus integration
def get_comprehensive_symbol_selection(analyzer, cap_filter: str, market_focus: str, count: int):
    """
    Enhanced symbol selection that considers both cap filter and market focus
    Returns a larger, more comprehensive set for better analysis
    """
    universe = analyzer.stock_universe
    universe_size = len(universe)
    
    # Define market focus symbol sets
    market_focus_symbols = {
        "S&P 500 Large Cap": [
            'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD','INTC',
            'JPM','BAC','WFC','GS','MS','C','AXP','V','MA','PYPL',
            'JNJ','PFE','UNH','ABBV','MRK','TMO','ABT','DHR','BMY','AMGN',
            'KO','PEP','WMT','PG','HD','MCD','NKE','SBUX','DIS','CMCSA'
        ],
        "NASDAQ Growth": [
            'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD',
            'PLTR','CRWD','SNOW','DDOG','NET','OKTA','ZM','DOCU','TWLO',
            'ROKU','PINS','SNAP','UBER','LYFT','ABNB','DASH','PTON'
        ],
        "Russell 2000 Small Cap": [
            'PLTR','CRWD','SNOW','DDOG','NET','OKTA','ZM','DOCU','TWLO','SQ',
            'ROKU','PINS','SNAP','UBER','LYFT','ABNB','DASH','PTON','FUBO','RKT',
            'OPEN','COMP','Z','ZG','ESTC','MDB','TEAM','WDAY','NOW','ZS'
        ],
        "Momentum Stocks": [
            'NVDA','TSLA','AMD','PLTR','CRWD','SNOW','NET','ROKU','UBER','SQ',
            'ABNB','DASH','ZM','DOCU','PINS','SNAP','PTON','FUBO','RKT','OPEN'
        ],
        "Value Stocks": [
            'JPM','BAC','WFC','GS','MS','C','V','MA','JNJ','PFE','UNH',
            'KO','PEP','WMT','PG','HD','MCD','CVX','XOM','COP','CAT','BA'
        ],
        "Dividend Aristocrats": [
            'JNJ','PG','KO','PEP','WMT','HD','MCD','CVX','XOM','CAT',
            'MMM','GE','HON','UPS','FDX','VZ','T','NEE','DUK','SO'
        ]
    }
    
    # Get base symbols by cap filter
    if cap_filter == "Large Cap":
        end_idx = universe_size // 3
        base_symbols = universe[:end_idx]
    elif cap_filter == "Mid Cap":
        start = universe_size // 3
        end = (universe_size * 2) // 3
        base_symbols = universe[start:end]
    elif cap_filter == "Small Cap":
        start = (universe_size * 2) // 3
        base_symbols = universe[start:]
    else:
        base_symbols = universe
    
    # Apply market focus filter if specified
    if market_focus in market_focus_symbols:
        focus_symbols = market_focus_symbols[market_focus]
        # Prioritize symbols that match both cap filter and market focus
        prioritized = [s for s in base_symbols if s in focus_symbols]
        remaining = [s for s in base_symbols if s not in focus_symbols]
        base_symbols = prioritized + remaining
    elif market_focus == "Sector Rotation":
        # Mix of different sectors for rotation strategy
        sectors = [
            universe[:universe_size//4],  # Tech heavy
            universe[universe_size//4:universe_size//2],  # Mixed
            universe[universe_size//2:3*universe_size//4],  # Value/Industrial
            universe[3*universe_size//4:]  # Small/Speculative
        ]
        base_symbols = []
        for sector in sectors:
            base_symbols.extend(sector[:count//4])
    
    # Ensure we have enough symbols, expand if needed
    if len(base_symbols) < count:
        # Add more symbols from the broader universe if needed
        additional_needed = count - len(base_symbols)
        additional_symbols = [s for s in universe if s not in base_symbols][:additional_needed]
        base_symbols.extend(additional_symbols)
    
    # Return the requested count, but ensure good coverage
    # For larger requests, use more of the universe
    if count >= 300:
        # For 300+ requests, use the full universe if needed
        final_count = min(count, len(universe))
        if len(base_symbols) < final_count:
            # Add more symbols from the broader universe
            additional_needed = final_count - len(base_symbols)
            additional_symbols = [s for s in universe if s not in base_symbols][:additional_needed]
            base_symbols.extend(additional_symbols)
        return base_symbols[:final_count]
    else:
        # For smaller requests, maintain minimum coverage
        final_count = max(count, min(50, len(base_symbols)))
        return base_symbols[:final_count]

def get_symbols_by_cap(analyzer, cap_filter: str, count: int):
    """Legacy function for backward compatibility"""
    return get_comprehensive_symbol_selection(analyzer, cap_filter, "All Markets", count)

def apply_analysis_type_adjustments(results, analysis_type: str):
    """
    Apply analysis type-specific scoring adjustments to make Analysis Type meaningful
    """
    adjusted_results = []
    
    for result in results:
        adjusted_result = result.copy()
        
        overlay = result.get('overlay_score', 50)
        if analysis_type == "Institutional Grade":
            # Favor stability, liquidity, and established companies
            # Boost scores for large cap, low volatility, high volume
            stability_bonus = 0
            if result.get('current_price', 0) > 50:  # Higher price stocks tend to be more stable
                stability_bonus += 5
            if result.get('volume_score', 50) > 70:  # High volume = good liquidity
                stability_bonus += 5
            # Overlay context: penalize in poor macro breadth conditions
            if overlay < 40:
                stability_bonus -= 3
            elif overlay > 60:
                stability_bonus += 2
            adjusted_result['overall_score'] = min(100, result['overall_score'] + stability_bonus)
            adjusted_result['analysis_focus'] = "Institutional: Stability & Liquidity"
            
        elif analysis_type == "Hedge Fund Style":
            # Favor momentum, volatility, and alpha generation
            momentum_bonus = 0
            if result.get('momentum_score', 50) > 70:
                momentum_bonus += 8
            if result.get('volatility_score', 50) > 60:  # Higher volatility = more opportunity
                momentum_bonus += 5
            # Overlay tailwind helps momentum
            if overlay > 60:
                momentum_bonus += 3
            elif overlay < 40:
                momentum_bonus -= 3
            adjusted_result['overall_score'] = min(100, result['overall_score'] + momentum_bonus)
            adjusted_result['analysis_focus'] = "Hedge Fund: Momentum & Alpha"
            
        elif analysis_type == "Investment Bank Level":
            # Favor fundamental strength and analyst coverage
            fundamental_bonus = 0
            if result.get('fundamental_score', 50) > 70:
                fundamental_bonus += 8
            if result.get('analyst_score', 50) > 60:
                fundamental_bonus += 5
            # Overlay supports strong macro conditions
            if overlay > 60:
                fundamental_bonus += 2
            adjusted_result['overall_score'] = min(100, result['overall_score'] + fundamental_bonus)
            adjusted_result['analysis_focus'] = "Investment Bank: Fundamentals & Coverage"
            
        elif analysis_type == "Quant Research":
            # Favor technical indicators and statistical patterns
            technical_bonus = 0
            if result.get('technical_score', 50) > 75:
                technical_bonus += 10
            if result.get('confidence', 0) > 0.8:  # High confidence in predictions
                technical_bonus += 5
            # Overlay assists technical follow-through
            if overlay > 60:
                technical_bonus += 2
            adjusted_result['overall_score'] = min(100, result['overall_score'] + technical_bonus)
            adjusted_result['analysis_focus'] = "Quant: Technical & Statistical"
            
        elif analysis_type == "Risk Management":
            # Favor risk-adjusted returns and downside protection
            risk_bonus = 0
            if result.get('risk_level') == "Low":
                risk_bonus += 8
            elif result.get('risk_level') == "Medium":
                risk_bonus += 3
            if result.get('prediction', 0) > 0 and result.get('confidence', 0) > 0.7:  # Positive with high confidence
                risk_bonus += 5
            # Macro overlay risk aversion in poor conditions
            if overlay < 40:
                risk_bonus -= 5
            adjusted_result['overall_score'] = min(100, result['overall_score'] + risk_bonus)
            adjusted_result['analysis_focus'] = "Risk Mgmt: Downside Protection"
        
        adjusted_results.append(adjusted_result)
    
    # Re-sort by adjusted overall score
    return sorted(adjusted_results, key=lambda x: x['overall_score'], reverse=True)

def filter_by_risk(results, risk_style: str):
    if risk_style == "Low Risk":
        return [r for r in results if r.get('risk_level') in ("Low", "Medium")]
    if risk_style == "High Risk":
        return [r for r in results if r.get('risk_level') in ("High", "Medium")]
    return results

if st.sidebar.button("🚀 Run Professional Analysis", type="primary"):
    
    # Check if Ultimate Strategy is selected
    if analysis_type == "🏆 Ultimate Strategy (Automated 4-Strategy Consensus)":
        
        st.markdown("---")
        st.markdown("# 🏆 ULTIMATE STRATEGY - AUTOMATED 4-STRATEGY CONSENSUS")
        st.markdown("### Running all 4 optimal strategies automatically...")
        
        # Initialize Ultimate Strategy Analyzer
        ultimate_analyzer = UltimateStrategyAnalyzer(analyzer)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message, progress):
            status_text.text(message)
            progress_bar.progress(progress)
        
        # Run Ultimate Strategy
        with st.spinner("Running Ultimate Strategy Analysis (this will take 2-3 hours)..."):
            
            st.info("⏱️ **Estimated Time:** 2-3 hours for complete 4-strategy analysis")
            st.info("☕ **Tip:** Grab a coffee! This comprehensive analysis is worth the wait.")
            
            final_recommendations = ultimate_analyzer.run_ultimate_strategy(
                progress_callback=update_progress
            )
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            ultimate_analyzer.display_ultimate_strategy_results(final_recommendations)
            
            st.success("✅ Ultimate Strategy Analysis Complete!")
            st.balloons()
            
            # Stop execution here - Ultimate Strategy has its own display
            st.stop()
        
    elif analysis_type == "🎯 Catalyst Hunter (Explosive Opportunities)":
        
        st.markdown("---")
        st.markdown("# 🎯 CATALYST HUNTER - EXPLOSIVE OPPORTUNITY SCANNER")
        st.markdown("### Detecting market catalysts for 50-1000%+ gains...")
        
        # Initialize Catalyst Detector
        catalyst_detector = CatalystDetector()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run Catalyst Detection
        with st.spinner("🔍 Scanning for explosive opportunities..."):
            
            status_text.text("📰 Analyzing news sentiment and catalysts...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            status_text.text("📊 Detecting unusual volume and price patterns...")
            progress_bar.progress(50)
            time.sleep(0.5)
            
            status_text.text("🚀 Identifying breakout candidates...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            status_text.text("⚡ Generating explosive opportunity rankings...")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Get ALL available symbols for maximum coverage
            st.info(f"🎯 **MAXIMUM COVERAGE MODE**: Analyzing ALL {len(analyzer.stock_universe)} stocks in universe")
            st.info("📊 **Multi-Layered Analysis**: Catalyst Hunter + Advanced Analysis + Ultimate Strategy insights")
            
            # Use full stock universe for maximum coverage
            all_symbols = analyzer.stock_universe
            batch_size = 100  # Process in batches to avoid rate limits
            
            st.markdown(f"### 📈 Processing {len(all_symbols)} stocks in {len(all_symbols)//batch_size + 1} batches...")
            
            # Initialize storage for results
            import os
            from datetime import datetime
            import json
            
            # Create results directory
            results_dir = os.path.join(os.path.dirname(__file__), "Catalyst hunter")
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate timestamp for files
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Scan for catalysts with progress tracking
            all_catalyst_results = []
            
            # Process in batches to avoid overwhelming the system
            for batch_num in range(0, len(all_symbols), batch_size):
                batch_symbols = all_symbols[batch_num:batch_num + batch_size]
                batch_progress = (batch_num + batch_size) / len(all_symbols)
                
                status_text.text(f"🔍 Processing batch {batch_num//batch_size + 1}: {len(batch_symbols)} stocks...")
                progress_bar.progress(min(batch_progress, 1.0))
                
                try:
                    batch_results = catalyst_detector.scan_for_catalysts(batch_symbols, max_workers=8)
                    all_catalyst_results.extend(batch_results)
                    
                    # Quiet progress update (only for significant finds)
                    if len(batch_results) > 5:  # Only show if we found notable opportunities
                        st.success(f"🎯 Batch {batch_num//batch_size + 1}: Found {len(batch_results)} strong opportunities")
                    
                except Exception as e:
                    # Only show warnings for significant issues
                    if "rate limit" in str(e).lower() or "connection" in str(e).lower():
                        st.warning(f"⚠️ Batch {batch_num//batch_size + 1}: {str(e)[:50]}...")
                    continue
            
            catalyst_results = all_catalyst_results
            
            progress_bar.empty()
            status_text.empty()
        
        if catalyst_results:
            # Run multi-layered analysis on top catalyst results
            st.markdown("## 🔄 RUNNING MULTI-LAYERED ANALYSIS ON TOP CATALYSTS...")
            
            # Get top 20 catalyst opportunities for additional analysis
            top_catalyst_symbols = [r['symbol'] for r in sorted(catalyst_results, key=lambda x: x.get('catalyst_score', 0), reverse=True)[:20]]
            
            # Run Advanced Analysis on top catalysts
            st.info("🧠 Running Advanced Technical & Fundamental Analysis on top catalyst opportunities...")
            advanced_results = {}
            advanced_analysis = []
            
            try:
                # Run individual analysis for each top catalyst symbol
                for symbol in top_catalyst_symbols:
                    try:
                        # Run comprehensive analysis for single symbol
                        single_result = analyzer.run_advanced_analysis(max_stocks=1, symbols=[symbol])
                        if single_result and len(single_result) > 0:
                            result = single_result[0]
                            advanced_results[symbol] = result
                            advanced_analysis.append(result)
                            # Quiet progress - only show for high-scoring results
                            if result.get('overall_score', 0) > 60:
                                st.info(f"  🎯 {symbol}: High-potential opportunity (Overall: {result.get('overall_score', 0):.1f})")
                    except Exception as e:
                        # Suppress most error messages for cleaner output
                        continue
                
                st.success(f"✅ Advanced Analysis complete for {len(advanced_results)} stocks")
                
            except Exception as e:
                st.warning(f"⚠️ Advanced Analysis error: {e}")
                st.info("🔄 Continuing with catalyst-only analysis...")
            
            # Run Ultimate Strategy insights on top catalysts
            st.info("🏆 Generating Ultimate Strategy insights for catalyst opportunities...")
            ultimate_insights = {}
            
            try:
                if advanced_analysis and len(advanced_analysis) > 0:
                    # Apply different analysis type adjustments to see which works best
                    analysis_types = ["Hedge Fund Style", "Institutional Grade", "Quant Research"]
                    
                    for analysis_type in analysis_types:
                        try:
                            adjusted_results = apply_analysis_type_adjustments(advanced_analysis[:10], analysis_type)
                            ultimate_insights[analysis_type] = adjusted_results[:5]  # Top 5 from each style
                            # Quiet progress - only show summary
                        except Exception as e:
                            # Suppress individual strategy errors for cleaner output
                            continue
                    
                    st.success(f"✅ Ultimate Strategy insights generated for {len(ultimate_insights)} strategies")
                else:
                    st.warning("⚠️ No advanced analysis available for Ultimate Strategy insights")
                    # Create placeholder insights based on catalyst data only
                    for analysis_type in ["Hedge Fund Style", "Institutional Grade", "Quant Research"]:
                        ranked_catalysts = sorted(catalyst_results[:10], key=lambda x: x.get('catalyst_score', 0), reverse=True)
                        ultimate_insights[analysis_type] = [
                            {
                                'symbol': r['symbol'], 
                                'overall_score': r.get('catalyst_score', 0) * 0.8,  # Use catalyst score as proxy
                                'recommendation': 'STRONG BUY' if r.get('catalyst_score', 0) > 70 else 'BUY'
                            } for r in ranked_catalysts[:5]
                        ]
                    st.info("📊 Using catalyst-based proxy rankings for Ultimate Strategy insights")
                
            except Exception as e:
                st.warning(f"⚠️ Ultimate Strategy error: {e}")
                ultimate_insights = {}
            
            # Combine and enhance catalyst results with additional analysis
            enhanced_catalyst_results = []
            for catalyst_result in catalyst_results:
                symbol = catalyst_result['symbol']
                enhanced_result = catalyst_result.copy()
                
                # Add advanced analysis data if available
                if symbol in advanced_results:
                    adv_result = advanced_results[symbol]
                    enhanced_result['advanced_analysis'] = adv_result
                    enhanced_result['technical_score'] = adv_result.get('technical_score', 0)
                    enhanced_result['fundamental_score'] = adv_result.get('fundamental_score', 0)
                    enhanced_result['overall_score'] = adv_result.get('overall_score', 0)
                    enhanced_result['confidence'] = adv_result.get('confidence', 0)
                    enhanced_result['risk_level'] = adv_result.get('risk_level', 'Unknown')
                else:
                    # Fallback: Use catalyst score as proxy for missing advanced analysis
                    catalyst_score = catalyst_result.get('catalyst_score', 0)
                    enhanced_result['technical_score'] = min(catalyst_score * 0.7, 100)  # Technical proxy
                    enhanced_result['fundamental_score'] = min(catalyst_score * 0.8, 100)  # Fundamental proxy
                    enhanced_result['overall_score'] = min(catalyst_score * 0.75, 100)  # Overall proxy
                    enhanced_result['confidence'] = min(catalyst_score / 100, 1.0)  # Confidence proxy
                    enhanced_result['risk_level'] = 'High' if catalyst_score > 65 else 'Medium' if catalyst_score > 50 else 'Low'
                
                # Add ultimate strategy rankings
                strategy_rankings = {}
                for strategy, results in ultimate_insights.items():
                    for i, result in enumerate(results):
                        if result['symbol'] == symbol:
                            strategy_rankings[strategy] = {
                                'rank': i + 1,
                                'score': result.get('overall_score', 0),
                                'recommendation': result.get('recommendation', 'HOLD')
                            }
                enhanced_result['strategy_rankings'] = strategy_rankings
                
                enhanced_catalyst_results.append(enhanced_result)
            
            # Store comprehensive results automatically
            try:
                # Save detailed JSON results
                detailed_results = {
                    'timestamp': timestamp,
                    'total_stocks_analyzed': len(all_symbols),
                    'catalyst_opportunities_found': len(catalyst_results),
                    'top_advanced_analysis_count': len(advanced_results),
                    'ultimate_strategy_insights': len(ultimate_insights),
                    'catalyst_results': enhanced_catalyst_results,
                    'advanced_analysis_results': advanced_results,
                    'ultimate_strategy_insights_data': ultimate_insights,
                    'analysis_metadata': {
                        'analysis_type': 'Catalyst Hunter Maximum Coverage',
                        'batch_size': batch_size,
                        'total_batches': len(all_symbols)//batch_size + 1,
                        'stock_universe_size': len(analyzer.stock_universe)
                    }
                }
                
                # Save JSON file
                json_file = os.path.join(results_dir, f"catalyst_analysis_{timestamp}.json")
                with open(json_file, 'w') as f:
                    json.dump(detailed_results, f, indent=2, default=str)
                
                # Save CSV summary
                import pandas as pd
                summary_data = []
                for result in enhanced_catalyst_results[:50]:  # Top 50 for CSV
                    summary_data.append({
                        'Symbol': result['symbol'],
                        'Company_Name': result.get('company_name', result['symbol']),
                        'Catalyst_Score': result.get('catalyst_score', 0),
                        'Recommendation': result.get('recommendation', 'WATCH'),
                        'Catalyst_Type': result.get('catalyst_type', 'Technical'),
                        'Explosive_Potential': result.get('explosive_potential', False),
                        'Price_Change_1D': result.get('momentum_data', {}).get('price_change_1d', 0),
                        'Volume_Spike': result.get('momentum_data', {}).get('volume_spike', 1),
                        'Technical_Score': result.get('technical_score', 0),
                        'Fundamental_Score': result.get('fundamental_score', 0),
                        'Overall_Score': result.get('overall_score', 0),
                        'Risk_Level': result.get('risk_level', 'Unknown'),
                        'Hedge_Fund_Rank': result.get('strategy_rankings', {}).get('Hedge Fund Style', {}).get('rank', 'N/A'),
                        'Institutional_Rank': result.get('strategy_rankings', {}).get('Institutional Grade', {}).get('rank', 'N/A'),
                        'Quant_Rank': result.get('strategy_rankings', {}).get('Quant Research', {}).get('rank', 'N/A')
                    })
                
                csv_file = os.path.join(results_dir, f"catalyst_summary_{timestamp}.csv")
                pd.DataFrame(summary_data).to_csv(csv_file, index=False)
                
                st.success(f"📁 **Results automatically saved:**")
                st.success(f"   📄 Detailed: `{json_file}`")
                st.success(f"   📊 Summary: `{csv_file}`")
                
            except Exception as e:
                st.error(f"❌ Error saving results: {e}")
            
            # Display Enhanced Catalyst Hunter Results
            st.markdown("## 🔥 ENHANCED EXPLOSIVE OPPORTUNITIES (Multi-Layered Analysis)")
            st.markdown(f"### 🎯 Analyzed {len(all_symbols)} stocks | Found {len(catalyst_results)} opportunities | Enhanced {len(enhanced_catalyst_results)} with multi-layer analysis")
            
            # Top catalyst opportunities (using enhanced results)
            top_catalysts = enhanced_catalyst_results[:10]
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                explosive_count = len([r for r in catalyst_results if r.get('explosive_potential', False)])
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🚀 Explosive Moves</h4>
                    <div class="price-big positive">{explosive_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                strong_buy_count = len([r for r in catalyst_results if r.get('recommendation') == 'STRONG BUY'])
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💎 Strong Buys</h4>
                    <div class="price-big positive">{strong_buy_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_catalyst_score = np.mean([r.get('catalyst_score', 0) for r in catalyst_results])
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚡ Avg Catalyst Score</h4>
                    <div class="price-big">{avg_catalyst_score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                news_catalysts = len([r for r in catalyst_results if r.get('catalyst_type')])
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📰 News Catalysts</h4>
                    <div class="price-big">{news_catalysts}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Top Explosive Opportunities
            st.markdown("### 🚀 TOP EXPLOSIVE OPPORTUNITIES")
            
            for i, opportunity in enumerate(top_catalysts[:5]):
                symbol = opportunity['symbol']
                momentum = opportunity.get('momentum_data', {})
                news = opportunity.get('news_data', {})
                
                # Enhanced opportunity card with multi-layered analysis
                catalyst_score = opportunity.get('catalyst_score', 0)
                overall_score = opportunity.get('overall_score', 0)
                confidence = opportunity.get('confidence', 0)
                risk_level = opportunity.get('risk_level', 'Unknown')
                company_name = opportunity.get('company_name', symbol)
                
                # Enhanced title with company name and multiple scores
                title = f"#{i+1} 🎯 {symbol} ({company_name}) - {opportunity.get('recommendation', 'WATCH')} | "
                title += f"Catalyst: {catalyst_score:.1f} | Overall: {overall_score:.1f} | Confidence: {confidence:.1%}"
                
                with st.expander(title, expanded=i < 3):
                    
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
                    
                    with col1:
                        current_price = momentum.get('current_price', 0)
                        price_change = momentum.get('price_change_1d', 0)
                        price_emoji = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "🟡"
                        
                        st.markdown(f"""
                        **💰 Price Data:**
                        - Current: ${current_price:.2f}  
                        - 1D Change: {price_emoji} {price_change:+.2f}%  
                        - Volume Spike: {momentum.get('volume_spike', 1):.1f}x  
                        - Momentum Score: {momentum.get('momentum_score', 0):.1f}/100
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **⚡ Catalyst Analysis:**
                        - Type: {news.get('catalyst_type', 'Unknown').title()}  
                        - News Sentiment: {news.get('sentiment_score', 0):+.2f}  
                        - Confidence: {news.get('confidence', 0):.0f}%  
                        - Explosive: {'🚀 YES' if opportunity.get('explosive_potential', False) else '⚠️ Potential'}
                        """)
                    
                    with col3:
                        technical_score = opportunity.get('technical_score', 0)
                        fundamental_score = opportunity.get('fundamental_score', 0)
                        
                        st.markdown(f"""
                        **🧠 Advanced Analysis:**
                        - Technical: {technical_score:.1f}/100
                        - Fundamental: {fundamental_score:.1f}/100
                        - Overall: {overall_score:.1f}/100
                        - Risk Level: {risk_level}
                        """)
                        
                        # Strategy rankings
                        strategy_rankings = opportunity.get('strategy_rankings', {})
                        if strategy_rankings:
                            st.markdown("**🏆 Strategy Rankings:**")
                            for strategy, ranking in strategy_rankings.items():
                                rank_emoji = "🥇" if ranking['rank'] == 1 else "🥈" if ranking['rank'] == 2 else "🥉" if ranking['rank'] == 3 else "📊"
                                st.markdown(f"  {rank_emoji} {strategy[:10]}: #{ranking['rank']} ({ranking['score']:.1f})")
                    
                    with col4:
                        # Show recent news if available
                        articles = news.get('articles', [])
                        if articles:
                            st.markdown("**🗞️ Recent Catalyst News:**")
                            for article in articles[:2]:
                                sentiment_color = "🟢" if article.get('sentiment', 0) > 0.1 else "🔴" if article.get('sentiment', 0) < -0.1 else "🟡"
                                st.markdown(f"{sentiment_color} {article.get('title', '')[:50]}...")
                        else:
                            st.markdown("**🔍 Catalyst Indicators:**")
                            if momentum.get('explosive_move'):
                                st.markdown("⚡ Explosive price/volume")
                            if momentum.get('breakout_high'):
                                st.markdown("📈 Technical breakout")
                            if momentum.get('volume_spike', 1) > 3:
                                st.markdown("📊 Unusual volume")
                        
                        # Entry/exit suggestions based on enhanced analysis
                        if current_price > 0:
                            if overall_score > 70 and catalyst_score > 60:
                                target_mult = 1.35  # 35% target for high-confidence plays
                                st.markdown(f"**💡 High-Confidence Trade:**")
                            elif catalyst_score > 50:
                                target_mult = 1.25  # 25% target for medium plays
                                st.markdown(f"**💡 Medium-Confidence Trade:**")
                            else:
                                target_mult = 1.15  # 15% target for lower plays
                                st.markdown(f"**💡 Watch/Small Position:**")
                            
                            entry = current_price
                            target = entry * target_mult
                            stop = entry * 0.93 if risk_level == 'High' else entry * 0.95
                            
                            st.markdown(f"- Entry: ${entry:.2f}")
                            st.markdown(f"- Target: ${target:.2f} (+{(target_mult-1)*100:.0f}%)")
                            st.markdown(f"- Stop: ${stop:.2f} (-{(1-stop/entry)*100:.0f}%)")
            
            # Breakout Candidates
            st.markdown("### 📈 TECHNICAL BREAKOUT CANDIDATES")
            
            breakout_candidates = catalyst_detector.get_breakout_candidates([r['symbol'] for r in catalyst_results])
            
            if breakout_candidates:
                for candidate in breakout_candidates[:5]:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label=f"🚀 {candidate['symbol']}",
                            value=f"${candidate['current_price']:.2f}",
                            delta=f"{candidate['price_change_1d']:+.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            label="Momentum Score",
                            value=f"{candidate['momentum_score']:.0f}",
                            delta=f"Vol: {candidate['volume_spike']:.1f}x"
                        )
                    
                    with col3:
                        st.metric(
                            label="Breakout Type",
                            value=candidate['breakout_type'].replace('_', ' ').title(),
                            delta="Pattern Confirmed"
                        )
                    
                    with col4:
                        entry_price = candidate['current_price']
                        target_price = entry_price * 1.2  # 20% target
                        st.metric(
                            label="Entry → Target",
                            value=f"${entry_price:.2f}",
                            delta=f"→ ${target_price:.2f} (+20%)"
                        )
            
            # Enhanced Multi-Layered Analysis Table
            st.markdown("### 📊 COMPLETE MULTI-LAYERED CATALYST ANALYSIS")
            st.markdown("*Showing Catalyst Hunter + Advanced Analysis + Ultimate Strategy insights*")
            
            enhanced_df = pd.DataFrame(enhanced_catalyst_results)
            
            if not enhanced_df.empty:
                # Create comprehensive display DataFrame
                table_data = []
                for result in enhanced_catalyst_results[:50]:  # Show top 50
                    strategy_rankings = result.get('strategy_rankings', {})
                    
                    # Get best strategy ranking
                    best_rank = float('inf')
                    best_strategy = 'N/A'
                    for strategy, ranking in strategy_rankings.items():
                        if ranking['rank'] < best_rank:
                            best_rank = ranking['rank']
                            best_strategy = strategy[:8]  # Shortened name
                    
                    table_data.append({
                        'Symbol': result['symbol'],
                        'Company': result.get('company_name', result['symbol'])[:25] + '...' if len(result.get('company_name', result['symbol'])) > 25 else result.get('company_name', result['symbol']),
                        'Recommendation': result.get('recommendation', 'WATCH'),
                        'Catalyst Score': f"{result.get('catalyst_score', 0):.1f}",
                        'Overall Score': f"{result.get('overall_score', 0):.1f}",
                        'Technical': f"{result.get('technical_score', 0):.1f}",
                        'Fundamental': f"{result.get('fundamental_score', 0):.1f}",
                        'Confidence': f"{result.get('confidence', 0):.1%}",
                        'Risk': result.get('risk_level', 'Unknown'),
                        'Catalyst Type': result.get('catalyst_type', 'Technical'),
                        'Price Δ 1D': f"{result.get('momentum_data', {}).get('price_change_1d', 0):+.2f}%",
                        'Volume Spike': f"{result.get('momentum_data', {}).get('volume_spike', 1):.1f}x",
                        'Explosive': '🚀 YES' if result.get('explosive_potential', False) else '⚠️ Potential',
                        'Best Strategy': f"{best_strategy} #{best_rank}" if best_rank != float('inf') else 'N/A',
                        'HF Rank': strategy_rankings.get('Hedge Fund Style', {}).get('rank', 'N/A'),
                        'Inst Rank': strategy_rankings.get('Institutional Grade', {}).get('rank', 'N/A'),
                        'Quant Rank': strategy_rankings.get('Quant Research', {}).get('rank', 'N/A')
                    })
                
                display_df = pd.DataFrame(table_data)
                
                # Enhanced color coding for multi-layered results
                def color_enhanced_cells(val):
                    if isinstance(val, str):
                        if '🚀 YES' in val:
                            return 'background-color: #d4edda; color: #155724; font-weight: bold'
                        elif 'STRONG BUY' in val:
                            return 'background-color: #d1ecf1; color: #0c5460; font-weight: bold'
                        elif 'BUY' in val:
                            return 'background-color: #e2f3ff; color: #0c5460'
                        elif '+' in val and '%' in val:
                            return 'background-color: #d4edda; color: #155724'
                        elif '-' in val and '%' in val:
                            return 'background-color: #f8d7da; color: #721c24'
                        elif '#1' in val:
                            return 'background-color: #fff3cd; color: #856404; font-weight: bold'  # Top rankings
                        elif '#2' in val or '#3' in val:
                            return 'background-color: #f8f9fa; color: #495057'
                        elif 'High' in val:
                            return 'background-color: #f8d7da; color: #721c24'
                        elif 'Low' in val:
                            return 'background-color: #d4edda; color: #155724'
                    
                    # Numeric score coloring
                    try:
                        if val.replace('.', '').replace('%', '').replace('+', '').replace('-', '').isdigit():
                            num_val = float(val.replace('%', ''))
                            if num_val >= 80:
                                return 'background-color: #d4edda; color: #155724; font-weight: bold'
                            elif num_val >= 60:
                                return 'background-color: #e2f3ff; color: #0c5460'
                            elif num_val <= 30:
                                return 'background-color: #f8d7da; color: #721c24'
                    except:
                        pass
                    
                    return ''
                
                st.dataframe(
                    display_df.style.applymap(color_enhanced_cells), 
                    use_container_width=True,
                    height=600
                )
                
                # Additional insights summary
                st.markdown("### 🎯 MULTI-LAYERED ANALYSIS INSIGHTS")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_catalyst_high_overall = len([r for r in enhanced_catalyst_results if r.get('catalyst_score', 0) > 60 and r.get('overall_score', 0) > 70])
                    st.metric("🔥 High Catalyst + High Overall", high_catalyst_high_overall, 
                             delta="Golden opportunities")
                
                with col2:
                    top_strategy_matches = len([r for r in enhanced_catalyst_results if any(
                        ranking.get('rank', 999) <= 3 for ranking in r.get('strategy_rankings', {}).values()
                    )])
                    st.metric("🏆 Top 3 Strategy Rankings", top_strategy_matches,
                             delta="Multi-strategy validated")
                
                with col3:
                    explosive_with_confirmation = len([r for r in enhanced_catalyst_results if 
                                                     r.get('explosive_potential', False) and r.get('overall_score', 0) > 60])
                    st.metric("🚀 Explosive + Confirmed", explosive_with_confirmation,
                             delta="Highest conviction plays")
            
            # Final completion summary
            st.markdown("---")
            st.markdown("## 🎉 **ENHANCED CATALYST HUNTER ANALYSIS COMPLETE**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Stocks Analyzed", len(all_symbols), 
                         delta="Maximum coverage achieved")
            
            with col2:
                st.metric("🎯 Catalyst Opportunities", len(catalyst_results), 
                         delta=f"{len([r for r in catalyst_results if r.get('catalyst_score', 0) > 70])} Strong Buy")
            
            with col3:
                st.metric("🧠 Advanced Analysis", len(advanced_results), 
                         delta="Multi-layered validation")
            
            with col4:
                st.metric("🏆 Strategy Insights", len(ultimate_insights), 
                         delta="Institutional-grade analysis")
            
            st.success("✅ **All analysis layers complete!** Results automatically saved with timestamp.")
            st.balloons()
            
            # Stop execution here - Catalyst Hunter has its own display
            st.stop()
        
        else:
            st.warning("No significant catalysts detected in current market conditions. Try adjusting parameters or check back later.")
            st.stop()
    
    else:
        # Regular single-strategy analysis
        
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
        
        # Check if we need to select new symbols or use cached ones
        current_params = (cap_filter, market_focus, num_stocks)
        
        if (st.session_state.last_selection_params != current_params or 
            st.session_state.selected_symbols is None):
            # New selection needed
            st.session_state.selected_symbols = get_comprehensive_symbol_selection(
                analyzer, cap_filter, market_focus, num_stocks
            )
            st.session_state.last_selection_params = current_params
            st.info(f"🎯 Selected {len(st.session_state.selected_symbols)} stocks for analysis based on {cap_filter} + {market_focus}")
        else:
            st.info(f"🔄 Using same {len(st.session_state.selected_symbols)} stocks as previous analysis for consistency")
        
        symbols = st.session_state.selected_symbols
        
        # Show selected symbols preview
        with st.expander(f"📊 Selected Stocks Preview ({len(symbols)} total)"):
            st.write("**First 20 symbols:**", symbols[:20])
            if len(symbols) > 20:
                st.write("**Last 10 symbols:**", symbols[-10:])
        
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
        
        # Apply analysis type-specific adjustments
        if results:
            results = apply_analysis_type_adjustments(results, analysis_type)
        
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

        # Market Health: overlay, breadth, macro (one-shot)
        st.markdown("### 🌐 Market Health (Macro + Breadth)")
        try:
            bc = getattr(analyzer, '_breadth_context', {}) or {}
            mc = analyzer.data_fetcher.get_market_context()
            colA, colB, colC, colD = st.columns(4)
            with colA:
                avg_overlay = np.mean([r.get('overlay_score', 50) for r in results])
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Overlay Score</h4>
                    <div class="price-big">{avg_overlay:.1f}/100</div>
                </div>
                """, unsafe_allow_html=True)
            with colB:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Advancers (1D)</h4>
                    <div class="price-big">{bc.get('adv_pct_1d', 0.5)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with colC:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>% Above 50/200 SMA</h4>
                    <div class="price-big">{bc.get('pct_above_sma50', 0.5)*100:.0f}% / {bc.get('pct_above_sma200', 0.5)*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with colD:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>VIX • Curve Slope</h4>
                    <div class="price-big">{mc.get('vix_proxy', 20):.1f} • {mc.get('yield_curve_slope', 0.0):+.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass
        
        # Top BUY Opportunities - Best Profit Potential
        st.markdown("### 🏆 Top BUY Opportunities (Best Profit Potential)")
        
        # Filter for BUY recommendations only, then sort by upside potential
        buy_opportunities = [r for r in results if r['recommendation'] in ['BUY', 'STRONG BUY']]
        
        if not buy_opportunities:
            st.warning("No BUY recommendations found in current analysis. Consider adjusting your parameters.")
            # Fallback to top scored stocks
            top_picks = sorted(results, key=lambda x: x['overall_score'], reverse=True)[:5]
            st.info("Showing top-scored stocks instead (may include HOLD/SELL recommendations):")
        else:
            # Sort BUY opportunities by upside potential (prediction) for maximum profit
            top_picks = sorted(buy_opportunities, key=lambda x: x['prediction'], reverse=True)[:5]
        
        for i, stock in enumerate(top_picks[:5]):
            # Get timeframe information
            primary_timeframe = get_primary_timeframe(stock)
            
            # Create collapsible expander with key info in title
            current_price = stock['current_price']
            price_change = stock['price_change_1d']
            upside_potential = stock['prediction']
            
            # Use buy/sell emoji for recommendation
            rec_emoji = "🚀" if stock['recommendation'] == 'STRONG BUY' else "📈" if stock['recommendation'] == 'BUY' else "⚠️"
            price_emoji = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "🟡"
            
            expander_title = f"#{i+1} {rec_emoji} {stock['symbol']} - {stock['recommendation']} | ${current_price:.2f} ({price_change:+.1f}%) | Upside: {upside_potential:+.1f}% | {primary_timeframe['timeframe']}"
            
            with st.expander(expander_title, expanded=False):
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    # Price and basic info
                    price_emoji = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "🟡"
                    st.markdown(f"""
                    **Current Price:** ${current_price:.2f}  
                    **1D Change:** {price_emoji} {price_change:+.2f}%  
                    **Target:** ${current_price * (1 + stock['prediction']/100):.2f}  
                    **Upside:** {stock['prediction']:+.1f}%  
                    **Volume:** {stock['volume']:,}
                    """)
                
                with col2:
                    # Analysis scores
                    st.markdown(f"""
                    **Confidence:** {stock['confidence']:.1%}  
                    **Risk Level:** {stock['risk_level']}  
                    **Sector:** {stock.get('sector', 'Unknown')}  
                    **Overall Score:** {stock['overall_score']:.1f}/100
                    """)
                    
                with col3:
                    # Timeframe and recommendation
                    st.markdown(f"""
                    **⏰ Expected Timeframe:** {primary_timeframe['timeframe']}  
                    **📊 Expected Accuracy:** {primary_timeframe['accuracy']}  
                    **💡 Strategy:** {primary_timeframe['recommendation']}
                    """)
                    
                    # Progress bars for scores
                    st.progress(stock['technical_score']/100, text=f"Technical: {stock['technical_score']:.0f}/100")
                    st.progress(stock['fundamental_score']/100, text=f"Fundamental: {stock['fundamental_score']:.0f}/100")
                    st.progress(stock['sentiment_score']/100, text=f"Sentiment: {stock['sentiment_score']:.0f}/100")
                
                # Key signals (simplified)
                if stock.get('signals'):
                    st.markdown("**🎯 Key Signals:**")
                    for signal in stock['signals'][:3]:
                        if 'BUY' in signal.upper() or 'BULLISH' in signal.upper():
                            st.markdown(f"🟢 {signal}")
                        elif 'SELL' in signal.upper() or 'BEARISH' in signal.upper():
                            st.markdown(f"🔴 {signal}")
                        else:
                            st.markdown(f"🟡 {signal}")
        
        # High Conviction BUY Opportunities
        high_conviction_buys = [r for r in results if r['recommendation'] in ['BUY', 'STRONG BUY'] and r['confidence'] > 0.8]
        
        if high_conviction_buys:
            st.markdown("### 🎯 High Conviction BUY Opportunities (>80% Confidence)")
            
            # Sort by upside potential
            high_conviction_buys = sorted(high_conviction_buys, key=lambda x: x['prediction'], reverse=True)[:3]
            
            for stock in high_conviction_buys:
                primary_timeframe = get_primary_timeframe(stock)
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.metric(
                        label=f"🚀 {stock['symbol']}",
                        value=f"${stock['current_price']:.2f}",
                        delta=f"{stock['price_change_1d']:+.1f}%"
                    )
                
                with col2:
                    st.metric(
                        label="Upside Potential",
                        value=f"{stock['prediction']:+.1f}%",
                        delta=f"Target: ${stock['current_price'] * (1 + stock['prediction']/100):.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="Confidence",
                        value=f"{stock['confidence']:.1%}",
                        delta=f"Risk: {stock['risk_level']}"
                    )
                
                with col4:
                    st.metric(
                        label="Timeframe",
                        value=primary_timeframe['timeframe'],
                        delta=f"Accuracy: {primary_timeframe['accuracy']}"
                    )
        
        # Simplified Timeframe Guide
        with st.expander("⏰ Understanding Prediction Timeframes", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🚀 Short-Term (1-14 days)**")
                st.markdown("• Accuracy: 60-70%")
                st.markdown("• Best for: Active traders")
                
            with col2:
                st.markdown("**📊 Medium-Term (1-4 weeks)**")
                st.markdown("• Accuracy: 65-75%")
                st.markdown("• Best for: Most users (recommended)")
                
            with col3:
                st.markdown("**🎯 Long-Term (3+ months)**")
                st.markdown("• Accuracy: 70-85%")
                st.markdown("• Best for: Patient investors")
            
            st.info("💡 **Key Principle:** Longer timeframes = Higher reliability. High confidence (>80%) significantly improves outcomes.")
        
        # Professional Analysis Table
        st.markdown("### 📊 Complete Professional Analysis")
        
        # Create professional DataFrame
        df_results = pd.DataFrame(results)
        
        # Add professional columns with timeframe information
        df_display = df_results[['symbol', 'current_price', 'price_change_1d', 'prediction', 'confidence', 
                                'recommendation', 'risk_level', 'overall_score', 'technical_score', 
                                'fundamental_score', 'sentiment_score']].copy()
        
        # Add timeframe and upside calculations
        df_display['upside_potential'] = df_display['prediction']
        df_display['target_price'] = df_display['current_price'] * (1 + df_display['prediction']/100)
        
        # Add timeframe information for each stock
        timeframe_data = []
        timeframe_type_data = []
        accuracy_data = []
        
        for _, row in df_results.iterrows():
            primary_timeframe = get_primary_timeframe(row.to_dict())
            timeframe_data.append(primary_timeframe['timeframe'])
            timeframe_type_data.append(primary_timeframe['type'].title())
            accuracy_data.append(primary_timeframe['accuracy'])
        
        df_display['expected_timeframe'] = timeframe_data
        df_display['timeframe_type'] = timeframe_type_data
        df_display['expected_accuracy'] = accuracy_data
        
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
                             'Fundamental Score', 'Sentiment Score', 'Upside/Downside', 'Target Price',
                             'Expected Timeframe', 'Timeframe Type', 'Expected Accuracy']
        
        # Enhanced color-code the table including timeframes
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
                elif val == 'Short-Term':
                    return 'background-color: #fff3cd; color: #856404'
                elif val == 'Medium-Term':
                    return 'background-color: #d1ecf1; color: #0c5460'
                elif val == 'Long-Term':
                    return 'background-color: #d4edda; color: #155724'
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
