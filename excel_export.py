#!/usr/bin/env python3
"""
Excel Export Functionality for SmartTrade AI
Professional reporting and portfolio management
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def export_analysis_to_excel(results, analysis_params=None, filename=None):
    """Export analysis results to Excel with multiple sheets"""
    
    if not results:
        return None, "No results to export"
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SmartTrade_Analysis_{timestamp}.xlsx"
    
    try:
        # Create Excel writer object
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Sheet 1: Summary Dashboard
            create_summary_sheet(results, writer, analysis_params)
            
            # Sheet 2: Strong Buy Recommendations
            create_recommendations_sheet(results, writer, 'STRONG BUY')
            
            # Sheet 3: All Buy Signals
            create_recommendations_sheet(results, writer, ['STRONG BUY', 'BUY', 'WEAK BUY'], 'All_Buy_Signals')
            
            # Sheet 4: Detailed Analysis
            create_detailed_analysis_sheet(results, writer)
            
            # Sheet 5: Technical Indicators
            create_technical_sheet(results, writer)
            
            # Sheet 6: Risk Analysis
            create_risk_analysis_sheet(results, writer)
            
            # Sheet 7: Sector Analysis
            create_sector_analysis_sheet(results, writer)
            
            # Sheet 8: Performance Metrics
            create_performance_sheet(results, writer)
        
        return filename, f"Successfully exported {len(results)} stocks to {filename}"
        
    except Exception as e:
        return None, f"Export failed: {str(e)}"

def create_summary_sheet(results, writer, analysis_params):
    """Create summary dashboard sheet"""
    
    # Analysis summary
    total_stocks = len(results)
    strong_buy = len([r for r in results if r.get('recommendation') == 'STRONG BUY'])
    buy = len([r for r in results if r.get('recommendation') == 'BUY'])
    weak_buy = len([r for r in results if r.get('recommendation') == 'WEAK BUY'])
    hold = len([r for r in results if r.get('recommendation') == 'HOLD'])
    sell_signals = len([r for r in results if r.get('recommendation', '').endswith('SELL')])
    
    # Create summary data
    summary_data = {
        'Metric': [
            'Analysis Date',
            'Total Stocks Analyzed',
            'Strong Buy Recommendations',
            'Buy Recommendations', 
            'Weak Buy Recommendations',
            'Hold Recommendations',
            'Sell Signals',
            'Success Rate (%)',
            'Average Score',
            'Top Performer',
            'Analysis Parameters'
        ],
        'Value': [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_stocks,
            strong_buy,
            buy,
            weak_buy,
            hold,
            sell_signals,
            f"{(strong_buy + buy + weak_buy) / total_stocks * 100:.1f}%" if total_stocks > 0 else "0%",
            f"{np.mean([r.get('overall_score', 0) for r in results]):.1f}",
            max(results, key=lambda x: x.get('overall_score', 0)).get('symbol', 'N/A') if results else 'N/A',
            str(analysis_params) if analysis_params else 'Standard Analysis'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

def create_recommendations_sheet(results, writer, recommendation_types, sheet_name=None):
    """Create recommendations sheet"""
    
    if isinstance(recommendation_types, str):
        recommendation_types = [recommendation_types]
        sheet_name = sheet_name or recommendation_types[0].replace(' ', '_')
    else:
        sheet_name = sheet_name or 'Recommendations'
    
    # Filter results
    filtered_results = [r for r in results if r.get('recommendation') in recommendation_types]
    
    if not filtered_results:
        # Create empty sheet with message
        empty_df = pd.DataFrame({'Message': ['No recommendations found for the selected criteria']})
        empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
        return
    
    # Create recommendations data
    recommendations_data = []
    for result in filtered_results:
        recommendations_data.append({
            'Symbol': result.get('symbol', ''),
            'Company': result.get('company_name', ''),
            'Recommendation': result.get('recommendation', ''),
            'Current Price': f"${result.get('current_price', 0):.2f}",
            'Predicted Return': f"{result.get('prediction', 0) * 100:.1f}%",
            'Overall Score': result.get('overall_score', 0),
            'Technical Score': result.get('technical_score', 0),
            'Fundamental Score': result.get('fundamental_score', 0),
            'Risk Level': result.get('risk_level', ''),
            'Sector': result.get('sector', ''),
            'Market Cap': result.get('market_cap', 0),
            'Volume': result.get('volume', 0),
            'P/E Ratio': result.get('pe_ratio', 0),
            'Confidence': f"{result.get('confidence', 0) * 100:.1f}%"
        })
    
    recommendations_df = pd.DataFrame(recommendations_data)
    recommendations_df = recommendations_df.sort_values('Overall Score', ascending=False)
    recommendations_df.to_excel(writer, sheet_name=sheet_name, index=False)

def create_detailed_analysis_sheet(results, writer):
    """Create detailed analysis sheet with all metrics"""
    
    detailed_data = []
    for result in results:
        detailed_data.append({
            'Symbol': result.get('symbol', ''),
            'Company': result.get('company_name', ''),
            'Recommendation': result.get('recommendation', ''),
            'Current Price': result.get('current_price', 0),
            'Predicted Return %': result.get('prediction', 0) * 100,
            'Confidence %': result.get('confidence', 0) * 100,
            'Overall Score': result.get('overall_score', 0),
            'Technical Score': result.get('technical_score', 0),
            'Fundamental Score': result.get('fundamental_score', 0),
            'Sentiment Score': result.get('sentiment_score', 0),
            'Risk Level': result.get('risk_level', ''),
            'Sector': result.get('sector', ''),
            'Market Cap': result.get('market_cap', 0),
            'Volume': result.get('volume', 0),
            'P/E Ratio': result.get('pe_ratio', 0),
            'P/B Ratio': result.get('pb_ratio', 0),
            'ROE %': result.get('roe', 0),
            'Debt/Equity': result.get('debt_equity', 0),
            'Revenue Growth %': result.get('revenue_growth', 0),
            'Earnings Growth %': result.get('earnings_growth', 0),
            'RSI': result.get('rsi', 0),
            'MACD Signal': result.get('macd_signal', ''),
            '50-Day MA': result.get('ma_50', 0),
            '200-Day MA': result.get('ma_200', 0),
            'Bollinger Position': result.get('bollinger_position', ''),
            'Support Level': result.get('support', 0),
            'Resistance Level': result.get('resistance', 0)
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df = detailed_df.sort_values('Overall Score', ascending=False)
    detailed_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)

def create_technical_sheet(results, writer):
    """Create technical indicators sheet"""
    
    technical_data = []
    for result in results:
        signals = result.get('signals', {})
        technical_data.append({
            'Symbol': result.get('symbol', ''),
            'RSI': result.get('rsi', 0),
            'RSI Signal': signals.get('rsi_signal', ''),
            'MACD': result.get('macd', 0),
            'MACD Signal': signals.get('macd_signal', ''),
            'Stochastic %K': result.get('stoch_k', 0),
            'Stochastic %D': result.get('stoch_d', 0),
            'Williams %R': result.get('williams_r', 0),
            'CCI': result.get('cci', 0),
            'ADX': result.get('adx', 0),
            'Bollinger Upper': result.get('bb_upper', 0),
            'Bollinger Lower': result.get('bb_lower', 0),
            'Bollinger Position': result.get('bollinger_position', ''),
            'Volume SMA': result.get('volume_sma', 0),
            'Price vs MA50': f"{((result.get('current_price', 0) / result.get('ma_50', 1) - 1) * 100):.1f}%",
            'Price vs MA200': f"{((result.get('current_price', 0) / result.get('ma_200', 1) - 1) * 100):.1f}%",
            'Support': result.get('support', 0),
            'Resistance': result.get('resistance', 0),
            'Technical Score': result.get('technical_score', 0)
        })
    
    technical_df = pd.DataFrame(technical_data)
    technical_df = technical_df.sort_values('Technical Score', ascending=False)
    technical_df.to_excel(writer, sheet_name='Technical_Indicators', index=False)

def create_risk_analysis_sheet(results, writer):
    """Create risk analysis sheet"""
    
    risk_data = []
    for result in results:
        risk_data.append({
            'Symbol': result.get('symbol', ''),
            'Risk Level': result.get('risk_level', ''),
            'Beta': result.get('beta', 0),
            'Volatility': result.get('volatility', 0),
            'Sharpe Ratio': result.get('sharpe_ratio', 0),
            'Max Drawdown': result.get('max_drawdown', 0),
            'VaR (95%)': result.get('var_95', 0),
            'Debt/Equity': result.get('debt_equity', 0),
            'Current Ratio': result.get('current_ratio', 0),
            'Quick Ratio': result.get('quick_ratio', 0),
            'Interest Coverage': result.get('interest_coverage', 0),
            'Recommendation': result.get('recommendation', ''),
            'Overall Score': result.get('overall_score', 0)
        })
    
    risk_df = pd.DataFrame(risk_data)
    risk_df = risk_df.sort_values('Risk Level')
    risk_df.to_excel(writer, sheet_name='Risk_Analysis', index=False)

def create_sector_analysis_sheet(results, writer):
    """Create sector analysis sheet"""
    
    # Group by sector
    sector_summary = {}
    for result in results:
        sector = result.get('sector', 'Unknown')
        if sector not in sector_summary:
            sector_summary[sector] = {
                'count': 0,
                'strong_buy': 0,
                'buy': 0,
                'avg_score': 0,
                'total_score': 0,
                'stocks': []
            }
        
        sector_summary[sector]['count'] += 1
        sector_summary[sector]['total_score'] += result.get('overall_score', 0)
        sector_summary[sector]['stocks'].append(result.get('symbol', ''))
        
        recommendation = result.get('recommendation', '')
        if recommendation == 'STRONG BUY':
            sector_summary[sector]['strong_buy'] += 1
        elif recommendation == 'BUY':
            sector_summary[sector]['buy'] += 1
    
    # Calculate averages
    sector_data = []
    for sector, data in sector_summary.items():
        sector_data.append({
            'Sector': sector,
            'Total Stocks': data['count'],
            'Strong Buy': data['strong_buy'],
            'Buy': data['buy'],
            'Buy Rate %': f"{(data['strong_buy'] + data['buy']) / data['count'] * 100:.1f}%",
            'Avg Score': data['total_score'] / data['count'] if data['count'] > 0 else 0,
            'Top Stocks': ', '.join(data['stocks'][:5])  # Top 5 stocks
        })
    
    sector_df = pd.DataFrame(sector_data)
    sector_df = sector_df.sort_values('Avg Score', ascending=False)
    sector_df.to_excel(writer, sheet_name='Sector_Analysis', index=False)

def create_performance_sheet(results, writer):
    """Create performance metrics sheet"""
    
    # Calculate various performance metrics
    scores = [r.get('overall_score', 0) for r in results]
    predictions = [r.get('prediction', 0) for r in results]
    
    performance_data = {
        'Metric': [
            'Total Stocks Analyzed',
            'Average Overall Score',
            'Median Overall Score',
            'Standard Deviation',
            'Top 10% Threshold',
            'Bottom 10% Threshold',
            'Average Predicted Return',
            'Max Predicted Return',
            'Min Predicted Return',
            'Strong Buy Count',
            'Buy Count',
            'Hold Count',
            'Sell Count',
            'High Risk Count',
            'Medium Risk Count',
            'Low Risk Count'
        ],
        'Value': [
            len(results),
            f"{np.mean(scores):.2f}",
            f"{np.median(scores):.2f}",
            f"{np.std(scores):.2f}",
            f"{np.percentile(scores, 90):.2f}",
            f"{np.percentile(scores, 10):.2f}",
            f"{np.mean(predictions) * 100:.2f}%",
            f"{np.max(predictions) * 100:.2f}%",
            f"{np.min(predictions) * 100:.2f}%",
            len([r for r in results if r.get('recommendation') == 'STRONG BUY']),
            len([r for r in results if r.get('recommendation') == 'BUY']),
            len([r for r in results if r.get('recommendation') == 'HOLD']),
            len([r for r in results if 'SELL' in r.get('recommendation', '')]),
            len([r for r in results if r.get('risk_level') == 'High']),
            len([r for r in results if r.get('risk_level') == 'Medium']),
            len([r for r in results if r.get('risk_level') == 'Low'])
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)

def add_excel_export_to_streamlit(results):
    """Add Excel export functionality to Streamlit app"""
    
    import streamlit as st
    
    if results and len(results) > 0:
        st.markdown("---")
        st.subheader("📊 Export Results")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**Export your analysis results to Excel for further analysis and portfolio management.**")
        
        with col2:
            export_filename = st.text_input("Filename (optional)", placeholder="SmartTrade_Analysis")
        
        with col3:
            if st.button("📥 Export to Excel", type="primary"):
                # Add timestamp if no filename provided
                if not export_filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_filename = f"SmartTrade_Analysis_{timestamp}"
                
                # Ensure .xlsx extension
                if not export_filename.endswith('.xlsx'):
                    export_filename += '.xlsx'
                
                # Export to Excel
                filename, message = export_analysis_to_excel(results, filename=export_filename)
                
                if filename:
                    st.success(f"✅ {message}")
                    
                    # Provide download link
                    if os.path.exists(filename):
                        with open(filename, 'rb') as f:
                            st.download_button(
                                label="📥 Download Excel File",
                                data=f.read(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                else:
                    st.error(f"❌ {message}")
        
        # Show export details
        with st.expander("📋 Export Details"):
            st.markdown("""
            **Excel file includes 8 sheets:**
            - **Summary**: Overview and key metrics
            - **Strong Buy**: Top recommendations only
            - **All Buy Signals**: All buy recommendations
            - **Detailed Analysis**: Complete analysis data
            - **Technical Indicators**: All technical metrics
            - **Risk Analysis**: Risk assessment data
            - **Sector Analysis**: Sector breakdown
            - **Performance Metrics**: Statistical analysis
            """)

if __name__ == "__main__":
    # Test the export functionality
    sample_results = [
        {
            'symbol': 'AAPL',
            'company_name': 'Apple Inc.',
            'recommendation': 'STRONG BUY',
            'current_price': 175.50,
            'prediction': 0.15,
            'confidence': 0.85,
            'overall_score': 85,
            'technical_score': 80,
            'fundamental_score': 90,
            'risk_level': 'Medium',
            'sector': 'Technology'
        }
    ]
    
    filename, message = export_analysis_to_excel(sample_results)
    print(f"Test export: {message}")
