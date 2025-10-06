#!/usr/bin/env python3
"""
Quick Catalyst Scanner - Immediate Explosive Opportunity Detection
Fast 5-minute scan for the most promising catalyst opportunities
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from catalyst_detector import CatalystDetector
import pandas as pd
import time
from datetime import datetime

def quick_scan():
    """Perform a quick 5-minute catalyst scan on high-potential symbols"""
    
    print("🎯 QUICK CATALYST SCAN - EXPLOSIVE OPPORTUNITY HUNTER")
    print("=" * 60)
    print(f"📅 Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚡ Target: Next AMD (+27%) or RGC (+10000%) opportunity")
    print()
    
    # Initialize detector
    detector = CatalystDetector()
    
    # High-potential symbol list (focused on explosive movers)
    explosive_candidates = [
        # AI/Tech leaders (partnership potential)
        'AMD', 'NVDA', 'META', 'MSFT', 'GOOGL',
        
        # Growth/momentum stocks (breakout potential) 
        'PLTR', 'CRWD', 'SNOW', 'NET', 'DDOG',
        
        # High-volatility plays (explosive potential)
        'TSLA', 'RIVN', 'LCID', 'SOFI', 'UPST',
        
        # Small cap rockets (RGC-style potential)
        'BBAI', 'SMCI', 'RGTI', 'SOUN', 'AVAV'
    ]
    
    print(f"🔍 Scanning {len(explosive_candidates)} high-potential symbols...")
    print("⏱️ Estimated time: 3-5 minutes")
    print()
    
    # Quick parallel scan (limit workers for speed)
    start_time = time.time()
    
    try:
        results = detector.scan_for_catalysts(explosive_candidates, max_workers=8)
        
        scan_time = time.time() - start_time
        
        if results:
            # Sort by catalyst score (highest first)
            results = sorted(results, key=lambda x: x.get('catalyst_score', 0), reverse=True)
            
            print("🔥 EXPLOSIVE OPPORTUNITIES DETECTED")
            print("=" * 80)
            print(f"⏱️ Scan completed in {scan_time:.1f} seconds")
            print(f"🎯 Found {len(results)} opportunities")
            print()
            
            # Top 5 explosive opportunities
            print("🚀 TOP 5 EXPLOSIVE OPPORTUNITIES:")
            print("-" * 80)
            print(f"{'Symbol':<8} {'Score':<8} {'Rec':<12} {'Catalyst':<12} {'Price Δ':<10} {'Vol Spike':<10}")
            print("-" * 80)
            
            top_opportunities = []
            
            for i, result in enumerate(results[:5]):
                symbol = result['symbol']
                score = result.get('catalyst_score', 0)
                recommendation = result.get('recommendation', 'WATCH')
                catalyst_type = result.get('catalyst_type', 'Technical')
                explosive = result.get('explosive_potential', False)
                
                momentum = result.get('momentum_data', {})
                price_change = momentum.get('price_change_1d', 0)
                volume_spike = momentum.get('volume_spike', 1)
                
                # Recommendation emoji
                if recommendation == 'STRONG BUY':
                    rec_emoji = '🚀'
                elif recommendation == 'BUY':
                    rec_emoji = '📈'
                else:
                    rec_emoji = '⚪'
                
                print(f"{symbol:<8} {score:6.1f} {rec_emoji} {recommendation:<10} {catalyst_type:<12} {price_change:+6.2f}% {volume_spike:7.1f}x")
                
                # Collect for detailed analysis
                if score >= 50 or explosive:
                    top_opportunities.append(result)
            
            # Detailed analysis of top opportunities
            if top_opportunities:
                print()
                print("💎 DETAILED CATALYST ANALYSIS:")
                print("=" * 60)
                
                for opportunity in top_opportunities[:3]:
                    symbol = opportunity['symbol']
                    momentum = opportunity.get('momentum_data', {})
                    news = opportunity.get('news_data', {})
                    
                    print(f"\n🎯 {symbol} - {opportunity.get('recommendation', 'WATCH')}")
                    print(f"   💰 Current Price: ${momentum.get('current_price', 0):.2f}")
                    print(f"   📈 Price Change: {momentum.get('price_change_1d', 0):+.2f}%")
                    print(f"   📊 Volume Spike: {momentum.get('volume_spike', 1):.1f}x normal")
                    print(f"   ⚡ Catalyst Score: {opportunity.get('catalyst_score', 0):.1f}/100")
                    print(f"   🗞️ Catalyst Type: {news.get('catalyst_type', 'Unknown').title()}")
                    print(f"   🎯 Explosive Potential: {'🚀 YES' if opportunity.get('explosive_potential', False) else '⚠️ Moderate'}")
                    
                    # Show key news if available
                    articles = news.get('articles', [])
                    if articles and len(articles) > 0:
                        print(f"   📰 Key News: {articles[0].get('title', '')[:50]}...")
                    
                    # Entry/exit suggestion
                    entry_price = momentum.get('current_price', 0)
                    if entry_price > 0:
                        target_price = entry_price * 1.25  # 25% target
                        stop_price = entry_price * 0.95   # 5% stop
                        print(f"   💡 Entry: ${entry_price:.2f} | Target: ${target_price:.2f} (+25%) | Stop: ${stop_price:.2f} (-5%)")
            
            # Summary statistics
            print()
            print("📊 SCAN SUMMARY:")
            print("-" * 40)
            strong_buys = len([r for r in results if r.get('recommendation') == 'STRONG BUY'])
            buys = len([r for r in results if r.get('recommendation') == 'BUY'])
            explosive_moves = len([r for r in results if r.get('explosive_potential', False)])
            news_catalysts = len([r for r in results if r.get('catalyst_type') and r.get('catalyst_type') != 'Technical'])
            avg_score = sum(r.get('catalyst_score', 0) for r in results) / len(results) if results else 0
            
            print(f"🚀 Strong Buy Signals: {strong_buys}")
            print(f"📈 Buy Signals: {buys}")  
            print(f"💥 Explosive Potential: {explosive_moves}")
            print(f"📰 News-Driven Catalysts: {news_catalysts}")
            print(f"⚡ Average Catalyst Score: {avg_score:.1f}")
            
            # Action recommendations
            print()
            print("🎯 IMMEDIATE ACTIONS:")
            print("-" * 30)
            
            if strong_buys > 0:
                print("🚀 URGENT: Strong Buy signals detected - Review immediately!")
            elif explosive_moves > 0:
                print("💥 ALERT: Explosive moves detected - Monitor closely!")
            elif buys > 0:
                print("📈 OPPORTUNITY: Buy signals available - Consider positions")
            else:
                print("⚪ HOLD: No immediate explosive opportunities detected")
            
            if avg_score > 60:
                print("🔥 Market showing strong catalyst activity!")
            elif avg_score > 45:
                print("📊 Moderate catalyst activity detected")
            else:
                print("😴 Low catalyst activity - Wait for better setups")
            
        else:
            print("⚠️ No significant catalysts detected in current market conditions")
            print("💡 Try again during market hours or after news events")
    
    except KeyboardInterrupt:
        print("\n⚠️ Scan interrupted by user")
        
    except Exception as e:
        print(f"❌ Error during catalyst scan: {e}")
        print("💡 Try running the full trading app for more detailed analysis")
    
    print()
    print("=" * 60)
    print("💰 Quick Catalyst Scan Complete!")
    print("🚀 Ready to catch the next explosive opportunity!")
    print()
    print("💡 For full analysis, run: streamlit run professional_trading_app.py")
    print("   Then select 'Catalyst Hunter' mode for comprehensive scanning")

if __name__ == "__main__":
    quick_scan()
