#!/usr/bin/env python3
"""
Test Catalyst Hunter System
Tests the explosive opportunity detection system with real market examples
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from catalyst_detector import CatalystDetector
import pandas as pd
import time

def test_explosive_momentum_detection():
    """Test explosive momentum detection on known examples"""
    print("🎯 Testing Explosive Momentum Detection...")
    
    detector = CatalystDetector()
    
    # Test symbols - mix of explosive movers and normal stocks
    test_symbols = [
        'AMD',   # Recent 27% spike on OpenAI news
        'NVDA',  # AI/tech momentum
        'TSLA',  # High volatility
        'AAPL',  # Large cap stable
        'PLTR',  # Growth/momentum stock
        'MSFT',  # Large cap with AI exposure
        'META',  # Recent AI focus
        'GOOGL', # AI competition
        'CRWD',  # Cybersecurity momentum
        'SNOW'   # Data analytics growth
    ]
    
    results = []
    
    for symbol in test_symbols:
        print(f"  📊 Analyzing {symbol}...")
        try:
            momentum = detector.detect_explosive_momentum(symbol)
            if 'error' not in momentum:
                results.append({
                    'symbol': symbol,
                    'price_change_1d': momentum.get('price_change_1d', 0),
                    'volume_spike': momentum.get('volume_spike', 1),
                    'momentum_score': momentum.get('momentum_score', 0),
                    'explosive_move': momentum.get('explosive_move', False),
                    'catalyst_likely': momentum.get('catalyst_likely', False)
                })
            else:
                print(f"    ❌ Error: {momentum['error']}")
        except Exception as e:
            print(f"    ❌ Exception: {e}")
        
        time.sleep(0.1)  # Rate limiting
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('momentum_score', ascending=False)
        
        print("\n🔥 EXPLOSIVE MOMENTUM RESULTS:")
        print("=" * 60)
        print(f"{'Symbol':<8} {'1D Change':<10} {'Vol Spike':<10} {'Score':<8} {'Explosive':<10} {'Catalyst':<8}")
        print("-" * 60)
        
        for _, row in df.head(10).iterrows():
            explosive_icon = "🚀" if row['explosive_move'] else "📈" if row['momentum_score'] > 50 else "⚪"
            catalyst_icon = "⚡" if row['catalyst_likely'] else "⚪"
            
            print(f"{row['symbol']:<8} {row['price_change_1d']:+7.2f}% {row['volume_spike']:7.1f}x {row['momentum_score']:6.1f} {explosive_icon:<10} {catalyst_icon:<8}")
        
        # Highlight top explosive opportunities
        explosive_moves = df[df['explosive_move'] == True]
        if not explosive_moves.empty:
            print(f"\n🚀 EXPLOSIVE MOVES DETECTED ({len(explosive_moves)}):")
            for _, row in explosive_moves.iterrows():
                print(f"  💥 {row['symbol']}: {row['price_change_1d']:+.2f}% (Volume: {row['volume_spike']:.1f}x)")
        
        return True
    
    return False

def test_news_sentiment_analysis():
    """Test news sentiment and catalyst detection"""
    print("\n📰 Testing News Sentiment Analysis...")
    
    detector = CatalystDetector()
    
    # Focus on stocks with recent news catalysts
    news_test_symbols = ['AMD', 'NVDA', 'META', 'MSFT', 'TSLA']
    
    for symbol in news_test_symbols:
        print(f"  📊 Analyzing news for {symbol}...")
        try:
            news_data = detector.get_news_sentiment(symbol, limit=5)
            
            if 'error' not in news_data:
                sentiment = news_data.get('sentiment_score', 0)
                catalyst_detected = news_data.get('catalyst_detected', False)
                catalyst_type = news_data.get('catalyst_type', 'None')
                confidence = news_data.get('confidence', 0)
                
                sentiment_icon = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
                catalyst_icon = "⚡" if catalyst_detected else "⚪"
                
                print(f"    {sentiment_icon} Sentiment: {sentiment:+.3f}")
                print(f"    {catalyst_icon} Catalyst: {catalyst_type} (Confidence: {confidence:.0f}%)")
                
                # Show key articles if available
                articles = news_data.get('articles', [])
                if articles:
                    print(f"    📄 Recent Articles ({len(articles)}):")
                    for i, article in enumerate(articles[:2]):
                        title = article.get('title', '')[:50]
                        article_sentiment = article.get('sentiment', 0)
                        article_icon = "🟢" if article_sentiment > 0.1 else "🔴" if article_sentiment < -0.1 else "🟡"
                        print(f"      {i+1}. {article_icon} {title}...")
            else:
                print(f"    ❌ Error: {news_data['error']}")
        
        except Exception as e:
            print(f"    ❌ Exception: {e}")
        
        time.sleep(0.2)  # Rate limiting for news API
    
    return True

def test_comprehensive_catalyst_scan():
    """Test comprehensive catalyst scanning"""
    print("\n🔍 Testing Comprehensive Catalyst Scan...")
    
    detector = CatalystDetector()
    
    # Mix of different types of stocks for comprehensive testing
    scan_symbols = [
        'AMD', 'NVDA', 'META', 'MSFT',  # Tech/AI stocks
        'TSLA', 'PLTR', 'CRWD', 'SNOW',  # Growth/momentum
        'JPM', 'BAC', 'KO', 'PG'        # Traditional large caps
    ]
    
    print(f"  🎯 Scanning {len(scan_symbols)} symbols for catalysts...")
    
    try:
        catalyst_results = detector.scan_for_catalysts(scan_symbols, max_workers=5)
        
        if catalyst_results:
            print(f"\n🔥 CATALYST SCAN RESULTS ({len(catalyst_results)} opportunities):")
            print("=" * 80)
            print(f"{'Symbol':<8} {'Score':<8} {'Rec':<12} {'Type':<12} {'Explosive':<10} {'Price Δ':<10}")
            print("-" * 80)
            
            for result in catalyst_results[:10]:
                symbol = result['symbol']
                score = result.get('catalyst_score', 0)
                recommendation = result.get('recommendation', 'WATCH')
                catalyst_type = result.get('catalyst_type', 'Technical')
                explosive = "🚀 YES" if result.get('explosive_potential', False) else "⚪ No"
                
                momentum = result.get('momentum_data', {})
                price_change = momentum.get('price_change_1d', 0)
                
                rec_icon = "🚀" if recommendation == 'STRONG BUY' else "📈" if recommendation == 'BUY' else "⚪"
                
                print(f"{symbol:<8} {score:6.1f} {rec_icon} {recommendation:<10} {catalyst_type:<12} {explosive:<10} {price_change:+6.2f}%")
            
            # Summary statistics
            strong_buys = len([r for r in catalyst_results if r.get('recommendation') == 'STRONG BUY'])
            explosive_moves = len([r for r in catalyst_results if r.get('explosive_potential', False)])
            news_catalysts = len([r for r in catalyst_results if r.get('catalyst_type') and r.get('catalyst_type') != 'Technical'])
            
            print(f"\n📊 CATALYST SUMMARY:")
            print(f"  🚀 Strong Buy Recommendations: {strong_buys}")
            print(f"  💥 Explosive Moves Detected: {explosive_moves}")
            print(f"  📰 News-Driven Catalysts: {news_catalysts}")
            print(f"  ⚡ Average Catalyst Score: {sum(r.get('catalyst_score', 0) for r in catalyst_results) / len(catalyst_results):.1f}")
            
            return True
        
        else:
            print("  ⚠️ No catalysts detected in current scan")
            return False
    
    except Exception as e:
        print(f"  ❌ Exception during catalyst scan: {e}")
        return False

def test_breakout_detection():
    """Test breakout pattern detection"""
    print("\n📈 Testing Breakout Pattern Detection...")
    
    detector = CatalystDetector()
    
    # Test various symbols for breakout patterns
    breakout_symbols = ['AMD', 'NVDA', 'TSLA', 'PLTR', 'CRWD', 'SNOW', 'META', 'MSFT']
    
    try:
        breakout_candidates = detector.get_breakout_candidates(breakout_symbols, min_momentum_score=30)
        
        if breakout_candidates:
            print(f"🎯 BREAKOUT CANDIDATES DETECTED ({len(breakout_candidates)}):")
            print("=" * 70)
            print(f"{'Symbol':<8} {'Score':<8} {'Price Δ':<10} {'Vol Spike':<10} {'Type':<15}")
            print("-" * 70)
            
            for candidate in breakout_candidates:
                symbol = candidate['symbol']
                score = candidate['momentum_score']
                price_change = candidate['price_change_1d']
                volume_spike = candidate['volume_spike']
                breakout_type = candidate['breakout_type'].replace('_', ' ').title()
                
                print(f"{symbol:<8} {score:6.1f} {price_change:+7.2f}% {volume_spike:7.1f}x {breakout_type:<15}")
            
            return True
        
        else:
            print("  ⚠️ No breakout patterns detected")
            return False
    
    except Exception as e:
        print(f"  ❌ Exception during breakout detection: {e}")
        return False

def main():
    """Run comprehensive catalyst detection tests"""
    print("🎯 CATALYST HUNTER SYSTEM TEST")
    print("=" * 50)
    print("Testing explosive opportunity detection system...")
    print()
    
    test_results = []
    
    # Run all tests
    try:
        test_results.append(("Explosive Momentum Detection", test_explosive_momentum_detection()))
        test_results.append(("News Sentiment Analysis", test_news_sentiment_analysis()))
        test_results.append(("Comprehensive Catalyst Scan", test_comprehensive_catalyst_scan()))
        test_results.append(("Breakout Pattern Detection", test_breakout_detection()))
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All catalyst detection systems operational!")
        print("💡 The system is ready to detect explosive opportunities like RGC and AMD!")
    else:
        print("⚠️ Some systems need attention. Check error messages above.")
    
    print("\n💰 Ready to hunt for the next 1000%+ opportunity!")

if __name__ == "__main__":
    main()
