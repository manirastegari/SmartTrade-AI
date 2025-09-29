#!/usr/bin/env python3
"""
Performance Test Script for AI Trading Application
Tests the optimized vs original performance
"""

import time
import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_analyzer import AdvancedTradingAnalyzer

def performance_test():
    """Run performance test with different stock counts"""
    
    print("🚀 AI Trading Application - Performance Test")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test configurations
    test_configs = [
        {"stocks": 25, "name": "Small Test"},
        {"stocks": 50, "name": "Medium Test"}, 
        {"stocks": 100, "name": "Large Test"},
        {"stocks": 200, "name": "Production Test"}
    ]
    
    results = []
    
    for config in test_configs:
        print(f"🧪 Running {config['name']} ({config['stocks']} stocks)")
        print("-" * 40)
        
        try:
            # Initialize analyzer
            analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
            
            # Run analysis
            start_time = time.time()
            analysis_results = analyzer.run_advanced_analysis(max_stocks=config['stocks'])
            end_time = time.time()
            
            # Calculate metrics
            duration = end_time - start_time
            successful = len(analysis_results)
            rate = successful / duration if duration > 0 else 0
            
            result = {
                'config': config['name'],
                'stocks_requested': config['stocks'],
                'stocks_analyzed': successful,
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'rate_per_second': rate,
                'rate_per_minute': rate * 60
            }
            
            results.append(result)
            
            print(f"✅ Results:")
            print(f"   📊 Analyzed: {successful}/{config['stocks']} stocks")
            print(f"   ⏱️  Duration: {duration/60:.2f} minutes ({duration:.1f} seconds)")
            print(f"   🚀 Rate: {rate:.2f} stocks/second ({rate*60:.1f} stocks/minute)")
            print(f"   🎯 Success Rate: {successful/config['stocks']*100:.1f}%")
            print()
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            print()
            continue
    
    # Summary
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"{result['config']:15} | {result['stocks_analyzed']:3d} stocks | "
              f"{result['duration_minutes']:6.2f} min | {result['rate_per_minute']:6.1f} stocks/min")
    
    print()
    
    # Projections
    if results:
        avg_rate = sum(r['rate_per_minute'] for r in results) / len(results)
        print("🔮 PROJECTIONS (based on average rate)")
        print("-" * 40)
        
        projections = [100, 200, 300, 400, 500]
        for stocks in projections:
            estimated_minutes = stocks / avg_rate
            print(f"{stocks:3d} stocks: ~{estimated_minutes:5.1f} minutes ({estimated_minutes/60:4.1f} hours)")
    
    print()
    print("🎉 Performance test complete!")
    return results

if __name__ == "__main__":
    try:
        results = performance_test()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
