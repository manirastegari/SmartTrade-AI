#!/usr/bin/env python3
"""
Debug script to check the stock universe size and fix cap selection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_analyzer import AdvancedTradingAnalyzer

def debug_universe():
    """Debug the stock universe"""
    print("🔍 Debugging Stock Universe...")
    
    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    universe = analyzer.stock_universe
    
    print(f"📊 Total symbols in universe: {len(universe)}")
    print(f"📊 First 10 symbols: {universe[:10]}")
    print(f"📊 Symbols 200-210: {universe[200:210] if len(universe) > 210 else 'Not enough symbols'}")
    print(f"📊 Symbols 500-510: {universe[500:510] if len(universe) > 510 else 'Not enough symbols'}")
    print(f"📊 Last 10 symbols: {universe[-10:]}")
    
    # Check for duplicates
    unique_symbols = list(set(universe))
    duplicates = len(universe) - len(unique_symbols)
    print(f"📊 Duplicate symbols: {duplicates}")
    
    if duplicates > 0:
        from collections import Counter
        counts = Counter(universe)
        dups = {symbol: count for symbol, count in counts.items() if count > 1}
        print(f"📊 Duplicate symbols found: {dups}")

if __name__ == "__main__":
    debug_universe()
