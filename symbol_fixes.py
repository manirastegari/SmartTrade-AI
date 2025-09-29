#!/usr/bin/env python3
"""
Symbol Fixes for AI Trading Application
"""

def get_symbol_mappings():
    """Get correct symbol mappings"""
    
    symbol_fixes = {
        'BERKB': 'BRK-B',  # Berkshire Hathaway Class B
        'BRKB': 'BRK-B',   # Alternative spelling
        'BF-B': 'BF.B',    # Brown-Forman Class B
        'BRK.B': 'BRK-B'   # Yahoo Finance format
    }
    
    delisted_symbols = [
        'VXX',   # Delisted volatility ETF
        'UVXY',  # May be delisted or suspended
    ]
    
    return symbol_fixes, delisted_symbols

def fix_symbol(symbol):
    """Fix common symbol issues"""
    
    symbol_fixes, delisted_symbols = get_symbol_mappings()
    
    # Check if delisted
    if symbol in delisted_symbols:
        return None, f"Symbol {symbol} is delisted"
    
    # Apply fixes
    if symbol in symbol_fixes:
        fixed_symbol = symbol_fixes[symbol]
        return fixed_symbol, f"Fixed {symbol} → {fixed_symbol}"
    
    return symbol, "No fix needed"

if __name__ == "__main__":
    test_symbols = ['BERKB', 'VXX', 'AAPL', 'BRK.B']
    
    print("🔧 SYMBOL FIXES TEST")
    print("=" * 30)
    
    for symbol in test_symbols:
        fixed, message = fix_symbol(symbol)
        print(f"{symbol:8} → {fixed or 'SKIP':8} ({message})")
