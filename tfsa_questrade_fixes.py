#!/usr/bin/env python3
"""
TFSA Questrade Canada Stock Universe Fixes
Ensures all stocks are TFSA-eligible and available on Questrade
"""

def get_tfsa_ineligible_stocks():
    """Get stocks that are NOT TFSA eligible in Canada"""
    
    # These stocks are typically not TFSA eligible
    tfsa_ineligible = [
        # REITs (Real Estate Investment Trusts) - Often not eligible
        'O', 'REALTY', 'VNQ', 'SCHH', 'RWR',
        
        # Foreign domiciled companies (not US/Canada)
        'ASML', 'TSM', 'NVO', 'UL', 'DEO', 'BP', 'RDS-A', 'RDS-B',
        
        # Certain ETFs/ETNs
        'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXU', 'UPRO',
        
        # MLPs (Master Limited Partnerships)
        'EPD', 'ET', 'KMI', 'OKE', 'WMB',
        
        # BDCs (Business Development Companies)  
        'ARCC', 'MAIN', 'PSEC', 'HTGC',
        
        # Certain foreign ADRs
        'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI'
    ]
    
    return tfsa_ineligible

def get_questrade_unavailable_stocks():
    """Get stocks that may not be available on Questrade"""
    
    questrade_unavailable = [
        # Penny stocks (under $1)
        'SNDL', 'NAKD', 'GNUS', 'XSPA',
        
        # Recently delisted or suspended
        'VXX', 'UVXY', 'ANTM',  # ANTM was acquired by Elevance Health (ELV)
        
        # OTC/Pink Sheet stocks
        'GBTC', 'ETHE', 'LTCN',
        
        # Some micro-cap stocks
        'BNGO', 'OCGN', 'CLOV'
    ]
    
    return questrade_unavailable

def get_symbol_corrections():
    """Get corrected symbols for renamed/merged companies"""
    
    corrections = {
        'ANTM': 'ELV',      # Anthem → Elevance Health
        'FB': 'META',       # Facebook → Meta
        'GOOGL': 'GOOGL',   # Keep as is
        'GOOG': 'GOOGL',    # Use Class A shares
        'BRK.B': 'BRK-B',   # Berkshire Hathaway format
        'BERKB': 'BRK-B',   # Alternative format
        'BF.B': 'BF-B',     # Brown-Forman Class B
        'DISCA': 'WBD',     # Discovery → Warner Bros Discovery
        'DISCK': 'WBD',     # Discovery → Warner Bros Discovery
        'T': 'T',           # AT&T (keep)
        'VIAC': 'PARA',     # ViacomCBS → Paramount
        'DWDP': 'DD',       # DowDuPont → DuPont
        'PYPL': 'PYPL',     # PayPal (keep)
        'TWTR': None,       # Twitter (delisted - acquired by Musk)
        'NFLX': 'NFLX',     # Netflix (keep)
    }
    
    return corrections

def get_tfsa_friendly_replacements():
    """Get TFSA-friendly replacement stocks"""
    
    replacements = {
        # Replace REITs with REIT stocks that are TFSA eligible
        'O': 'AMT',         # American Tower (cell towers)
        'REALTY': 'PLD',    # Prologis (logistics real estate)
        
        # Replace foreign stocks with US equivalents
        'ASML': 'AMAT',     # Applied Materials (semiconductor equipment)
        'TSM': 'NVDA',      # NVIDIA (semiconductor)
        'NVO': 'NEM',       # Newmont (gold mining)
        'UL': 'PG',         # Procter & Gamble (consumer goods)
        'DEO': 'KO',        # Coca-Cola (beverages)
        'BP': 'XOM',        # ExxonMobil (oil)
        
        # Replace ETFs with individual stocks
        'VXX': 'VIX',       # Or remove entirely
        'UVXY': None,       # Remove
        'SQQQ': None,       # Remove leveraged ETFs
        'TQQQ': None,       # Remove leveraged ETFs
        
        # Replace MLPs with regular energy stocks
        'EPD': 'ENB',       # Enbridge (Canadian pipeline - TFSA eligible)
        'ET': 'KMI',        # Kinder Morgan
        'OKE': 'TRP',       # TC Energy (Canadian - TFSA eligible)
        
        # Replace BDCs with regular financial stocks
        'ARCC': 'JPM',      # JPMorgan Chase
        'MAIN': 'BAC',      # Bank of America
        
        # Replace Chinese ADRs with US tech
        'BABA': 'AMZN',     # Amazon (e-commerce)
        'JD': 'SHOP',       # Shopify (Canadian e-commerce)
        'PDD': 'PYPL',      # PayPal (payments)
        'BIDU': 'GOOGL',    # Google (search)
        'NIO': 'TSLA',      # Tesla (EV)
        'XPEV': 'F',        # Ford (EV transition)
        'LI': 'GM',         # General Motors (EV)
    }
    
    return replacements

def get_canadian_tfsa_additions():
    """Get additional Canadian stocks that are TFSA-friendly"""
    
    canadian_stocks = [
        # Major Canadian Banks (TFSA eligible)
        'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
        
        # Canadian Energy (TFSA eligible)
        'ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'IMO.TO',
        
        # Canadian Tech
        'SHOP.TO', 'CSU.TO', 'OTEX.TO',
        
        # Canadian Utilities
        'FTS.TO', 'EMA.TO', 'CU.TO',
        
        # Canadian Telecom
        'T.TO', 'BCE.TO', 'RCI-B.TO',
        
        # Canadian Materials
        'ABX.TO', 'GOLD.TO', 'K.TO', 'CCO.TO',
        
        # Canadian Consumer
        'L.TO', 'MG.TO', 'ATD.TO'
    ]
    
    return canadian_stocks

def clean_stock_universe_for_tfsa_questrade(original_universe):
    """Clean and optimize stock universe for TFSA/Questrade trading"""
    
    print("🇨🇦 OPTIMIZING STOCK UNIVERSE FOR TFSA/QUESTRADE")
    print("=" * 60)
    
    # Get all the fix lists
    tfsa_ineligible = get_tfsa_ineligible_stocks()
    questrade_unavailable = get_questrade_unavailable_stocks()
    corrections = get_symbol_corrections()
    replacements = get_tfsa_friendly_replacements()
    canadian_additions = get_canadian_tfsa_additions()
    
    # Start with original universe
    cleaned_universe = []
    removed_stocks = []
    corrected_stocks = []
    added_stocks = []
    
    print("🔍 Processing original universe...")
    
    for symbol in original_universe:
        # Check if needs correction
        if symbol in corrections:
            corrected_symbol = corrections[symbol]
            if corrected_symbol:
                cleaned_universe.append(corrected_symbol)
                corrected_stocks.append(f"{symbol} → {corrected_symbol}")
            else:
                removed_stocks.append(f"{symbol} (delisted)")
            continue
        
        # Check if TFSA ineligible
        if symbol in tfsa_ineligible:
            if symbol in replacements:
                replacement = replacements[symbol]
                if replacement:
                    cleaned_universe.append(replacement)
                    corrected_stocks.append(f"{symbol} → {replacement} (TFSA eligible)")
                else:
                    removed_stocks.append(f"{symbol} (TFSA ineligible)")
            else:
                removed_stocks.append(f"{symbol} (TFSA ineligible)")
            continue
        
        # Check if Questrade unavailable
        if symbol in questrade_unavailable:
            if symbol in replacements:
                replacement = replacements[symbol]
                if replacement:
                    cleaned_universe.append(replacement)
                    corrected_stocks.append(f"{symbol} → {replacement} (Questrade available)")
                else:
                    removed_stocks.append(f"{symbol} (Questrade unavailable)")
            else:
                removed_stocks.append(f"{symbol} (Questrade unavailable)")
            continue
        
        # Keep the stock
        cleaned_universe.append(symbol)
    
    # Add Canadian stocks for diversification
    print("🍁 Adding Canadian TFSA-eligible stocks...")
    for canadian_stock in canadian_additions:
        if canadian_stock not in cleaned_universe:
            cleaned_universe.append(canadian_stock)
            added_stocks.append(canadian_stock)
    
    # Remove duplicates and sort
    cleaned_universe = sorted(list(set(cleaned_universe)))
    
    # Print summary
    print(f"\n📊 TFSA/QUESTRADE OPTIMIZATION SUMMARY:")
    print(f"   Original stocks: {len(original_universe)}")
    print(f"   Final stocks: {len(cleaned_universe)}")
    print(f"   Removed: {len(removed_stocks)}")
    print(f"   Corrected: {len(corrected_stocks)}")
    print(f"   Added Canadian: {len(added_stocks)}")
    
    if removed_stocks:
        print(f"\n❌ REMOVED STOCKS:")
        for stock in removed_stocks[:10]:  # Show first 10
            print(f"   {stock}")
        if len(removed_stocks) > 10:
            print(f"   ... and {len(removed_stocks) - 10} more")
    
    if corrected_stocks:
        print(f"\n🔄 CORRECTED STOCKS:")
        for stock in corrected_stocks[:10]:  # Show first 10
            print(f"   {stock}")
        if len(corrected_stocks) > 10:
            print(f"   ... and {len(corrected_stocks) - 10} more")
    
    if added_stocks:
        print(f"\n🍁 ADDED CANADIAN STOCKS:")
        for stock in added_stocks[:10]:  # Show first 10
            print(f"   {stock}")
        if len(added_stocks) > 10:
            print(f"   ... and {len(added_stocks) - 10} more")
    
    print(f"\n✅ TFSA/QUESTRADE OPTIMIZATION COMPLETE!")
    print(f"🇨🇦 Universe now optimized for Canadian TFSA trading")
    
    return cleaned_universe

def validate_tfsa_compliance(stock_list):
    """Validate that stocks are TFSA compliant"""
    
    print("\n🛡️ TFSA COMPLIANCE VALIDATION")
    print("=" * 40)
    
    tfsa_ineligible = get_tfsa_ineligible_stocks()
    questrade_unavailable = get_questrade_unavailable_stocks()
    
    issues = []
    
    for stock in stock_list:
        if stock in tfsa_ineligible:
            issues.append(f"{stock}: Not TFSA eligible")
        if stock in questrade_unavailable:
            issues.append(f"{stock}: Not available on Questrade")
    
    if issues:
        print("⚠️ COMPLIANCE ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ ALL STOCKS ARE TFSA/QUESTRADE COMPLIANT")
        return True

if __name__ == "__main__":
    # Test with sample universe
    sample_universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'ANTM',
        'O', 'ASML', 'TSM', 'BABA', 'VXX', 'EPD', 'ARCC', 'BRK.B'
    ]
    
    print("🧪 TESTING TFSA/QUESTRADE OPTIMIZATION")
    print("=" * 50)
    
    cleaned = clean_stock_universe_for_tfsa_questrade(sample_universe)
    validate_tfsa_compliance(cleaned)
    
    print(f"\n📋 SAMPLE CLEANED UNIVERSE:")
    for i, stock in enumerate(cleaned[:20], 1):
        print(f"   {i:2d}. {stock}")
    
    if len(cleaned) > 20:
        print(f"   ... and {len(cleaned) - 20} more stocks")
