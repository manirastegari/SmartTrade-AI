#!/usr/bin/env python3
"""
TFSA-Optimized Stock Universe for Canadian Questrade Trading
"""

def get_tfsa_optimized_universe():
    """Get comprehensive TFSA-eligible stock universe for Questrade"""
    
    universe = [
        # === US LARGE CAP TECH (TFSA ELIGIBLE) ===
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
        'ADBE', 'CRM', 'ORCL', 'INTU', 'ADP', 'CSCO', 'IBM', 'QCOM', 'TXN', 'AVGO',
        'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ADI', 'NOW', 'TEAM', 'WDAY', 'SNOW',
        
        # === US FINANCIAL SERVICES (TFSA ELIGIBLE) ===
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL',
        'COF', 'USB', 'PNC', 'TFC', 'BK', 'STT', 'TROW', 'BLK', 'SCHW',
        'ICE', 'CME', 'NDAQ', 'MKTX', 'CBOE', 'MSCI', 'SPGI', 'MCO', 'FIS', 'FISV',
        
        # === US HEALTHCARE (TFSA ELIGIBLE) ===
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'LLY', 'CVS', 'CI', 'ELV', 'GILD', 'VRTX', 'REGN', 'ILMN', 'MRNA', 'BNTX',
        'ZTS', 'SYK', 'ISRG', 'EW', 'BSX', 'MDT', 'BIIB', 'HUM', 'ANTM', 'CNC',
        
        # === US CONSUMER DISCRETIONARY (TFSA ELIGIBLE) ===
        'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS', 'TGT', 'COST', 'LOW', 'TJX',
        'BKNG', 'MAR', 'HLT', 'CCL', 'RCL', 'NCLH', 'MGM', 'LVS', 'WYNN', 'CZR',
        'LULU', 'DECK', 'CROX', 'SKX', 'VFC', 'RL', 'PVH', 'TPG', 'ETSY', 'W',
        
        # === US CONSUMER STAPLES (TFSA ELIGIBLE) ===
        'KO', 'PEP', 'WMT', 'PG', 'CL', 'KMB', 'CHD', 'CLX', 'TSN', 'K',
        'GIS', 'CPB', 'CAG', 'SJM', 'HSY', 'MDLZ', 'MNST', 'KDP', 'STZ', 'TAP',
        
        # === US INDUSTRIALS (TFSA ELIGIBLE) ===
        'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX',
        'NOC', 'GD', 'LHX', 'TDG', 'ITW', 'ETN', 'EMR', 'PH', 'DOV', 'CMI',
        'PCAR', 'WAB', 'CHRW', 'EXPD', 'JBHT', 'KNX', 'ODFL', 'SAIA', 'XPO', 'ARCB',
        
        # === US ENERGY (TFSA ELIGIBLE) ===
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'HES',
        'DVN', 'FANG', 'MRO', 'APA', 'CNX', 'AR', 'SM', 'NOG', 'CLR', 'WLL',
        
        # === US MATERIALS (TFSA ELIGIBLE) ===
        'LIN', 'APD', 'SHW', 'ECL', 'IFF', 'PPG', 'EMN', 'FCX', 'NEM', 'GOLD',
        'AA', 'X', 'NUE', 'STLD', 'CLF', 'MT', 'TX', 'SCCO', 'TECK', 'VALE',
        
        # === US UTILITIES (TFSA ELIGIBLE) ===
        'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'XEL', 'SRE', 'PEG', 'ED',
        'EIX', 'ETR', 'ES', 'FE', 'AEE', 'CMS', 'DTE', 'NI', 'LNT', 'EVRG',
        
        # === US REAL ESTATE (TFSA ELIGIBLE - REITS) ===
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'MAA', 'UDR',
        'ESS', 'CPT', 'BXP', 'KIM', 'SPG', 'REG', 'FRT', 'SLG', 'VTR', 'WELL',
        
        # === CANADIAN STOCKS (TFSA ELIGIBLE) ===
        # Major Canadian Banks
        'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
        
        # Canadian Energy
        'ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'IMO.TO', 'CVE.TO', 'ARX.TO', 'MEG.TO',
        
        # Canadian Materials/Mining
        'ABX.TO', 'GOLD.TO', 'K.TO', 'CCO.TO', 'FM.TO', 'TKO.TO', 'HBM.TO', 'PAAS.TO',
        
        # Canadian Technology
        'SHOP.TO', 'CSU.TO', 'OTEX.TO', 'NVEI.TO', 'LSPD.TO', 'NUVEI.TO',
        
        # Canadian Utilities
        'FTS.TO', 'EMA.TO', 'CU.TO', 'H.TO', 'AQN.TO', 'BIP-UN.TO',
        
        # Canadian Telecom
        'T.TO', 'BCE.TO', 'RCI-B.TO', 'TMUS', 'VZ', 'T',
        
        # Canadian Consumer/Retail
        'L.TO', 'MG.TO', 'ATD.TO', 'DOL.TO', 'TFII.TO', 'GIB-A.TO',
        
        # Canadian REITs (TFSA Eligible)
        'REI-UN.TO', 'CAR-UN.TO', 'HR-UN.TO', 'FCR-UN.TO', 'SRU-UN.TO',
        
        # Canadian Industrials
        'CNR.TO', 'CP.TO', 'WSP.TO', 'STN.TO', 'TFI.TO', 'BYD.TO',
        
        # === GROWTH STOCKS (TFSA ELIGIBLE) ===
        'PLTR', 'DDOG', 'NET', 'CRWD', 'OKTA', 'ZM', 'DOCU', 'TWLO', 'ROKU', 'SQ',
        'UBER', 'LYFT', 'ABNB', 'DASH', 'PINS', 'SNAP', 'ZS', 'ESTC', 'MDB',
        
        # === DIVIDEND ARISTOCRATS (TFSA ELIGIBLE) ===
        'JNJ', 'KO', 'PG', 'MMM', 'CAT', 'CVX', 'XOM', 'WMT', 'MCD', 'HD',
        'VZ', 'T', 'IBM', 'GE', 'BA', 'DIS', 'NKE', 'MSFT', 'AAPL', 'JPM',
        
        # === SEMICONDUCTOR FOCUS (TFSA ELIGIBLE) ===
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MRVL', 'ADI', 'LRCX', 'KLAC',
        'AMAT', 'MU', 'SLAB', 'SWKS', 'QRVO', 'CRUS', 'SYNA', 'MPWR', 'MCHP', 'ON',
        
        # === BIOTECH FOCUS (TFSA ELIGIBLE) ===
        'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'MRNA', 'BNTX', 'AMGN', 'CELG', 'INCY',
        'BMRN', 'ALXN', 'SGEN', 'TECH', 'RARE', 'FOLD', 'ARWR', 'EDIT', 'CRSP', 'NTLA',
    ]
    
    # Remove duplicates and sort
    universe = sorted(list(set(universe)))
    
    return universe

def get_tfsa_compliance_info():
    """Get TFSA compliance information"""
    
    info = {
        'eligible_criteria': [
            'US and Canadian publicly traded companies',
            'Listed on major exchanges (NYSE, NASDAQ, TSX)',
            'Not foreign domiciled (non-US/Canada)',
            'Not MLPs (Master Limited Partnerships)',
            'Not certain ETFs/ETNs with complex structures',
            'Not penny stocks (generally above $1)',
            'Available on Questrade platform'
        ],
        'questrade_availability': [
            'All major US exchanges supported',
            'TSX and TSX-V supported',
            'Real-time quotes available',
            'Options trading available for most',
            'Fractional shares available for some'
        ],
        'tax_advantages': [
            'No tax on capital gains',
            'No tax on dividends',
            'No tax on interest income',
            'Tax-free growth',
            'Tax-free withdrawals'
        ],
        'contribution_limits': {
            '2024': 7000,
            '2023': 6500,
            '2022': 6000,
            'lifetime_max': 95000  # Approximate as of 2024
        }
    }
    
    return info

if __name__ == "__main__":
    universe = get_tfsa_optimized_universe()
    info = get_tfsa_compliance_info()
    
    print("🇨🇦 TFSA-OPTIMIZED STOCK UNIVERSE")
    print("=" * 50)
    print(f"Total stocks: {len(universe)}")
    
    # Count by exchange
    us_stocks = [s for s in universe if not s.endswith('.TO')]
    canadian_stocks = [s for s in universe if s.endswith('.TO')]
    
    print(f"US stocks: {len(us_stocks)}")
    print(f"Canadian stocks: {len(canadian_stocks)}")
    
    print(f"\n📊 SAMPLE STOCKS:")
    print("US Stocks:", ', '.join(us_stocks[:10]))
    print("Canadian Stocks:", ', '.join(canadian_stocks[:10]))
    
    print(f"\n🛡️ TFSA COMPLIANCE:")
    for criteria in info['eligible_criteria'][:5]:
        print(f"   ✅ {criteria}")
    
    print(f"\n💰 2024 TFSA CONTRIBUTION LIMIT: ${info['contribution_limits']['2024']:,}")
    print(f"📈 Perfect for long-term tax-free growth!")
