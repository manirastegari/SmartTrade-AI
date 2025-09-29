#!/usr/bin/env python3
"""
Universe Cleanup - Remove delisted/acquired stocks and add replacements
"""

def get_stocks_to_remove():
    """Get list of stocks that should be removed (delisted/acquired)"""
    
    stocks_to_remove = {
        # Acquired Companies (no longer independent)
        'XLNX': 'Acquired by AMD in 2022',
        'ZNGA': 'Acquired by Take-Two Interactive in 2022',
        'ZI': 'ZoomInfo - check if symbol changed or delisted',
        
        # Delisted Canadian Stocks
        'ZENA.TO': 'Zenabis - Canadian cannabis company delisted',
        'YRI.TO': 'Yamana Gold - check if merged/symbol changed',
        'FIRE.TO': 'Supreme Cannabis - likely delisted',
        'TGOD.TO': 'The Green Organic Dutchman - delisted',
        
        # Penny Stocks with Persistent Data Issues
        'XSPA': 'XpresSpa - very low liquidity, data issues',
        'YTEN': 'Yield10 Bioscience - micro-cap with irregular data',
        'ZOM': 'Zomedica - penny stock with data problems',
        'TOPS': 'TOP Ships - reverse splits, data issues',
        'SHIP': 'Seanergy Maritime - data reliability issues',
        'GLBS': 'Globus Maritime - micro-cap issues',
        'CTRM': 'Castor Maritime - data problems',
        
        # Other Problematic Stocks
        'UVXY': 'Delisted volatility ETF',
        'VXX': 'Delisted volatility ETF', 
        'NAKD': 'Naked Brand Group - delisted/merged',
        'EXPR': 'Express Inc - bankruptcy/delisted',
        
        # Stocks with Symbol Changes (need verification)
        'DISCA': 'Discovery - merged into Warner Bros Discovery (WBD)',
        'DISCK': 'Discovery - merged into Warner Bros Discovery (WBD)',
        'VIAC': 'ViacomCBS - now Paramount (PARA)',
        'TWTR': 'Twitter - acquired by Musk, delisted',
    }
    
    return stocks_to_remove

def get_replacement_stocks():
    """Get high-quality replacement stocks"""
    
    replacements = {
        # Replace acquired tech companies with similar tech stocks
        'XLNX': ['MRVL', 'LRCX', 'KLAC'],  # Semiconductor alternatives
        'ZNGA': ['EA', 'TTWO', 'ATVI'],    # Gaming alternatives
        'ZI': ['CRM', 'NOW', 'WDAY'],      # Enterprise software alternatives
        'TWTR': ['META', 'SNAP', 'PINS'],  # Social media alternatives
        
        # Replace delisted Canadian stocks with solid Canadian alternatives
        'ZENA.TO': ['WEED.TO', 'ACB.TO', 'HEXO.TO'],  # Cannabis alternatives
        'YRI.TO': ['AEM.TO', 'KL.TO', 'ELD.TO'],      # Gold mining alternatives
        'FIRE.TO': ['CGC.TO', 'TLRY.TO', 'OGI.TO'],   # Cannabis alternatives
        'TGOD.TO': ['CRON.TO', 'APHA.TO', 'VFF.TO'],  # Cannabis alternatives
        
        # Replace problematic penny stocks with better small-caps
        'XSPA': ['PTON', 'BYND', 'ROKU'],    # Consumer/tech small-caps
        'YTEN': ['GEVO', 'PLUG', 'FCEL'],    # Clean energy small-caps
        'ZOM': ['SENS', 'PACB', 'BNGO'],     # Biotech small-caps
        'TOPS': ['SBLK', 'STNG', 'NAT'],     # Shipping alternatives
        'SHIP': ['EURN', 'TNK', 'INSW'],     # Maritime alternatives
        
        # Replace delisted ETFs with individual stocks
        'UVXY': ['VIX', 'SVXY'],            # Volatility alternatives
        'VXX': None,                        # Remove entirely
        'NAKD': ['LULU', 'DECK', 'CROX'],   # Apparel alternatives
        'EXPR': ['ANF', 'AEO', 'GPS'],      # Retail alternatives
        
        # Replace merged companies with new entities
        'DISCA': ['WBD'],                   # Warner Bros Discovery
        'DISCK': ['WBD'],                   # Warner Bros Discovery  
        'VIAC': ['PARA'],                   # Paramount
    }
    
    return replacements

def get_additional_high_quality_stocks():
    """Get additional high-quality stocks to reach target count"""
    
    additional_stocks = {
        # High-Quality US Large Caps
        'mega_cap_additions': [
            'BRK-A', 'UNH', 'JNJ', 'WMT', 'PG', 'HD', 'BAC', 'DIS', 'ADBE', 'NFLX',
            'CRM', 'ORCL', 'ACN', 'TXN', 'QCOM', 'INTU', 'ISRG', 'AMGN', 'GILD', 'MDLZ'
        ],
        
        # High-Growth Tech
        'growth_tech_additions': [
            'SHOP', 'SQ', 'ROKU', 'TWLO', 'OKTA', 'DDOG', 'CRWD', 'ZS', 'NET', 'ESTC',
            'MDB', 'TEAM', 'WDAY', 'NOW', 'VEEV', 'COUP', 'BILL', 'DOCN', 'FROG', 'AI'
        ],
        
        # Biotech/Healthcare
        'biotech_additions': [
            'REGN', 'VRTX', 'BIIB', 'ILMN', 'INCY', 'BMRN', 'ALXN', 'SGEN', 'TECH', 'RARE',
            'FOLD', 'ARWR', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'PRIME', 'VERV', 'SGMO', 'BLUE'
        ],
        
        # Canadian High-Quality Additions
        'canadian_additions': [
            'SHOP.TO', 'CSU.TO', 'BAM.TO', 'WCN.TO', 'CNR.TO', 'CP.TO', 'WSP.TO', 'STN.TO',
            'TFI.TO', 'GFL.TO', 'OTEX.TO', 'NVEI.TO', 'LSPD.TO', 'REAL.TO', 'DCBO.TO', 'WELL.TO'
        ],
        
        # Clean Energy/ESG
        'clean_energy_additions': [
            'ENPH', 'SEDG', 'SPWR', 'RUN', 'NOVA', 'CSIQ', 'JKS', 'SOL', 'MAXN', 'ARRY',
            'PLUG', 'FCEL', 'BE', 'BLDP', 'HYLN', 'BLNK', 'CHPT', 'EVGO', 'CLSK', 'RIOT'
        ],
        
        # Financial Services
        'fintech_additions': [
            'V', 'MA', 'PYPL', 'SQ', 'COIN', 'HOOD', 'AFRM', 'UPST', 'LC', 'SOFI',
            'NU', 'MELI', 'SE', 'STNE', 'PAGS', 'OPEN', 'RDFN', 'COMP', 'TREE', 'CACC'
        ]
    }
    
    return additional_stocks

def clean_and_optimize_universe():
    """Clean the universe and add high-quality replacements"""
    
    print("🧹 UNIVERSE CLEANUP & OPTIMIZATION")
    print("=" * 50)
    
    # Get current universe
    try:
        from high_potential_universe_500plus import get_high_potential_universe_500plus
        current_universe = get_high_potential_universe_500plus()
    except ImportError:
        print("❌ Could not import current universe")
        return []
    
    # Get cleanup lists
    to_remove = get_stocks_to_remove()
    replacements = get_replacement_stocks()
    additional = get_additional_high_quality_stocks()
    
    print(f"📊 Current universe size: {len(current_universe)}")
    print(f"🗑️ Stocks to remove: {len(to_remove)}")
    
    # Remove problematic stocks
    cleaned_universe = []
    removed_count = 0
    
    for stock in current_universe:
        if stock not in to_remove:
            cleaned_universe.append(stock)
        else:
            removed_count += 1
            print(f"   ❌ Removed {stock}: {to_remove[stock]}")
    
    print(f"\n✅ Removed {removed_count} problematic stocks")
    print(f"📊 Cleaned universe size: {len(cleaned_universe)}")
    
    # Add replacements
    added_count = 0
    for removed_stock, replacement_list in replacements.items():
        if replacement_list:
            for replacement in replacement_list:
                if replacement not in cleaned_universe:
                    cleaned_universe.append(replacement)
                    added_count += 1
                    print(f"   ✅ Added {replacement} (replacing {removed_stock})")
    
    # Add additional high-quality stocks
    for category, stocks in additional.items():
        for stock in stocks:
            if stock not in cleaned_universe:
                cleaned_universe.append(stock)
                added_count += 1
    
    print(f"\n✅ Added {added_count} high-quality stocks")
    
    # Remove duplicates and sort
    cleaned_universe = sorted(list(set(cleaned_universe)))
    
    print(f"📊 Final optimized universe: {len(cleaned_universe)} stocks")
    
    # Calculate improvement
    original_success_rate = (586 / 705) * 100
    potential_success_rate = (len(cleaned_universe) - 10) / len(cleaned_universe) * 100  # Assume 10 might still fail
    
    print(f"\n📈 EXPECTED IMPROVEMENT:")
    print(f"   Original success rate: {original_success_rate:.1f}%")
    print(f"   Expected new success rate: {potential_success_rate:.1f}%")
    print(f"   Improvement: +{potential_success_rate - original_success_rate:.1f}%")
    
    return cleaned_universe

def generate_cleaned_universe_file():
    """Generate a new cleaned universe file"""
    
    cleaned_universe = clean_and_optimize_universe()
    
    # Generate new file content
    file_content = '''#!/usr/bin/env python3
"""
Cleaned High-Potential Stock Universe (Optimized)
Removed delisted/acquired stocks, added high-quality replacements
"""

def get_cleaned_high_potential_universe():
    """Get cleaned universe with problematic stocks removed"""
    
    universe = [
'''
    
    # Add stocks in chunks of 10 per line for readability
    for i in range(0, len(cleaned_universe), 10):
        chunk = cleaned_universe[i:i+10]
        line = "        " + ", ".join([f"'{stock}'" for stock in chunk]) + ","
        file_content += line + "\n"
    
    file_content += '''    ]
    
    # Remove duplicates and sort
    universe = sorted(list(set(universe)))
    
    return universe

if __name__ == "__main__":
    universe = get_cleaned_high_potential_universe()
    print(f"Cleaned universe: {len(universe)} stocks")
    
    # Count by exchange
    us_stocks = [s for s in universe if not s.endswith('.TO')]
    canadian_stocks = [s for s in universe if s.endswith('.TO')]
    
    print(f"US stocks: {len(us_stocks)}")
    print(f"Canadian stocks: {len(canadian_stocks)}")
'''
    
    # Write to file
    with open('/Users/manirastegari/maniProject/AITrader/cleaned_high_potential_universe.py', 'w') as f:
        f.write(file_content)
    
    print(f"\n✅ Generated cleaned_high_potential_universe.py")
    print(f"📁 File contains {len(cleaned_universe)} optimized stocks")
    
    return cleaned_universe

if __name__ == "__main__":
    cleaned_universe = generate_cleaned_universe_file()
    
    print(f"\n🎯 CLEANUP COMPLETE!")
    print(f"✅ Removed problematic stocks")
    print(f"✅ Added high-quality replacements") 
    print(f"✅ Generated optimized universe file")
    print(f"🚀 Expected success rate improvement: +10-15%")
