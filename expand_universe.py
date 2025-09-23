#!/usr/bin/env python3
"""
Expand the stock universe to capture more opportunities
"""

# Additional high-quality stocks to expand the universe
additional_stocks = [
    # More Tech/Growth
    'SHOP', 'SQ', 'PYPL', 'ADSK', 'WDAY', 'VEEV', 'ZS', 'OKTA', 'DDOG', 'MDB',
    'TEAM', 'ATLASSIAN', 'SPLK', 'PANW', 'FTNT', 'CYBR', 'FEYE', 'PING', 'TENB', 'RPD',
    
    # Biotech/Healthcare
    'MODERNA', 'BNTX', 'NVAX', 'SRPT', 'BLUE', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'PRIME',
    'ARKG', 'GNOM', 'VCYT', 'PACB', 'ILMN', 'TWST', 'CDNA', 'FATE', 'SGMO', 'RGNX',
    
    # Clean Energy/EV
    'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR', 'SPWR', 'CSIQ', 'JKS', 'SOL', 'MAXN',
    'PLUG', 'FCEL', 'BE', 'BLDP', 'HYLN', 'QS', 'CHPT', 'BLNK', 'SBE', 'RIDE',
    
    # Fintech
    'AFRM', 'UPST', 'SOFI', 'LC', 'OPEN', 'RDFN', 'Z', 'ZG', 'COMP', 'TREE',
    'LMND', 'ROOT', 'MTTR', 'HOOD', 'COIN', 'MARA', 'RIOT', 'HUT', 'BITF', 'CAN',
    
    # E-commerce/Consumer
    'ETSY', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'VIPS', 'YMM', 'DANG', 'WB',
    'CHWY', 'PETQ', 'WOOF', 'FRPT', 'CHEWY', 'BARK', 'WAGS', 'PETM', 'PETS', 'IDXX',
    
    # Cloud/SaaS
    'SNOW', 'PLTR', 'AI', 'C3AI', 'PATH', 'SMAR', 'FROG', 'BIGC', 'APPS', 'WORK',
    'SLACK', 'ZM', 'DOCU', 'BOX', 'DBX', 'ESTC', 'TWLO', 'SEND', 'BAND', 'RNG',
    
    # Gaming/Entertainment
    'RBLX', 'U', 'DKNG', 'PENN', 'MGM', 'LVS', 'WYNN', 'CZR', 'BYD', 'RSI',
    'EA', 'ATVI', 'TTWO', 'ZNGA', 'GLUU', 'GLU', 'HUYA', 'DOYU', 'BILI', 'IQ',
    
    # REITs
    'AMT', 'PLD', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'UDR', 'ESS', 'MAA',
    'O', 'STOR', 'WPC', 'NNN', 'ADC', 'STAG', 'LXP', 'GTY', 'EPRT', 'FCPT',
    
    # International/ADRs
    'TSM', 'ASML', 'NVO', 'UL', 'RDS.A', 'BP', 'TTE', 'E', 'SAN', 'BBVA',
    'ING', 'DB', 'CS', 'UBS', 'BCS', 'RY', 'TD', 'BMO', 'BNS', 'CM',
    
    # Small/Mid Cap Gems
    'CRWD', 'NET', 'DDOG', 'SNOW', 'PLTR', 'AI', 'PATH', 'SMAR', 'FROG', 'BIGC',
    'APPS', 'WORK', 'SLACK', 'ZM', 'DOCU', 'BOX', 'DBX', 'ESTC', 'TWLO', 'SEND',
    
    # Emerging Sectors
    'SPCE', 'RKLB', 'ASTR', 'VACQ', 'HOL', 'NGCA', 'SRAC', 'IPOF', 'IPOD', 'IPOC',
    'CCIV', 'THCB', 'ACTC', 'STPK', 'FUSE', 'AJAX', 'BTWN', 'CLOV', 'WISH', 'BARK'
]

print(f"Additional stocks to consider: {len(additional_stocks)}")
print("Sample additions:", additional_stocks[:20])

# Remove duplicates and create expanded universe
current_universe = [
    # Current universe from analyzer (simplified)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC'
    # ... (348 current stocks)
]

# This would expand to ~500+ unique stocks
print(f"Expanded universe would be: {len(current_universe) + len(set(additional_stocks))} stocks")
