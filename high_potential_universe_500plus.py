#!/usr/bin/env python3
"""
500+ High-Potential Stock Universe for TFSA/Questrade
Targeting 5-100%+ profit potential across all market caps and industries
"""

def get_high_potential_universe_500plus():
    """Get 500+ high-potential stocks for maximum profit opportunities"""
    
    universe = [
        # === MEGA CAP TECH (5-50% potential) ===
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD',
        'ADBE', 'CRM', 'ORCL', 'INTU', 'ADP', 'CSCO', 'IBM', 'QCOM', 'TXN', 'AVGO',
        'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ADI', 'NOW', 'TEAM', 'WDAY', 'SNOW',
        'PLTR', 'DDOG', 'NET', 'CRWD', 'OKTA', 'ZM', 'DOCU', 'TWLO', 'SQ', 'PYPL',
        
        # === HIGH-GROWTH TECH (10-100%+ potential) ===
        'ROKU', 'PINS', 'SNAP', 'UBER', 'LYFT', 'ABNB', 'DASH', 'ZS', 'ESTC', 'MDB',
        'SPLK', 'VEEV', 'COUP', 'BILL', 'DOCN', 'FROG', 'AI', 'C3AI', 'SMCI', 'ARM',
        'RBLX', 'U', 'PATH', 'GTLB', 'S', 'WORK', 'ASAN', 'MNDY', 'ZI', 'CFLT',
        'SMAR', 'APPN', 'PCTY', 'NCNO', 'ALRM', 'TENB', 'CYBR', 'FEYE', 'PANW', 'FTNT',
        
        # === SEMICONDUCTOR BOOM (15-80% potential) ===
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MRVL', 'ADI', 'LRCX', 'KLAC',
        'AMAT', 'MU', 'SLAB', 'SWKS', 'QRVO', 'CRUS', 'SYNA', 'MPWR', 'MCHP', 'ON',
        'WOLF', 'CREE', 'NVEC', 'FORM', 'POWI', 'VICR', 'AOSL', 'AMBA', 'RMBS', 'SITM',
        'ALGM', 'DIOD', 'HIMX', 'MXL', 'PLAB', 'SMTC', 'TTMI', 'UCTT', 'VECO', 'XLNX',
        
        # === BIOTECH/PHARMA HIGH POTENTIAL (20-200%+ potential) ===
        'MRNA', 'BNTX', 'NVAX', 'VXRT', 'INO', 'OCGN', 'BNGO', 'PACB', 'ILMN', 'REGN',
        'GILD', 'BIIB', 'VRTX', 'AMGN', 'CELG', 'INCY', 'BMRN', 'ALXN', 'SGEN', 'TECH',
        'RARE', 'FOLD', 'ARWR', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'PRIME', 'VERV', 'SGMO',
        'BLUE', 'FATE', 'CGEM', 'RGNX', 'SRPT', 'IONS', 'EXAS', 'VEEV', 'TDOC', 'DXCM',
        'ISRG', 'INTUV', 'ALGN', 'HOLX', 'MASI', 'NEOG', 'NVCR', 'OMCL', 'PODD', 'TMDX',
        
        # === CLEAN ENERGY/EV REVOLUTION (25-150% potential) ===
        'TSLA', 'NIO', 'XPEV', 'LI', 'LCID', 'RIVN', 'F', 'GM', 'FORD', 'GOEV',
        'NKLA', 'RIDE', 'WKHS', 'HYLN', 'BLNK', 'CHPT', 'EVGO', 'PLUG', 'FCEL', 'BE',
        'ENPH', 'SEDG', 'SPWR', 'CSIQ', 'JKS', 'SOL', 'RUN', 'NOVA', 'MAXN', 'ARRY',
        'NEE', 'ICLN', 'PBW', 'QCLN', 'CNRG', 'ACES', 'SMOG', 'GRID', 'LIT', 'BATT',
        
        # === FINTECH DISRUPTION (15-120% potential) ===
        'SQ', 'PYPL', 'V', 'MA', 'COIN', 'HOOD', 'AFRM', 'UPST', 'LC', 'SOFI',
        'OPEN', 'RDFN', 'Z', 'ZG', 'COMP', 'TREE', 'CACC', 'WRLD', 'STNE', 'PAGS',
        'NU', 'MELI', 'SE', 'GRAB', 'BABA', 'JD', 'PDD', 'TCEHY', 'BIDU', 'TME',
        
        # === GAMING/METAVERSE (20-100% potential) ===
        'RBLX', 'U', 'EA', 'ATVI', 'TTWO', 'ZNGA', 'SKLZ', 'DKNG', 'PENN', 'MGM',
        'NVDA', 'AMD', 'INTC', 'QCOM', 'MRVL', 'LOGI', 'COR', 'HEAR', 'CRSR', 'SLGG',
        
        # === SPACE/DEFENSE (30-200% potential) ===
        'SPCE', 'ASTR', 'RDW', 'ASTS', 'PL', 'MAXR', 'SATS', 'LMT', 'RTX', 'NOC',
        'GD', 'LHX', 'TDG', 'HWM', 'TXT', 'KTOS', 'AVAV', 'CW', 'MOG-A', 'HEI',
        
        # === CANNABIS/PSYCHEDELICS (50-300% potential) ===
        'TLRY', 'CGC', 'ACB', 'CRON', 'SNDL', 'HEXO', 'OGI', 'APHA', 'CURLF', 'GTBIF',
        'TCNNF', 'CRLBF', 'MSOS', 'YOLO', 'THCX', 'POTX', 'MJ', 'CNBS', 'TOKE', 'BUDZ',
        
        # === SMALL CAP GROWTH (25-200%+ potential) ===
        'UPST', 'AFRM', 'SOFI', 'LC', 'OPEN', 'RDFN', 'COMP', 'TREE', 'CACC', 'WRLD',
        'PTON', 'BYND', 'OATLY', 'TTCF', 'VERY', 'UNFI', 'SFM', 'INGR', 'POST', 'K',
        'SMPL', 'CHEF', 'APPH', 'AVO', 'CALM', 'JJSF', 'LANC', 'LWAY', 'MGPI', 'RIBT',
        
        # === MICRO CAP HIGH POTENTIAL (50-500%+ potential) ===
        'BNGO', 'OCGN', 'SNDL', 'NAKD', 'GNUS', 'XSPA', 'SHIP', 'TOPS', 'GLBS', 'CTRM',
        'CASTOR', 'DARE', 'EARS', 'EXPR', 'FAMI', 'GEVO', 'HGEN', 'IDEX', 'JAGX', 'KOSS',
        'LKCO', 'MVIS', 'NNDM', 'OBSV', 'PROG', 'QUBT', 'RKDA', 'SENS', 'TELL', 'UAVS',
        'VBIV', 'WIMI', 'XELA', 'YTEN', 'ZOM', 'ABVC', 'ADTX', 'AEHR', 'ALPP', 'AMPE',
        
        # === CANADIAN HIGH POTENTIAL ===
        # Major Canadian Banks (5-25% potential)
        'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
        
        # Canadian Energy (15-80% potential)
        'ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'IMO.TO', 'CVE.TO', 'ARX.TO', 'MEG.TO',
        'WCP.TO', 'BTE.TO', 'CPG.TO', 'CJ.TO', 'ERF.TO', 'GXE.TO', 'HWX.TO', 'IPO.TO',
        'KEL.TO', 'NVA.TO', 'OBE.TO', 'POU.TO', 'SGY.TO', 'TVE.TO', 'VET.TO', 'WTE.TO',
        
        # Canadian Materials/Mining (20-100% potential)
        'ABX.TO', 'GOLD.TO', 'K.TO', 'CCO.TO', 'FM.TO', 'TKO.TO', 'HBM.TO', 'PAAS.TO',
        'AEM.TO', 'KL.TO', 'NGT.TO', 'ELD.TO', 'YRI.TO', 'EDV.TO', 'CG.TO', 'AR.TO',
        'AGI.TO', 'AUY.TO', 'B2GOLD.TO', 'CDE.TO', 'EGO.TO', 'FNV.TO', 'GFI.TO', 'HL.TO',
        
        # Canadian Tech (25-150% potential)
        'SHOP.TO', 'CSU.TO', 'OTEX.TO', 'NVEI.TO', 'LSPD.TO', 'NUVEI.TO', 'REAL.TO', 'DCBO.TO',
        'GDNP.TO', 'KXS.TO', 'MDF.TO', 'PKI.TO', 'QST.TO', 'RCG.TO', 'TOI.TO', 'WELL.TO',
        
        # Canadian Small/Mid Cap Growth (30-200% potential)
        'FOOD.TO', 'JAGG.TO', 'LUCK.TO', 'MOGO.TO', 'NTAR.TO', 'PMED.TO', 'QTRH.TO', 'RECO.TO',
        'SOLR.TO', 'TGOD.TO', 'VLNS.TO', 'WEED.TO', 'ZENA.TO', 'FIRE.TO', 'HEXO.TO', 'OGI.TO',
        
        # Canadian Utilities (10-40% potential)
        'FTS.TO', 'EMA.TO', 'CU.TO', 'H.TO', 'AQN.TO', 'BIP-UN.TO', 'ALA.TO', 'CPX.TO',
        
        # Canadian Telecom (5-30% potential)
        'T.TO', 'BCE.TO', 'RCI-B.TO', 'TMUS', 'VZ', 'T',
        
        # Canadian Consumer/Retail (15-60% potential)
        'L.TO', 'MG.TO', 'ATD.TO', 'DOL.TO', 'TFII.TO', 'GIB-A.TO', 'MTY.TO', 'QSR.TO',
        
        # Canadian REITs (10-50% potential)
        'REI-UN.TO', 'CAR-UN.TO', 'HR-UN.TO', 'FCR-UN.TO', 'SRU-UN.TO', 'CRT-UN.TO', 'DIR-UN.TO',
        
        # Canadian Industrials (15-70% potential)
        'CNR.TO', 'CP.TO', 'WSP.TO', 'STN.TO', 'TFI.TO', 'BYD.TO', 'TFII.TO', 'GFL.TO',
        
        # === US FINANCIAL SERVICES (10-60% potential) ===
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL',
        'COF', 'USB', 'PNC', 'TFC', 'BK', 'STT', 'TROW', 'BLK', 'SCHW', 'AMTD',
        'ICE', 'CME', 'NDAQ', 'MKTX', 'CBOE', 'MSCI', 'SPGI', 'MCO', 'FIS', 'FISV',
        
        # === US HEALTHCARE (15-80% potential) ===
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'LLY', 'CVS', 'CI', 'ELV', 'HUM', 'CNC', 'MOH', 'WCG', 'ANTM', 'WELL',
        
        # === US CONSUMER DISCRETIONARY (10-100% potential) ===
        'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS', 'TGT', 'COST', 'LOW', 'TJX',
        'BKNG', 'MAR', 'HLT', 'CCL', 'RCL', 'NCLH', 'MGM', 'LVS', 'WYNN', 'CZR',
        'LULU', 'DECK', 'CROX', 'SKX', 'VFC', 'RL', 'PVH', 'TPG', 'ETSY', 'W',
        
        # === US CONSUMER STAPLES (5-40% potential) ===
        'KO', 'PEP', 'WMT', 'PG', 'CL', 'KMB', 'CHD', 'CLX', 'TSN', 'K',
        'GIS', 'CPB', 'CAG', 'SJM', 'HSY', 'MDLZ', 'MNST', 'KDP', 'STZ', 'TAP',
        
        # === US INDUSTRIALS (15-70% potential) ===
        'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX',
        'NOC', 'GD', 'LHX', 'TDG', 'ITW', 'ETN', 'EMR', 'PH', 'DOV', 'CMI',
        'PCAR', 'WAB', 'CHRW', 'EXPD', 'JBHT', 'KNX', 'ODFL', 'SAIA', 'XPO', 'ARCB',
        
        # === US ENERGY (20-100% potential) ===
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'HES',
        'DVN', 'FANG', 'MRO', 'APA', 'CNX', 'AR', 'SM', 'NOG', 'CLR', 'WLL',
        'CTRA', 'OVV', 'RRC', 'MTDR', 'PR', 'VNOM', 'CRC', 'CPE', 'EQT', 'SWN',
        
        # === US MATERIALS (15-80% potential) ===
        'LIN', 'APD', 'SHW', 'ECL', 'IFF', 'PPG', 'EMN', 'FCX', 'NEM', 'GOLD',
        'AA', 'X', 'NUE', 'STLD', 'CLF', 'MT', 'TX', 'SCCO', 'TECK', 'VALE',
        'CF', 'MOS', 'FMC', 'LYB', 'DOW', 'DD', 'CE', 'RPM', 'SEE', 'PKG',
        
        # === US UTILITIES (5-35% potential) ===
        'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'XEL', 'SRE', 'PEG', 'ED',
        'EIX', 'ETR', 'ES', 'FE', 'AEE', 'CMS', 'DTE', 'NI', 'LNT', 'EVRG',
        
        # === US REAL ESTATE (10-60% potential) ===
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'MAA', 'UDR',
        'ESS', 'CPT', 'BXP', 'KIM', 'SPG', 'REG', 'FRT', 'SLG', 'VTR', 'WELL',
        'O', 'REALTY', 'NNN', 'ADC', 'FRT', 'KIM', 'MAC', 'PEI', 'REG', 'SKT',
        
        # === MEME STOCKS/HIGH VOLATILITY (50-1000%+ potential) ===
        'GME', 'AMC', 'BB', 'NOK', 'KOSS', 'EXPR', 'NAKD', 'SNDL', 'CLOV', 'WISH',
        'SPRT', 'IRNT', 'OPAD', 'ATER', 'BBIG', 'PROG', 'PHUN', 'DWAC', 'BENE', 'ANY',
        
        # === PENNY STOCKS WITH POTENTIAL (100-2000%+ potential) ===
        'BNGO', 'OCGN', 'SNDL', 'NAKD', 'GNUS', 'XSPA', 'SHIP', 'TOPS', 'GLBS', 'CTRM',
        'DARE', 'EARS', 'EXPR', 'FAMI', 'GEVO', 'HGEN', 'IDEX', 'JAGX', 'KOSS', 'LKCO',
        'MVIS', 'NNDM', 'OBSV', 'PROG', 'QUBT', 'RKDA', 'SENS', 'TELL', 'UAVS', 'VBIV',
        'WIMI', 'XELA', 'YTEN', 'ZOM', 'ABVC', 'ADTX', 'AEHR', 'ALPP', 'AMPE', 'ANVS',
        'APDN', 'ATNF', 'AVCT', 'BFRI', 'BHAT', 'BIOC', 'BKKT', 'BLRX', 'BMRA', 'BNTC',
    ]
    
    # Remove duplicates and sort
    universe = sorted(list(set(universe)))
    
    return universe

def get_profit_potential_categories():
    """Get profit potential categories"""
    
    categories = {
        'mega_cap_stable': {
            'potential': '5-50%',
            'timeframe': '6-24 months',
            'examples': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'risk': 'Low-Medium'
        },
        'high_growth_tech': {
            'potential': '10-100%+',
            'timeframe': '3-18 months',
            'examples': ['PLTR', 'SNOW', 'CRWD', 'NET'],
            'risk': 'Medium-High'
        },
        'biotech_breakthrough': {
            'potential': '20-200%+',
            'timeframe': '1-12 months',
            'examples': ['MRNA', 'NVAX', 'OCGN', 'BNGO'],
            'risk': 'High'
        },
        'clean_energy_revolution': {
            'potential': '25-150%',
            'timeframe': '6-36 months',
            'examples': ['TSLA', 'ENPH', 'PLUG', 'BE'],
            'risk': 'Medium-High'
        },
        'canadian_resources': {
            'potential': '15-80%',
            'timeframe': '3-24 months',
            'examples': ['CNQ.TO', 'ABX.TO', 'SHOP.TO'],
            'risk': 'Medium'
        },
        'penny_stocks_explosive': {
            'potential': '50-2000%+',
            'timeframe': '1-6 months',
            'examples': ['BNGO', 'SNDL', 'GNUS', 'MVIS'],
            'risk': 'Very High'
        },
        'meme_stocks_momentum': {
            'potential': '50-1000%+',
            'timeframe': '1-3 months',
            'examples': ['GME', 'AMC', 'BB', 'CLOV'],
            'risk': 'Extreme'
        }
    }
    
    return categories

if __name__ == "__main__":
    universe = get_high_potential_universe_500plus()
    categories = get_profit_potential_categories()
    
    print("🚀 HIGH-POTENTIAL STOCK UNIVERSE (500+)")
    print("=" * 60)
    print(f"Total stocks: {len(universe)}")
    
    # Count by exchange
    us_stocks = [s for s in universe if not s.endswith('.TO')]
    canadian_stocks = [s for s in universe if s.endswith('.TO')]
    
    print(f"US stocks: {len(us_stocks)}")
    print(f"Canadian stocks: {len(canadian_stocks)}")
    
    print(f"\n💰 PROFIT POTENTIAL CATEGORIES:")
    for category, details in categories.items():
        print(f"   {category}: {details['potential']} ({details['timeframe']})")
    
    print(f"\n🎯 SAMPLE HIGH-POTENTIAL STOCKS:")
    print("Mega Cap:", ', '.join([s for s in universe if s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']][:5]))
    print("Growth Tech:", ', '.join([s for s in universe if s in ['PLTR', 'SNOW', 'CRWD', 'NET', 'AI']][:5]))
    print("Biotech:", ', '.join([s for s in universe if s in ['MRNA', 'NVAX', 'OCGN', 'BNGO', 'PACB']][:5]))
    print("Canadian:", ', '.join([s for s in universe if s.endswith('.TO')][:5]))
    print("Penny Stocks:", ', '.join([s for s in universe if s in ['BNGO', 'SNDL', 'GNUS', 'MVIS', 'SENS']][:5]))
    
    print(f"\n✅ READY FOR HIGH-PROFIT ANALYSIS!")
    print(f"🎯 Targeting 5-2000%+ returns across all market caps!")
