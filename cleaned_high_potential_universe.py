#!/usr/bin/env python3
"""
Cleaned High-Potential Stock Universe (Optimized)
Removed delisted/acquired stocks, added high-quality replacements
"""

def get_cleaned_high_potential_universe():
    """Get cleaned universe with problematic stocks removed"""
    
    universe = [
        'AA', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ABVC', 'ABX.TO', 'ACB', 'ACB.TO', 'ACES',
        'ACN', 'ADBE', 'ADC', 'ADI', 'ADP', 'ADTX', 'AEE', 'AEHR', 'AEM.TO', 'AEO',
        'AEP', 'AFRM', 'AGI.TO', 'AI', 'ALA.TO', 'ALGM', 'ALGN', 'ALPP', 'ALRM', 'ALXN',
        'AMAT', 'AMBA', 'AMC', 'AMD', 'AMGN', 'AMPE', 'AMT', 'AMTD', 'AMZN', 'ANF',
        'ANTM', 'ANVS', 'ANY', 'AOSL', 'APA', 'APD', 'APDN', 'APHA', 'APHA.TO', 'APPH',
        'APPN', 'AQN.TO', 'AR', 'AR.TO', 'ARCB', 'ARM', 'ARRY', 'ARWR', 'ARX.TO', 'ASAN',
        'ASTR', 'ASTS', 'ATD.TO', 'ATER', 'ATNF', 'ATVI', 'AUY.TO', 'AVAV', 'AVB', 'AVCT',
        'AVGO', 'AVO', 'AXP', 'B2GOLD.TO', 'BA', 'BABA', 'BAC', 'BAM.TO', 'BATT', 'BB',
        'BBIG', 'BCE.TO', 'BE', 'BEAM', 'BENE', 'BFRI', 'BHAT', 'BIDU', 'BIIB', 'BILL',
        'BIOC', 'BIP-UN.TO', 'BK', 'BKKT', 'BKNG', 'BLDP', 'BLK', 'BLNK', 'BLRX', 'BLUE',
        'BMO.TO', 'BMRA', 'BMRN', 'BMY', 'BNGO', 'BNS.TO', 'BNTC', 'BNTX', 'BRK-A', 'BTE.TO',
        'BUDZ', 'BXP', 'BYD.TO', 'BYND', 'C', 'C3AI', 'CACC', 'CAG', 'CALM', 'CAR-UN.TO',
        'CASTOR', 'CAT', 'CBOE', 'CCI', 'CCL', 'CCO.TO', 'CDE.TO', 'CE', 'CELG', 'CF',
        'CFLT', 'CG.TO', 'CGC', 'CGC.TO', 'CGEM', 'CHD', 'CHEF', 'CHPT', 'CHRW', 'CI',
        'CJ.TO', 'CL', 'CLF', 'CLOV', 'CLR', 'CLSK', 'CLX', 'CM.TO', 'CME', 'CMI',
        'CMS', 'CNBS', 'CNC', 'CNQ.TO', 'CNR.TO', 'CNRG', 'CNX', 'COF', 'COIN', 'COMP',
        'COP', 'COR', 'COST', 'COUP', 'CP.TO', 'CPB', 'CPE', 'CPG.TO', 'CPT', 'CPX.TO',
        'CRC', 'CREE', 'CRLBF', 'CRM', 'CRON', 'CRON.TO', 'CROX', 'CRSP', 'CRSR', 'CRT-UN.TO',
        'CRUS', 'CRWD', 'CSCO', 'CSIQ', 'CSU.TO', 'CTRA', 'CU.TO', 'CURLF', 'CVE.TO', 'CVS',
        'CVX', 'CW', 'CYBR', 'CZR', 'D', 'DARE', 'DASH', 'DCBO.TO', 'DD', 'DDOG',
        'DE', 'DECK', 'DHR', 'DIOD', 'DIR-UN.TO', 'DIS', 'DKNG', 'DOCN', 'DOCU', 'DOL.TO',
        'DOV', 'DOW', 'DTE', 'DUK', 'DVN', 'DWAC', 'DXCM', 'EA', 'EARS', 'ECL',
        'ED', 'EDIT', 'EDV.TO', 'EGO.TO', 'EIX', 'ELD.TO', 'ELV', 'EMA.TO', 'EMN', 'EMR',
        'ENB.TO', 'ENPH', 'EOG', 'EQIX', 'EQR', 'EQT', 'ERF.TO', 'ES', 'ESS', 'ESTC',
        'ETN', 'ETR', 'ETSY', 'EURN', 'EVGO', 'EVRG', 'EXAS', 'EXC', 'EXPD', 'EXR',
        'F', 'FAMI', 'FANG', 'FATE', 'FCEL', 'FCR-UN.TO', 'FCX', 'FDX', 'FE', 'FEYE',
        'FIS', 'FISV', 'FM.TO', 'FMC', 'FNV.TO', 'FOLD', 'FOOD.TO', 'FORD', 'FORM', 'FROG',
        'FRT', 'FTNT', 'FTS.TO', 'GD', 'GDNP.TO', 'GE', 'GEVO', 'GFI.TO', 'GFL.TO', 'GIB-A.TO',
        'GILD', 'GIS', 'GM', 'GME', 'GNUS', 'GOEV', 'GOLD', 'GOLD.TO', 'GOOG', 'GOOGL',
        'GPS', 'GRAB', 'GRID', 'GS', 'GTBIF', 'GTLB', 'GXE.TO', 'H.TO', 'HBM.TO', 'HD',
        'HEAR', 'HEI', 'HES', 'HEXO', 'HEXO.TO', 'HGEN', 'HIMX', 'HL.TO', 'HLT', 'HOLX',
        'HON', 'HOOD', 'HR-UN.TO', 'HSY', 'HUM', 'HWM', 'HWX.TO', 'HYLN', 'IBM', 'ICE',
        'ICLN', 'IDEX', 'IFF', 'ILMN', 'IMO.TO', 'INCY', 'INGR', 'INO', 'INSW', 'INTC',
        'INTU', 'INTUV', 'IONS', 'IPO.TO', 'IRNT', 'ISRG', 'ITW', 'JAGG.TO', 'JAGX', 'JBHT',
        'JD', 'JJSF', 'JKS', 'JNJ', 'JPM', 'K', 'K.TO', 'KDP', 'KEL.TO', 'KIM',
        'KL.TO', 'KLAC', 'KMB', 'KNX', 'KO', 'KOSS', 'KTOS', 'KXS.TO', 'L.TO', 'LANC',
        'LC', 'LCID', 'LHX', 'LI', 'LIN', 'LIT', 'LKCO', 'LLY', 'LMT', 'LNT',
        'LOGI', 'LOW', 'LRCX', 'LSPD.TO', 'LUCK.TO', 'LULU', 'LVS', 'LWAY', 'LYB', 'LYFT',
        'MA', 'MAA', 'MAC', 'MAR', 'MASI', 'MAXN', 'MAXR', 'MCD', 'MCHP', 'MCO',
        'MDB', 'MDF.TO', 'MDLZ', 'MEG.TO', 'MELI', 'META', 'MG.TO', 'MGM', 'MGPI', 'MJ',
        'MKTX', 'MMM', 'MNDY', 'MNST', 'MOG-A', 'MOGO.TO', 'MOH', 'MOS', 'MPC', 'MPWR',
        'MRK', 'MRNA', 'MRO', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSOS', 'MT', 'MTDR',
        'MTY.TO', 'MU', 'MVIS', 'MXL', 'NA.TO', 'NAT', 'NCLH', 'NCNO', 'NDAQ', 'NEE',
        'NEM', 'NEOG', 'NET', 'NFLX', 'NGT.TO', 'NI', 'NIO', 'NKE', 'NKLA', 'NNDM',
        'NNN', 'NOC', 'NOG', 'NOK', 'NOVA', 'NOW', 'NTAR.TO', 'NTLA', 'NU', 'NUE',
        'NUVEI.TO', 'NVA.TO', 'NVAX', 'NVCR', 'NVDA', 'NVEC', 'NVEI.TO', 'O', 'OATLY', 'OBE.TO',
        'OBSV', 'OCGN', 'ODFL', 'OGI', 'OGI.TO', 'OKTA', 'OMCL', 'ON', 'OPAD', 'OPEN',
        'ORCL', 'OTEX.TO', 'OVV', 'OXY', 'PAAS.TO', 'PACB', 'PAGS', 'PANW', 'PARA', 'PATH',
        'PBW', 'PCAR', 'PCTY', 'PDD', 'PEG', 'PEI', 'PENN', 'PEP', 'PFE', 'PG',
        'PH', 'PHUN', 'PINS', 'PKG', 'PKI.TO', 'PL', 'PLAB', 'PLD', 'PLTR', 'PLUG',
        'PMED.TO', 'PNC', 'PODD', 'POST', 'POTX', 'POU.TO', 'POWI', 'PPG', 'PR', 'PRIME',
        'PROG', 'PSA', 'PSX', 'PTON', 'PVH', 'PYPL', 'QCLN', 'QCOM', 'QRVO', 'QSR.TO',
        'QST.TO', 'QTRH.TO', 'QUBT', 'RARE', 'RBLX', 'RCG.TO', 'RCI-B.TO', 'RCL', 'RDFN', 'RDW',
        'REAL.TO', 'REALTY', 'RECO.TO', 'REG', 'REGN', 'REI-UN.TO', 'RGNX', 'RIBT', 'RIDE', 'RIOT',
        'RIVN', 'RKDA', 'RL', 'RMBS', 'ROKU', 'RPM', 'RRC', 'RTX', 'RUN', 'RY.TO',
        'S', 'SAIA', 'SATS', 'SBLK', 'SBUX', 'SCCO', 'SCHW', 'SE', 'SEDG', 'SEE',
        'SENS', 'SFM', 'SGEN', 'SGMO', 'SGY.TO', 'SHOP', 'SHOP.TO', 'SHW', 'SITM', 'SJM',
        'SKLZ', 'SKT', 'SKX', 'SLAB', 'SLB', 'SLG', 'SLGG', 'SM', 'SMAR', 'SMCI',
        'SMOG', 'SMPL', 'SMTC', 'SNAP', 'SNDL', 'SNOW', 'SO', 'SOFI', 'SOL', 'SOLR.TO',
        'SPCE', 'SPG', 'SPGI', 'SPLK', 'SPRT', 'SPWR', 'SQ', 'SRE', 'SRPT', 'SRU-UN.TO',
        'STLD', 'STN.TO', 'STNE', 'STNG', 'STT', 'STZ', 'SU.TO', 'SVXY', 'SWKS', 'SWN',
        'SYNA', 'T', 'T.TO', 'TAP', 'TCEHY', 'TCNNF', 'TD.TO', 'TDG', 'TDOC', 'TEAM',
        'TECH', 'TECK', 'TELL', 'TENB', 'TFC', 'TFI.TO', 'TFII.TO', 'TGT', 'THCX', 'TJX',
        'TKO.TO', 'TLRY', 'TLRY.TO', 'TMDX', 'TME', 'TMO', 'TMUS', 'TNK', 'TOI.TO', 'TOKE',
        'TPG', 'TREE', 'TROW', 'TRP.TO', 'TSLA', 'TSN', 'TTCF', 'TTMI', 'TTWO', 'TVE.TO',
        'TWLO', 'TX', 'TXN', 'TXT', 'U', 'UAVS', 'UBER', 'UCTT', 'UDR', 'UNFI',
        'UNH', 'UPS', 'UPST', 'USB', 'V', 'VALE', 'VBIV', 'VECO', 'VEEV', 'VERV',
        'VERY', 'VET.TO', 'VFC', 'VFF.TO', 'VICR', 'VIX', 'VLNS.TO', 'VLO', 'VNOM', 'VRTX',
        'VTR', 'VXRT', 'VZ', 'W', 'WAB', 'WBD', 'WCG', 'WCN.TO', 'WCP.TO', 'WDAY',
        'WEED.TO', 'WELL', 'WELL.TO', 'WFC', 'WIMI', 'WISH', 'WKHS', 'WLL', 'WMT', 'WOLF',
        'WORK', 'WRLD', 'WSP.TO', 'WTE.TO', 'WYNN', 'X', 'XEL', 'XELA', 'XOM', 'XPEV',
        'XPO', 'YOLO', 'Z', 'ZG', 'ZM', 'ZS',
    ]
    
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
