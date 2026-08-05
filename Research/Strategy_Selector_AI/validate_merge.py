import pandas as pd
import logging
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MergeValidator")

def diagnose_merge():
    logger.info("=== STARTING MERGE DIAGNOSTICS ===")
    
    # 1. Load datasets using paths defined in config
    dna_path = getattr(config, 'MARKET_DNA_PATH', 'market_dna.csv')
    regime_path = getattr(config, 'MARKET_REGIMES_PATH', 'market_regimes.csv')
    
    logger.info(f"Loading Market DNA from: {dna_path}")
    logger.info(f"Loading Market Regimes from: {regime_path}")
    
    dna_df = pd.read_csv(dna_path)
    regime_df = pd.read_csv(regime_path)
    
    # 2. Print required metrics
    print(f"\nMarket DNA rows: {len(dna_df)}")
    print(f"Market Regime rows: {len(regime_df)}")
    
    # Identify date/merge key columns
    date_cols_dna = [c for c in dna_df.columns if 'date' in c.lower() or 'time' in c.lower()]
    date_cols_regime = [c for c in regime_df.columns if 'date' in c.lower() or 'time' in c.lower()]
    
    key_dna = date_cols_dna[0] if date_cols_dna else dna_df.columns[0]
    key_regime = date_cols_regime[0] if date_cols_regime else regime_df.columns[0]
    
    print(f"Merge key names -> DNA: '{key_dna}', Regime: '{key_regime}'")
    print(f"Merge data types -> DNA: {dna_df[key_dna].dtype}, Regime: {regime_df[key_regime].dtype}")
    
    print("\n--- First 20 Merge Keys (DNA) ---")
    print(dna_df[key_dna].head(20).tolist())
    
    print("\n--- First 20 Merge Keys (Regime) ---")
    print(regime_df[key_regime].head(20).tolist())
    
    # 3. Standardize and check formatting/mismatches
    dna_work = dna_df.copy()
    regime_work = regime_df.copy()
    
    dna_work['clean_key'] = pd.to_datetime(dna_work[key_dna], errors='coerce')
    regime_work['clean_key'] = pd.to_datetime(regime_work[key_regime], errors='coerce')
    
    missing_dna_keys = dna_work['clean_key'].isna().sum()
    missing_regime_keys = regime_work['clean_key'].isna().sum()
    
    print(f"\nRows with missing keys (DNA): {missing_dna_keys}")
    print(f"Rows with missing keys (Regime): {missing_regime_keys}")
    print(f"Duplicate keys in DNA: {dna_work['clean_key'].duplicated().sum()}")
    print(f"Duplicate keys in Regime: {regime_work['clean_key'].duplicated().sum()}")
    
    # 4. Perform joins
    inner_merged = pd.merge(dna_work, regime_work, left_on='clean_key', right_on='clean_key', how='inner')
    left_merged = pd.merge(dna_work, regime_work, left_on='clean_key', right_on='clean_key', how='left')
    outer_merged = pd.merge(dna_work, regime_work, left_on='clean_key', right_on='clean_key', how='outer')
    
    print(f"\nRows after inner merge: {len(inner_merged)}")
    print(f"Rows after left merge: {len(left_merged)}")
    print(f"Rows after outer merge: {len(outer_merged)}")
    
    rows_dropped = len(dna_work) - len(inner_merged)
    print(f"Rows dropped: {rows_dropped}")
    
    # 5. Generate Merge_Validation_Report.csv
    all_dates = pd.Series(pd.concat([dna_work['clean_key'], regime_work['clean_key']]).dropna().unique()).sort_values()
    
    report_rows = []
    for d in all_dates:
        in_dna = d in dna_work['clean_key'].values
        in_regime = d in regime_work['clean_key'].values
        merged = in_dna and in_regime
        
        reason = ""
        if not merged:
            if not in_dna: 
                reason = "Missing in Market DNA"
            elif not in_regime: 
                reason = "Missing in Market Regime"
            else: 
                reason = "Key Mismatch / Formatting Error"
                
        report_rows.append({
            "Date": str(d.date()) if pd.notnull(d) else str(d),
            "Exists in DNA": in_dna,
            "Exists in Regime": in_regime,
            "Merged": merged,
            "Reason Missing": reason
        })
        
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv("Merge_Validation_Report.csv", index=False)
    print("\nGenerated Merge_Validation_Report.csv successfully.")

if __name__ == "__main__":
    diagnose_merge()
