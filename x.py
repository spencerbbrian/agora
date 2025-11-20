import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, Any, List, Tuple
from pathlib import Path
import os

# --- Configuration (Based on your setup) ---
SNOWFLAKE_CONFIG: Dict[str, Any] = {
    "user": "SPENCERBBRIAN",
    "password": "%40Spencerbbrian1508", 
    "account": "MDEZLRF-VVB30479", 
    "warehouse": "COMPUTE_WH",
    "database": "AGORA",
    "schema": "DATASETS",
}

# Directory where your generated CSV files are stored
CSV_DIR = 'data' 

# --- List of all CSV files and their corresponding Snowflake table names ---
# Note: Snowflake table names are assumed to be lowercase
TABLE_MAP: List[Tuple[str, str]] = [
    # Master Data (Typically replaced completely, so 'replace' is fine for masters)
    ("dim_brand", "brands"),
    ("dim_products", "products"),
    ("dim_warehouse", "warehouses"),
    ("dim_store", "stores"),
    ("dim_suppliers", "suppliers"),
    
    # Fact/Transactional Data (Requires incremental strategy: 'append')
    ("fct_stocks", "stocks"),         # Stock is effectively REPLACED with the current state
    ("fct_orders", "orders"),         # APPEND for new orders
    ("fct_order_lines", "order_lines"), # APPEND for new order lines
    ("fct_order_log", "order_log"),   # APPEND for new log entries
    ("fct_transport_log", "transport_log"), # APPEND
    ("fct_returns", "returns"),       # APPEND
]

# --- Connection Helper ---
def connect_to_snowflake():
    config = SNOWFLAKE_CONFIG
    conn_string = (
        f"snowflake://{config['user']}:{config['password']}@{config['account']}/"
        f"{config['database']}/{config['schema']}?warehouse={config['warehouse']}"
    )
    return create_engine(conn_string)

def load_csv_to_snowflake(csv_filename: str, table_name: str):
    """
    Reads a CSV, determines the necessary SQL action (append/replace), and loads it.
    """
    engine = connect_to_snowflake()
    config = SNOWFLAKE_CONFIG
    file_path = Path(CSV_DIR) / f"{csv_filename}.csv"
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}. Skipping load for {table_name}.")
        return

    # Determine action: replace for dimension masters, append for facts/logs
    if table_name in ['brands', 'products', 'warehouses', 'stores', 'suppliers', 'stocks']:
        # Masters and the current state stocks table are usually fully replaced
        action = 'replace'
    else:
        # Orders, Logs, and Returns are appended incrementally
        action = 'append'

    try:
        print(f"➡️ Processing {csv_filename} ({action.upper()})...")
        
        # Read the file
        df = pd.read_csv(file_path)
        
        # Load to Snowflake
        df.to_sql(
            table_name.lower(), 
            con=engine,
            if_exists=action,
            index=False,
            chunksize=16000,
            schema=config['schema']
        )
        print(f"✅ Loaded {len(df)} records into {table_name}.")

    except Exception as e:
        print(f"\n❌ FAILED to load {table_name}.")
        print(f"Error: {e}")
    finally:
        engine.dispose()

# --- Main Execution ---

if __name__ == '__main__':
    print("--- Starting CSV Batch Loader ---")
    
    for csv_file, table_name in TABLE_MAP:
        load_csv_to_snowflake(csv_file, table_name)
        
    print("\n✅ Batch loading process complete.")