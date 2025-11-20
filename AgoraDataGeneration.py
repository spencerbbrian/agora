import os
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


OUT_DIR = "data"
BRANDS_PATH = "brands.csv"
PRODUCTS_PATH = "products.csv"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_DAILY_ORDERS = 1500
MAX_DAILY_ORDERS = 3000

QUANTITY_PATTERN = [1, 2, 3, 2, 4, 5, 1]

MIN_LINES_PER_ORDER = 1
MAX_LINES_PER_ORDER = 3

ORDER_RETURN_PROB = 0.15

DAYS_BACK = 60

TODAY_BASE = date.today()

city_to_region = {
    "Paris": "Île-de-France",
    "Boulogne-Billancourt": "Île-de-France",
    "Versailles": "Île-de-France",
    "Saint-Denis": "Île-de-France",
    "Nanterre": "Île-de-France",
    "Argenteuil": "Île-de-France",
    "Créteil": "Île-de-France",
    "Lyon": "Auvergne-Rhône-Alpes",
    "Grenoble": "Auvergne-Rhône-Alpes",
    "Clermont-Ferrand": "Auvergne-Rhône-Alpes",
    "Saint-Étienne": "Auvergne-Rhône-Alpes",
    "Annecy": "Auvergne-Rhône-Alpes",
    "Chambéry": "Auvergne-Rhône-Alpes",
    "Valence": "Auvergne-Rhône-Alpes",
    "Marseille": "Provence-Alpes-Côte d'Azur",
    "Nice": "Provence-Alpes-Côte d'Azur",
    "Toulon": "Provence-Alpes-Côte d'Azur",
    "Aix-en-Provence": "Provence-Alpes-Côte d'Azur",
    "Avignon": "Provence-Alpes-Côte d'Azur",
    "Cannes": "Provence-Alpes-Côte d'Azur",
    "Antibes": "Provence-Alpes-Côte d'Azur",
    "Toulouse": "Occitanie",
    "Montpellier": "Occitanie",
    "Nîmes": "Occitanie",
    "Perpignan": "Occitanie",
    "Béziers": "Occitanie",
    "Carcassonne": "Occitanie",
    "Albi": "Occitanie",
    "Lille": "Hauts-de-France",
    "Amiens": "Hauts-de-France",
    "Roubaix": "Hauts-de-France",
    "Tourcoing": "Hauts-de-France",
    "Dunkerque": "Hauts-de-France",
    "Calais": "Hauts-de-France",
    "Arras": "Hauts-de-France",
    "Bordeaux": "Nouvelle-Aquitaine",
    "Limoges": "Nouvelle-Aquitaine",
    "Poitiers": "Nouvelle-Aquitaine",
    "Pau": "Nouvelle-Aquitaine",
    "La Rochelle": "Nouvelle-Aquitaine",
    "Bayonne": "Nouvelle-Aquitaine",
    "Agen": "Nouvelle-Aquitaine",
    "Nantes": "Pays de la Loire",
    "Angers": "Pays de la Loire",
    "Le Mans": "Pays de la Loire",
    "Saint-Nazaire": "Pays de la Loire",
    "Cholet": "Pays de la Loire",
    "La Roche-sur-Yon": "Pays de la Loire",
    "Strasbourg": "Grand Est",
    "Reims": "Grand Est",
    "Metz": "Grand Est",
    "Nancy": "Grand Est",
    "Mulhouse": "Grand Est",
    "Colmar": "Grand Est",
    "Troyes": "Grand Est",
    "Rennes": "Bretagne",
    "Brest": "Bretagne",
    "Quimper": "Bretagne",
    "Vannes": "Bretagne",
    "Lorient": "Bretagne",
    "Le Havre": "Normandie",
    "Caen": "Normandie",
    "Rouen": "Normandie",
    "Cherbourg-en-Cotentin": "Normandie",
    "Évreux": "Normandie",
    "Dijon": "Bourgogne-Franche-Comté",
    "Besançon": "Bourgogne-Franche-Comté",
    "Nevers": "Bourgogne-Franche-Comté",
    "Chalon-sur-Saône": "Bourgogne-Franche-Comté",
    "Auxerre": "Bourgogne-Franche-Comté",
    "Orléans": "Centre-Val de Loire",
    "Tours": "Centre-Val de Loire",
    "Bourges": "Centre-Val de Loire",
    "Chartres": "Centre-Val de Loire",
    "Blois": "Centre-Val de Loire",
    "Ajaccio": "Corse",
    "Bastia": "Corse",
}

MASTER_WAREHOUSES = [
    {"id_warehouse": "WH01", "wh_name": "Warehouse 1", "city": "Paris", "region": "Île-de-France", "capacity": 120000},
    {"id_warehouse": "WH02", "wh_name": "Warehouse 2", "city": "Lyon", "region": "Auvergne-Rhône-Alpes", "capacity": 100000},
    {"id_warehouse": "WH03", "wh_name": "Warehouse 3", "city": "Marseille", "region": "Provence-Alpes-Côte d'Azur", "capacity": 90000},
    {"id_warehouse": "WH04", "wh_name": "Warehouse 4", "city": "Toulouse", "region": "Occitanie", "capacity": 80000},
    {"id_warehouse": "WH05", "id_warehouse": "WH05", "wh_name": "Warehouse 5", "city": "Lille", "region": "Hauts-de-France", "capacity": 70000},
    {"id_warehouse": "WH06", "wh_name": "Warehouse 6", "city": "Bordeaux", "region": "Nouvelle-Aquitaine", "capacity": 85000},
    {"id_warehouse": "WH07", "wh_name": "Warehouse 7", "city": "Nantes", "region": "Pays de la Loire", "capacity": 65000},
    {"id_warehouse": "WH08", "wh_name": "Warehouse 8", "city": "Strasbourg", "region": "Grand Est", "capacity": 60000},
    {"id_warehouse": "WH09", "wh_name": "Warehouse 9", "city": "Rennes", "region": "Bretagne", "capacity": 55000},
    {"id_warehouse": "WH10", "wh_name": "Warehouse 10", "city": "Le Havre", "region": "Normandie", "capacity": 50000},
    {"id_warehouse": "WH11", "wh_name": "Warehouse 11", "city": "Dijon", "region": "Bourgogne-Franche-Comté", "capacity": 50000},
    {"id_warehouse": "WH12", "wh_name": "Warehouse 12", "city": "Tours", "region": "Centre-Val de Loire", "capacity": 60000},
    {"id_warehouse": "WH13", "wh_name": "Warehouse 13", "city": "Nice", "region": "Provence-Alpes-Côte d'Azur", "capacity": 70000},
    {"id_warehouse": "WH14", "wh_name": "Warehouse 14", "city": "Montpellier", "region": "Occitanie", "capacity": 65000},
    {"id_warehouse": "WH15", "wh_name": "Warehouse 15", "city": "Grenoble", "region": "Auvergne-Rhône-Alpes", "capacity": 55000},
    {"id_warehouse": "WH16", "wh_name": "Warehouse 16", "city": "Reims", "region": "Grand Est", "capacity": 52000},
    {"id_warehouse": "WH17", "wh_name": "Warehouse 17", "city": "Angers", "region": "Pays de la Loire", "capacity": 48000},
    {"id_warehouse": "WH18", "wh_name": "Warehouse 18", "city": "Clermont-Ferrand", "region": "Auvergne-Rhône-Alpes", "capacity": 45000},
    {"id_warehouse": "WH19", "wh_name": "Warehouse 19", "city": "Nancy", "region": "Grand Est", "capacity": 42000},
    {"id_warehouse": "WH20", "wh_name": "Warehouse 20", "city": "Brest", "region": "Bretagne", "capacity": 40000},
]

for wh in MASTER_WAREHOUSES:
    if city_to_region.get(wh["city"]) != wh["region"]:
        raise ValueError(f"Region mismatch for {wh['id_warehouse']} ({wh['city']})")

MASTER_SUPPLIERS = [
    {"id_suppliers": i + 1, "supplier_name": name}
    for i, name in enumerate([
        "Azur Supply Co", "HexaTrade", "ParfumSource", "Gallic Logistics", "Riviera Imports",
        "MontBlanc Wholesale", "Seine Distribution", "Cassis Partners", "Lumière Supply", "Côte d'Or Trading",
        "Atlantic Sourcing", "Occitanie Vendors", "Alpine Trade", "Grand Est Freight", "Bretagne Commerce",
        "Normandie Goods", "Val de Loire Supplies", "Aquitaine Distribution", "PACA Wholesale", "Hauts-de-France Logistic"
    ])
]

RETURN_REASONS = [
    "Damaged", "Damaged", "Damaged",
    "Customer Refused",
    "Wrong Item", "Wrong Item",
    "Not As Described", "Not As Described",
    "Late Delivery",
    "Leaking Bottle",
    "Broken Seal",
    "Product Defect", "Product Defect"
]


def _path(name: str) -> Path:
    return Path(OUT_DIR) / f"{name}.csv"


def ensure_table(name: str, columns: List[str]) -> pd.DataFrame:
    p = _path(name)
    if p.exists():
        df = pd.read_csv(p)
        for col in columns:
            if col not in df.columns:
                df[col] = None
        return df[columns]
    df = pd.DataFrame(columns=columns)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return df


def save(name: str, df: pd.DataFrame):
    df.to_csv(_path(name), index=False, encoding="utf-8-sig")


def safe_concat(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return existing
    if existing is None or existing.empty:
        return new_df
    return pd.concat([existing, new_df], ignore_index=True)


def to_id_dim_date(d) -> int:
    if isinstance(d, datetime):
        d = d.date()
    return int(d.strftime("%Y%m%d"))


def ensure_refs(initial_date: date):
    today_sk = int(initial_date.strftime("%Y%m%d"))

    if _path("dim_brand").exists():
        _ = pd.read_csv(_path("dim_brand"))
    else:
        src_brands = pd.read_csv(BRANDS_PATH)

        if "brand_id" in src_brands.columns and "id_brand" not in src_brands.columns:
            src_brands = src_brands.rename(columns={"brand_id": "id_brand"})
        if "Brand" in src_brands.columns and "brand_name" not in src_brands.columns:
            src_brands = src_brands.rename(columns={"Brand": "brand_name"})
        if "id_brand" not in src_brands.columns:
            src_brands["id_brand"] = range(1, len(src_brands) + 1)

        expected_cols = {"id_brand", "brand_name"}
        if not expected_cols.issubset(src_brands.columns):
            raise ValueError(f"Le fichier {BRANDS_PATH} doit contenir au minimum les colonnes {expected_cols}")
        save("dim_brand", src_brands)

    if _path("dim_products").exists():
        _ = pd.read_csv(_path("dim_products"))
    else:
        products = pd.read_csv(PRODUCTS_PATH).copy()

        if "id_products" not in products.columns:
            products["id_products"] = range(1, len(products) + 1)

        if "id_brand" not in products.columns:
            if "brand_id" in products.columns:
                products["id_brand"] = products["brand_id"]
            elif "Brand" in products.columns:
                brands_dim = pd.read_csv(_path("dim_brand"))
                products = products.merge(
                    brands_dim[["id_brand", "brand_name"]],
                    left_on="Brand",
                    right_on="brand_name",
                    how="left"
                )
            else:
                products["id_brand"] = 1

        AED_TO_EUR = 0.25

        if "unit_price" not in products.columns:
            if "Price" in products.columns:
                products["Price"] = pd.to_numeric(products["Price"], errors="coerce").fillna(0)
                products["Price"] = products["Price"] * AED_TO_EUR
            else:
                products["Price"] = 99.0

        save("dim_products", products)

    if not _path("dim_warehouse").exists():
        save("dim_warehouse", pd.DataFrame(MASTER_WAREHOUSES))

    if not _path("dim_suppliers").exists():
        save("dim_suppliers", pd.DataFrame(MASTER_SUPPLIERS))

    if not _path("dim_store").exists():
        wh = pd.read_csv(_path("dim_warehouse"))
        rows = []
        for i, (_, w) in enumerate(wh.iterrows(), start=1):
            rows.append({"id_store": f"ST{i:03d}", "city": w.city, "region": w.region})
            rows.append({"id_store": f"ST{i+100:03d}", "city": w.city, "region": w.region})
        stores = pd.DataFrame(rows)

        def pick_sup(row):
            same_city = wh[wh["city"] == row["city"]]
            if not same_city.empty:
                return same_city.iloc[0]["id_warehouse"]
            same_reg = wh[wh["region"] == row["region"]]
            if not same_reg.empty:
                return same_reg.iloc[0]["id_warehouse"]
            return "WH01"

        stores["id_warehouse_supplying"] = stores.apply(pick_sup, axis=1)
        save("dim_store", stores)

    stocks_cols = [
        "id_products",
        "id_warehouse",
        "stock",
        "min_stock",
        "id_dim_date_last_stocked",
        "id_dim_date_last_updated",
    ]
    if (not _path("fct_stocks").exists()) or pd.read_csv(_path("fct_stocks")).empty:
        products = pd.read_csv(_path("dim_products"))
        warehouses = pd.read_csv(_path("dim_warehouse"))

        rows = []
        rng = np.random.default_rng(42)
        for _, p in products.iterrows():
            for _, w in warehouses.iterrows():
                rows.append({
                    "id_products": int(p["id_products"]),
                    "id_warehouse": w["id_warehouse"],
                    "stock": int(rng.integers(200, 1000)),
                    "min_stock": 50,
                    "id_dim_date_last_stocked": today_sk,
                    "id_dim_date_last_updated": today_sk,
                })
        stocks_df = pd.DataFrame(rows, columns=stocks_cols)
        save("fct_stocks", stocks_df)


def build_dim_date_global():
    start_date = date(2025, 8, 1)
    end_date = TODAY_BASE + timedelta(days=2 * 365)
    dates = pd.date_range(start_date, end_date, freq="D")
    rows = []
    for dt in dates:
        d = dt.date()
        rows.append({
            "id_dim_date": int(d.strftime("%Y%m%d")),
            "date": d.strftime("%Y-%m-%d"),
            "year": d.year,
            "quarter": (d.month - 1) // 3 + 1,
            "month": d.month,
            "month_name": dt.strftime("%B"),
            "day": d.day,
            "day_of_week": dt.isoweekday(),
            "day_name": dt.strftime("%A"),
            "is_weekend": dt.isoweekday() >= 6,
            "week_of_year": int(dt.strftime("%V")),
        })
    dim_date = pd.DataFrame(rows)
    save("dim_date", dim_date)


def compute_today_target_uniform(ref_date: date) -> int:
    seed = int(ref_date.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    base = int(rng.integers(MIN_DAILY_ORDERS, MAX_DAILY_ORDERS + 1))

    dow = ref_date.isoweekday()
    weekday_factor = {
        1: 0.85,  # Lundi
        2: 0.95,  # Mardi
        3: 1.00,  # Mercredi
        4: 1.05,  # Jeudi
        5: 1.20,  # Vendredi
        6: 1.25,  # Samedi
        7: 1.10,  # Dimanche
    }.get(dow, 1.0)

    val = int(round(base * weekday_factor))
    val = max(MIN_DAILY_ORDERS, min(MAX_DAILY_ORDERS, val))
    return val


def restock_low_stocks(fct_stocks: pd.DataFrame, current_date: date, rng: np.random.Generator) -> pd.DataFrame:
    if fct_stocks is None or fct_stocks.empty:
        return fct_stocks

    df = fct_stocks.copy()
    date_key = to_id_dim_date(current_date)

    low_mask = df["stock"] < df["min_stock"]
    if not low_mask.any():
        return df

    for idx in df.index[low_mask]:
        current_stock = int(df.at[idx, "stock"])
        min_stock = int(df.at[idx, "min_stock"])

        target_stock = max(2 * min_stock, current_stock + int(rng.integers(100, 401)))
        add_qty = target_stock - current_stock
        if add_qty <= 0:
            continue

        df.at[idx, "stock"] = current_stock + add_qty
        df.at[idx, "id_dim_date_last_stocked"] = date_key
        df.at[idx, "id_dim_date_last_updated"] = date_key

    return df


def run_for_date(current_date: date):
    products = pd.read_csv(_path("dim_products"))
    warehouses = pd.read_csv(_path("dim_warehouse"))
    stores = pd.read_csv(_path("dim_store"))
    suppliers = pd.read_csv(_path("dim_suppliers"))

    fct_orders = ensure_table(
        "fct_orders",
        ["id_orders", "id_store", "id_suppliers"]
    )

    fct_order_lines = ensure_table(
        "fct_order_lines",
        ["id_order_line", "id_orders", "id_products",
         "quantity", "requested_qty", "returned_flag"]
    )

    fct_order_log = ensure_table(
        "fct_order_log",
        [
            "id_order_log",
            "id_orders",
            "ts_ordered",
            "ts_shipped_warehouse",
            "ts_shipped_store",
            "ts_delivered",
            "ts_returned",
            "ts_last_update",
            "id_dim_date_ordered",
            "id_dim_date_shipped_wh",
            "id_dim_date_shipped_store",
            "id_dim_date_delivered",
            "id_dim_date_returned",
            "id_dim_date_last_update",
        ]
    )

    fct_transport_log = ensure_table(
        "fct_transport_log",
        [
            "id_transport_log",
            "id_warehouse",
            "id_store",
            "id_suppliers",
            "journey_status",
            "ts_stock_shipped",
            "ts_stock_received",
            "ts_last_update",
            "id_dim_date_stock_shipped",
            "id_dim_date_stock_received",
            "id_dim_date_last_update",
        ]
    )

    fct_returns = ensure_table(
        "fct_returns",
        ["id_returns", "id_orders", "id_dim_date_returned", "reason"]
    )

    fct_stocks = ensure_table(
        "fct_stocks",
        ["id_products", "id_warehouse", "stock", "min_stock",
         "id_dim_date_last_stocked", "id_dim_date_last_updated"]
    )

    today_key = int(current_date.strftime("%Y%m%d"))
    if not fct_order_log.empty:
        already = int((fct_order_log["id_dim_date_ordered"] == today_key).sum())
    else:
        already = 0

    target = compute_today_target_uniform(current_date)
    to_add = max(0, target - already)
    if to_add == 0:
        print(f"[{current_date}] Target: {target} | Already: {already} | Added: 0")
        return

    rng = np.random.default_rng()

    fct_stocks = restock_low_stocks(fct_stocks, current_date, rng)

    next_order_id = int(fct_orders["id_orders"].max()) + 1 if not fct_orders.empty else 1
    next_order_line_id = int(fct_order_lines["id_order_line"].max()) + 1 if not fct_order_lines.empty else 1
    next_return_id = int(fct_returns["id_returns"].max()) + 1 if not fct_returns.empty else 1

    prod_ids = products["id_products"].astype(int).to_numpy()
    if "unit_price" in products.columns:
        pop_weights = products["unit_price"].astype(float).clip(lower=1.0)
    else:
        pop_weights = np.ones(len(products))
    probs = pop_weights / pop_weights.sum()

    store_ids = stores["id_store"].tolist()
    supp_ids = suppliers["id_suppliers"].astype(int).tolist()

    def rr(seq, start, n):
        seq = list(seq)
        idx = np.arange(start, start + n) % len(seq)
        return [seq[i] for i in idx]

    start_idx = next_order_id - 1

    base_ts = datetime.combine(current_date, datetime.min.time())
    all_slots_sec = np.linspace(0, 86399, target, dtype=int)
    new_slots_sec = all_slots_sec[already:target]
    dt_ordered = [base_ts + timedelta(seconds=int(s)) for s in new_slots_sec]

    def compute_shipping_chain(ts_ord, rng_local: np.random.Generator):
        ts_wh = ts_ord + timedelta(days=1)

        extra_days = 0
        r = rng_local.random()
        if r < 0.10:
            extra_days = 1
        elif r < 0.12:
            extra_days = 2

        ts_store = ts_wh + timedelta(days=2 + extra_days)
        ts_del = ts_store + timedelta(days=1)
        return ts_wh, ts_store, ts_del, extra_days

    wh_by_city = warehouses.set_index("city")["id_warehouse"].to_dict()
    wh_by_region = warehouses.groupby("region")["id_warehouse"].first().to_dict()

    def supplying_for_store(scode: str) -> str:
        row = stores.loc[stores["id_store"] == scode].iloc[0]
        return wh_by_city.get(row["city"], wh_by_region.get(row["region"], "WH01"))

    new_orders = []
    new_lines = []
    new_olog = []
    new_tlog = []
    new_ret = []

    if not fct_stocks.empty:
        fct_stocks["id_products"] = fct_stocks["id_products"].astype(int)

    store_sel = rr(store_ids, start_idx, to_add)
    supp_sel = rr(supp_ids, start_idx, to_add)

    for idx in range(to_add):
        oid = next_order_id + idx
        scode = store_sel[idx]
        sup = int(supp_sel[idx])

        ts_ord = dt_ordered[idx]
        ts_wh, ts_store, ts_del, extra_days = compute_shipping_chain(ts_ord, rng)

        id_date_ordered = to_id_dim_date(ts_ord)
        id_date_shipped_wh = to_id_dim_date(ts_wh)
        id_date_shipped_store = to_id_dim_date(ts_store)
        id_date_delivered = to_id_dim_date(ts_del)

        wh_code = supplying_for_store(scode)

        is_order_returned = rng.random() < ORDER_RETURN_PROB
        has_allocated_qty = False

        new_orders.append({
            "id_orders": oid,
            "id_store": scode,
            "id_suppliers": sup,
        })

        n_lines = rng.integers(MIN_LINES_PER_ORDER, MAX_LINES_PER_ORDER + 1)
        for _ in range(n_lines):
            line_id = next_order_line_id
            next_order_line_id += 1

            pid = int(rng.choice(prod_ids, p=probs))
            qty_values = [1, 2, 3, 4, 5]
            qty_probs = [0.65, 0.20, 0.10, 0.04, 0.01]

            qty = int(rng.choice(qty_values, p=qty_probs))

            mask = (fct_stocks["id_products"] == pid) & (fct_stocks["id_warehouse"] == wh_code)

            if not mask.any():
                new_row = {
                    "id_products": pid,
                    "id_warehouse": wh_code,
                    "stock": 0,
                    "min_stock": 0,
                    "id_dim_date_last_stocked": id_date_shipped_wh,
                    "id_dim_date_last_updated": id_date_shipped_wh,
                }
                fct_stocks = pd.concat([fct_stocks, pd.DataFrame([new_row])], ignore_index=True)
                mask = (fct_stocks["id_products"] == pid) & (fct_stocks["id_warehouse"] == wh_code)

            available = int(fct_stocks.loc[mask, "stock"].iloc[0])
            alloc_qty = min(qty, available)

            fct_stocks.loc[mask, "stock"] = available - alloc_qty
            if alloc_qty > 0:
                has_allocated_qty = True
                fct_stocks.loc[mask, "id_dim_date_last_stocked"] = id_date_shipped_wh
                fct_stocks.loc[mask, "id_dim_date_last_updated"] = id_date_shipped_wh

            line_returned_flag = bool(is_order_returned and alloc_qty > 0)

            new_lines.append({
                "id_order_line": line_id,
                "id_orders": oid,
                "id_products": pid,
                "quantity": alloc_qty,
                "requested_qty": qty,
                "returned_flag": line_returned_flag,
            })

        ts_ret = None
        id_date_returned = None
        id_date_last_update = id_date_delivered
        ts_last_update_str = ts_del.strftime("%H:%M:%S")

        if is_order_returned and has_allocated_qty:
            ts_ret = ts_del
            id_date_returned = to_id_dim_date(ts_ret)
            id_date_last_update = id_date_returned
            ts_last_update_str = ts_ret.strftime("%H:%M:%S")

            reason = rng.choice(RETURN_REASONS)
            new_ret.append({
                "id_returns": next_return_id,
                "id_orders": oid,
                "id_dim_date_returned": id_date_returned,
                "reason": reason,
            })
            next_return_id += 1

        new_olog.append({
            "id_order_log": oid,
            "id_orders": oid,
            "ts_ordered": ts_ord.strftime("%H:%M:%S"),
            "ts_shipped_warehouse": ts_wh.strftime("%H:%M:%S"),
            "ts_shipped_store": ts_store.strftime("%H:%M:%S"),
            "ts_delivered": ts_del.strftime("%H:%M:%S"),
            "ts_returned": ts_ret.strftime("%H:%M:%S") if ts_ret is not None else None,
            "ts_last_update": ts_last_update_str,
            "id_dim_date_ordered": id_date_ordered,
            "id_dim_date_shipped_wh": id_date_shipped_wh,
            "id_dim_date_shipped_store": id_date_shipped_store,
            "id_dim_date_delivered": id_date_delivered,
            "id_dim_date_returned": id_date_returned,
            "id_dim_date_last_update": id_date_last_update,
        })

        status = "On-Time" if extra_days == 0 else "Late"
        new_tlog.append({
            "id_transport_log": oid,
            "id_warehouse": wh_code,
            "id_store": scode,
            "id_suppliers": sup,
            "journey_status": status,
            "ts_stock_shipped": ts_wh.strftime("%H:%M:%S"),
            "ts_stock_received": ts_del.strftime("%H:%M:%S"),
            "ts_last_update": ts_del.strftime("%H:%M:%S"),
            "id_dim_date_stock_shipped": id_date_shipped_wh,
            "id_dim_date_stock_received": id_date_delivered,
            "id_dim_date_last_update": id_date_last_update,
        })

    fct_orders = safe_concat(fct_orders, pd.DataFrame(new_orders))
    fct_order_lines = safe_concat(fct_order_lines, pd.DataFrame(new_lines))
    fct_order_log = safe_concat(fct_order_log, pd.DataFrame(new_olog))
    fct_transport_log = safe_concat(fct_transport_log, pd.DataFrame(new_tlog))
    fct_returns = safe_concat(fct_returns, pd.DataFrame(new_ret))

    save("fct_orders", fct_orders)
    save("fct_order_lines", fct_order_lines)
    save("fct_order_log", fct_order_log)
    save("fct_transport_log", fct_transport_log)
    save("fct_returns", fct_returns)
    save("fct_stocks", fct_stocks)

    print(f"[{current_date}] Target: {target} | Already: {already} | Added: {to_add}")


ensure_refs(TODAY_BASE)
build_dim_date_global()

current_day = TODAY_BASE
run_for_date(current_day)

folder = Path(OUT_DIR)
output_folder = folder / "xlsx"
output_folder.mkdir(exist_ok=True)
DAYS_BACK -= 1

for csv_file in folder.glob("*.csv"):
    xlsx_file = output_folder / (csv_file.stem + ".xlsx")
    try:
        df = pd.read_csv(csv_file)
        df.to_excel(xlsx_file, index=False, engine="openpyxl")
        print(f"Converted: {csv_file.name} -> {xlsx_file.name}")
    except Exception as e:
        print(f"Error converting {csv_file.name}: {e}")


