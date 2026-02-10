
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

def calculate_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw transactions and calculates the context vector.
    Used by Zone 1 (Training) to prepare data for the Bandit.
    
    Logic matches '03_feature_engineering.ipynb'.
    """
    # 1. Safety Copy
    df = df.copy()
    
    # 2. Robust Date Conversion (FIXED for StringDtype compatibility)
    if not is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # Drop rows where date conversion failed (if any)
    df = df.dropna(subset=['InvoiceDate'])

    # ------------------------------
    # 1. Aggregate Spending Metrics
    # ------------------------------
    # Using named aggregation for cleaner column names
    features = df.groupby("Customer ID").agg(
        total_spend=("TransactionValue", "sum"),
        avg_transaction_value=("TransactionValue", "mean"),
        transaction_count=("TransactionValue", "count"),
        quantity_mean=("Quantity", "mean"),
        quantity_std=("Quantity", "std"),
    )
    
    # Fix NaN std for users with only 1 transaction
    features["quantity_std"] = features["quantity_std"].fillna(0.0)

    # ------------------------------
    # 2. Return Behaviour
    # ------------------------------
    if "IsReturn" in df.columns:
        features["return_rate"] = df.groupby("Customer ID")["IsReturn"].mean()
    else:
        # Fallback if raw data doesn't have the flag pre-calculated
        # Assumes negative Quantity implies a return (Standard Retail Logic)
        features["return_rate"] = df.groupby("Customer ID")["Quantity"].apply(lambda x: (x < 0).mean())

    # ------------------------------
    # 3. Spending Volatility
    # ------------------------------
    # Improved: Use .dt.date instead of string slicing for safety
    df['day'] = df['InvoiceDate'].dt.date
    
    # Calculate daily spend sum per customer
    daily_spend = df.groupby(["Customer ID", "day"])["TransactionValue"].sum()
    
    # Calculate Std Dev of daily spend
    vol = daily_spend.groupby("Customer ID").std().fillna(0.0)
    features["spending_volatility"] = vol

    # ------------------------------
    # 4. Final Cleanup
    # ------------------------------
    # Ensure indices match and fill missing values
    return features.fillna(0.0)