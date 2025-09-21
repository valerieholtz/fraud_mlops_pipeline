# data/init_db.py
import pandas as pd
import sqlite3

# Load CSV once
df = pd.read_csv("data/fraud_transactions.csv")

# Create SQLite database
conn = sqlite3.connect("data/fraud.db")
df.to_sql("transactions", conn, if_exists="replace", index=False)
conn.close()

print("SQLite DB created at data/fraud.db with table 'transactions'")