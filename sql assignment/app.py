import pandas as pd
from sqlalchemy import create_engine

# Connect to Postgres
engine = create_engine("postgresql+psycopg2://user:7875010511@localhost:5432/superstore_db")

# Read Excel file
df = pd.read_excel("Sample - Superstore - Training (1).xlsx")

# Upload to Postgres
df.to_sql("superstore", engine, if_exists="replace", index=False)
