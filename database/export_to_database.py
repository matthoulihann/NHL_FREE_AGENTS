import mysql.connector
import pandas as pd

# MySQL Connection Setup
conn = mysql.connector.connect(
    host="yamanote.proxy.rlwy.net",  # or your Railway DB host
    port=14835,  # Railway assigns a custom port
    user="root",  # or your Railway username
    password="HzlDzyssSbtiZnjbwQdBbEiWBtnJtvYg",  # your Railway DB password
    database="railway",  # or whatever your Railway database is named
    ssl_disabled=True  # Use this if you hit SSL errors
)
cursor = conn.cursor()

# Load processed data
df = pd.read_csv("/results/player_value_assessment.csv")

# Rename columns to match MySQL schema
df = df.rename(columns={
    "Predicted AAV": "aav",
    "Final_Predicted_GAR_25_26": "projected_gar_25_26",
    "Fair_Market_Value": "market_value",
    "Value_Rating": "contract_value_score",
    "Value_Score": "value_per_gar",
    "Value_Category": "value_category",
    "Final_Contract_Term": "contract_term",
    "age": "age"
})

# Ensure correct data types
df["aav"] = df["aav"].astype(float)
df["projected_gar_25_26"] = df["projected_gar_25_26"].astype(float)
df["market_value"] = df["market_value"].astype(float)
df["contract_value_score"] = df["contract_value_score"].astype(float)
df["value_per_gar"] = df["value_per_gar"].astype(int)
df["contract_term"] = df["contract_term"].astype(int)
df["age"] = df["age"].astype(int)


def insert_data():
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO projected_contracts (
                player_name, aav, projected_gar_25_26, market_value, 
                contract_value_score, value_per_gar, value_category, contract_term, age
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                aav = VALUES(aav),
                projected_gar_25_26 = VALUES(projected_gar_25_26),
                market_value = VALUES(market_value),
                contract_value_score = VALUES(contract_value_score),
                value_per_gar = VALUES(value_per_gar),
                value_category = VALUES(value_category),
                contract_term = VALUES(contract_term),
                age = VALUES(age);
        """, (
            row['player_name'], row['aav'], row['projected_gar_25_26'], row['market_value'],
            row['contract_value_score'], row['value_per_gar'], row['value_category'],
            row['contract_term'], row['age']
        ))

    conn.commit()
    print("Data successfully inserted into MySQL database.")


# Run the insertion function
insert_data()

# Close the connection
cursor.close()
conn.close()
