import pandas as pd
import mysql.connector

# Load your updated CSV
df = pd.read_csv("/data/Updated_Free_Agent_GAR_24-25_with_All_GP.csv")

# Connect to Railway MySQL
conn = mysql.connector.connect(
    host="yamanote.proxy.rlwy.net",
    port=14835,
    user="root",
    password="HzlDzyssSbtiZnjbwQdBbEiWBtnJtvYg",
    database="railway",
    ssl_disabled=True
)
cursor = conn.cursor()

# STEP 1: Get player_id from stats table using player_name
cursor.execute("SELECT player_id, player_name FROM stats")
id_map = cursor.fetchall()
id_df = pd.DataFrame(id_map, columns=["player_id", "player_name"])

# Standardize names for joining
df['player_name_clean'] = df['player_name'].str.lower().str.strip()
id_df['player_name_clean'] = id_df['player_name'].str.lower().str.strip()

# Merge player_id into your DataFrame
df = df.merge(id_df[['player_id', 'player_name_clean']], on='player_name_clean', how='left')
df.drop(columns=['player_name_clean'], inplace=True)

# STEP 2: Update GP columns using player_id
update_query = """
    UPDATE stats SET
        gp_22_23 = %s,
        gp_23_24 = %s,
        gp_24_25 = %s
    WHERE player_id = %s
"""

for _, row in df.iterrows():
    if pd.notnull(row["player_id"]):
        values = [
            row.get("gp_22_23", None),
            row.get("gp_23_24", None),
            row.get("gp_24_25", None),
            int(row["player_id"])
        ]
        cursor.execute(update_query, values)

conn.commit()
cursor.close()
conn.close()

print("✅ GP columns successfully updated in the stats table.")
