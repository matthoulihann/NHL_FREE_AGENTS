import mysql.connector
import pandas as pd

# File path to your CSV
csv_file_path = '/data/Free_Agents_Data_with_Prev_Cap_Hit__.csv'

# Read the CSV using pandas
df = pd.read_csv(csv_file_path)

# Connect to your MySQL database
connection = mysql.connector.connect(
    host="localhost",
    user="mjhoulih",
    password="Buster3615!",
    database="nhl_free_agents"
)

cursor = connection.cursor()

# Insert data into the stats table row by row
for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO stats (
            player_name, position, age, yoe, prev_team, prev_aav, contract_type,
            toi_22_23, goals_22_23, a1_22_23, icf_22_23, ixg_22_23,
            giveaways_22_23, takeaways_22_23, gar_22_23, war_22_23, spar_22_23,
            toi_23_24, goals_23_24, a1_23_24, icf_23_24, ixg_23_24,
            giveaways_23_24, takeaways_23_24, gar_23_24, war_23_24, spar_23_24,
            toi_24_25, goals_24_25, a1_24_25, icf_24_25, ixg_24_25,
            giveaways_24_25, takeaways_24_25, gar_24_25, war_24_25, spar_24_25,
            `toi%_22_23`, `toi%_23_24`, `toi%_24_25`, prev_cap_hit_pct
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

# Commit changes and close the connection
connection.commit()
cursor.close()
connection.close()

print("CSV file successfully imported into the 'stats' table!")
