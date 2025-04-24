import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_path = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/models/results/predicted_contract_term.csv"
df = pd.read_csv(file_path)

target = "prev_cap_hit_pct"
df = df.dropna(subset=[target])

# Fill missing values for young players
performance_features = [
    "gar_22_23", "war_22_23", "toi_22_23",
    "gar_23_24", "war_23_24", "toi_23_24",
    "gar_24_25", "war_24_25", "toi_24_25"
]
df[performance_features] = df[performance_features].fillna(0)

df["prev_aav"] = df["prev_aav"].fillna(df["prev_aav"].median())  # Median prior contract value
df["yoe"] = df["yoe"].fillna(0)  # Years of experience: 0 for rookies

# Apply weighted calculations
df["weighted_gar_24_25"] = df["gar_24_25"] * 1.5
df["weighted_war_24_25"] = df["war_24_25"] * 1.5
df["weighted_toi_24_25"] = df["toi_24_25"] * 1.5

df["weighted_gar_23_24"] = df["gar_23_24"] * 1.0
df["weighted_war_23_24"] = df["war_23_24"] * 1.0
df["weighted_toi_23_24"] = df["toi_23_24"] * 1.0

df["weighted_gar_22_23"] = df["gar_22_23"] * 0.5
df["weighted_war_22_23"] = df["war_22_23"] * 0.5
df["weighted_toi_22_23"] = df["toi_22_23"] * 0.5

# Introduce Age Weighting (Age-Squared for Aging Curves)
df["age_squared"] = df["age"] ** 2

# Position & Contract Type Adjustments
df["position_adjustment"] = df["contract_type"].apply(lambda x: 1.1 if x == 2 else 1.0)
df["ufa_rfa_adjustment"] = df["contract_type"].apply(lambda x: 1.2 if x == 1 else 0.8)

features = [
    "age", "age_squared", "yoe", "prev_aav",
    "weighted_gar_24_25", "weighted_war_24_25", "weighted_toi_24_25",
    "weighted_gar_23_24", "weighted_war_23_24", "weighted_toi_23_24",
    "weighted_gar_22_23", "weighted_war_22_23", "weighted_toi_22_23",
    "position_adjustment", "ufa_rfa_adjustment",
    "Final_Contract_Term", "Predicted_Contract_Term"
]

df_cleaned = df[features + [target]].copy()

X = df_cleaned[features]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model again
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


df["Predicted Cap Hit %"] = model.predict(X)

# Calculate AAV for all players using $95.5M salary cap
salary_cap = 95500000
df["Predicted AAV"] = df["Predicted Cap Hit %"] * salary_cap / 100
df["Predicted AAV"] = df["Predicted AAV"].round(0).astype(int)


output_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/models/results/predicted_aav_all_players_updated.csv"
df_output = df[["player_name",  "Predicted Cap Hit %", "Predicted AAV"]]
df_output.to_csv(output_file, index=False)

print(f"Predicted AAV results saved to {output_file}")
