import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load engineered 2024 FA dataset
fa_deduped = pd.read_csv('/2024_FA_Term_Model_Results.csv')

# Feature engineering
def apply_optimized_weights(n1, n2, n3, w1=0.70, w2=0.20, w3=0.10):
    return (n1 * w1) + (n2 * w2) + (n3 * w3)

fa_deduped["TOI_Weighted_Optimized"] = apply_optimized_weights(
    fa_deduped["TOI_23_24"], fa_deduped["TOI_22_23"], fa_deduped["TOI_21_22"]
)
fa_deduped["GAR_Rolling_Avg"] = fa_deduped[["GAR_23_24", "GAR_22_23", "GAR_21_22"]].mean(axis=1)
fa_deduped["A1_Rolling_Avg"] = fa_deduped[["A1_23_24", "A1_22_23", "A1_21_22"]].mean(axis=1)

# Add placeholders for cap hit model
fa_deduped["age"] = 28
fa_deduped["yoe"] = 6
fa_deduped["prev_aav"] = 4000000
fa_deduped["contract_type"] = 1  # Assume UFA

# Add age-squared and adjustment features
fa_deduped["age_squared"] = fa_deduped["age"] ** 2
fa_deduped["position_adjustment"] = fa_deduped["contract_type"].apply(lambda x: 1.1 if x == 2 else 1.0)
fa_deduped["ufa_rfa_adjustment"] = fa_deduped["contract_type"].apply(lambda x: 1.2 if x == 1 else 0.8)

# Map to original cap hit model feature names
fa_deduped = fa_deduped.rename(columns={
    "GAR_Rolling_Avg": "weighted_gar_23_24",
    "A1_Rolling_Avg": "weighted_war_23_24",
    "TOI_Weighted_Optimized": "weighted_toi_23_24"
})

# Simulate previous/future seasons (copy & scale)
fa_deduped["weighted_gar_24_25"] = fa_deduped["weighted_gar_23_24"] * 1.5
fa_deduped["weighted_war_24_25"] = fa_deduped["weighted_war_23_24"] * 1.5
fa_deduped["weighted_toi_24_25"] = fa_deduped["weighted_toi_23_24"] * 1.5

fa_deduped["weighted_gar_22_23"] = fa_deduped["weighted_gar_23_24"] * 0.5
fa_deduped["weighted_war_22_23"] = fa_deduped["weighted_war_23_24"] * 0.5
fa_deduped["weighted_toi_22_23"] = fa_deduped["weighted_toi_23_24"] * 0.5

# Define feature set
cap_hit_features = [
    "age", "age_squared", "yoe", "prev_aav",
    "weighted_gar_24_25", "weighted_war_24_25", "weighted_toi_24_25",
    "weighted_gar_23_24", "weighted_war_23_24", "weighted_toi_23_24",
    "weighted_gar_22_23", "weighted_war_22_23", "weighted_toi_22_23",
    "position_adjustment", "ufa_rfa_adjustment",
    "Final_Contract_Term", "Predicted_Term"
]

# Drop missing
df_model = fa_deduped.dropna(subset=cap_hit_features).copy()
X = df_model[cap_hit_features]

# Train (or load pre-trained) model — here we'll train on dummy targets
y_dummy = np.random.uniform(0.5, 12.0, len(X))  # Replace with real targets if training

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y_dummy)

# Predict
df_model["Predicted Cap Hit %"] = model.predict(X)

# Convert to AAV using 2024–25 cap
cap_2024 = 88_000_000
df_model["Predicted AAV"] = (df_model["Predicted Cap Hit %"] / 100) * cap_2024
df_model["Predicted AAV"] = df_model["Predicted AAV"].round().astype(int)

# Export
output_path = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/2024_FA_Cap_Hit_Predictions.csv"
df_model[["player_name", "Predicted Cap Hit %", "Predicted AAV", "Final_Contract_Term", "Predicted_Term"]].to_csv(output_path, index=False)

print(f"✅ Exported: {output_path}")
