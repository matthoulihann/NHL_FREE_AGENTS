import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

fa_deduped = pd.read_csv('/2024-FA-Model_comparison.csv')
# Define the weighting function
def apply_optimized_weights(n1, n2, n3, w1=0.70, w2=0.20, w3=0.10):
    return (n1 * w1) + (n2 * w2) + (n3 * w3)

# Cleaned and merged 3-year FA dataset should already be loaded into `fa_deduped`

# Apply feature engineering
fa_deduped["TOI_Weighted_Optimized"] = apply_optimized_weights(
    fa_deduped["TOI_23_24"], fa_deduped["TOI_22_23"], fa_deduped["TOI_21_22"]
)
fa_deduped["GAR_Rolling_Avg"] = fa_deduped[["GAR_23_24", "GAR_22_23", "GAR_21_22"]].mean(axis=1)
fa_deduped["A1_Rolling_Avg"] = fa_deduped[["A1_23_24", "A1_22_23", "A1_21_22"]].mean(axis=1)
fa_deduped["iCF_Weighted_Optimized"] = apply_optimized_weights(
    fa_deduped["iCF_23_24"], fa_deduped["iCF_22_23"], fa_deduped["iCF_21_22"]
)
fa_deduped["A1_Weighted_Optimized"] = apply_optimized_weights(
    fa_deduped["A1_23_24"], fa_deduped["A1_22_23"], fa_deduped["A1_21_22"]
)
fa_deduped["Goals_Weighted_Optimized"] = apply_optimized_weights(
    fa_deduped["G_23_24"], fa_deduped["G_22_23"], fa_deduped["G_21_22"]
)

# TOI% placeholders
fa_deduped["TOI%_23_24"] = 0
fa_deduped["TOI%_22_23"] = 0
fa_deduped["TOI%_21_22"] = 0
fa_deduped["TOI%_Rolling_Avg"] = 0
fa_deduped["TOI%_Weighted_Optimized"] = 0

# Features for model (exclude prev_cap_hit_pct)
optimized_features = [
    "TOI%_Rolling_Avg", "GAR_Rolling_Avg", "A1_Rolling_Avg",
    "TOI%_Weighted_Optimized", "TOI_Weighted_Optimized",
    "iCF_Weighted_Optimized", "A1_Weighted_Optimized",
    "Goals_Weighted_Optimized"
]

# Filter rows with all required features + actual contract term
fa_ready = fa_deduped.dropna(subset=optimized_features + ["actual_contract_term"]).copy()
X = fa_ready[optimized_features]
y = fa_ready["actual_contract_term"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_scaled, y)

# Predict contract terms
fa_ready["Predicted_Term"] = model.predict(X_scaled).round().astype(int)

# Age cap rule (placeholder age = 28)
fa_ready["age"] = 28
def apply_max_contract(age, predicted_term):
    if age >= 35:
        return min(predicted_term, 3)
    elif 30 <= age <= 34:
        return min(predicted_term, 5)
    return predicted_term

fa_ready["Final_Contract_Term"] = fa_ready.apply(
    lambda row: apply_max_contract(row["age"], row["Predicted_Term"]), axis=1
)

# Evaluate model performance
mae = mean_absolute_error(y, fa_ready["Final_Contract_Term"])
r2 = r2_score(y, fa_ready["Final_Contract_Term"])

print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# View final result class
results = fa_ready[[
    "player_name", "actual_contract_term", "Predicted_Term", "Final_Contract_Term"
] + optimized_features]

results.to_csv("/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/2024_FA_Term_Model_Results.csv", index=False)
print("Exported predictions to 2024_FA_Term_Model_Results.csv")
