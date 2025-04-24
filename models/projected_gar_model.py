import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score



gar_22_23 = pd.read_csv("/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/GAR Stats/projected_gar_22-23.csv")
gar_23_24 = pd.read_csv("/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/GAR Stats/projected_gar_23-24.csv")
gar_24_25 = pd.read_csv("/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/GAR Stats/projected_gar_24-25.csv")

free_agents_df = pd.read_csv("/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/data/cleaned_free_agents_2025 (1).csv")

# Check for duplicate player names
print(f"Free Agents Duplicate Players: {free_agents_df.duplicated(subset=['player_name']).sum()}")


# Rename columns for consistency
gar_22_23 = gar_22_23.rename(columns={"Player": "player_name", "TOI_All": "TOI_22_23", "GAR": "GAR_22_23"})
gar_23_24 = gar_23_24.rename(columns={"Player": "player_name", "TOI_All": "TOI_23_24", "GAR": "GAR_23_24"})
gar_24_25 = gar_24_25.rename(columns={"Player": "player_name", "TOI_All": "TOI_24_25", "GAR": "GAR_24_25"})

# Aggregate GAR data to remove duplicates while preserving stats
agg_funcs = {
    "GAR_22_23": "sum",
    "TOI_22_23": "sum",
}
gar_22_23 = gar_22_23.groupby("player_name", as_index=False).agg(agg_funcs)

agg_funcs = {
    "GAR_23_24": "sum",
    "TOI_23_24": "sum",
}
gar_23_24 = gar_23_24.groupby("player_name", as_index=False).agg(agg_funcs)

agg_funcs = {
    "GAR_24_25": "sum",
    "TOI_24_25": "sum",
}
gar_24_25 = gar_24_25.groupby("player_name", as_index=False).agg(agg_funcs)


free_agents_df = free_agents_df.merge(gar_22_23, on="player_name", how="left")
free_agents_df = free_agents_df.merge(gar_23_24, on="player_name", how="left")
free_agents_df = free_agents_df.merge(gar_24_25, on="player_name", how="left")

# Fill missing TOI values with the median TOI
for col in ["TOI_22_23", "TOI_23_24", "TOI_24_25"]:
    free_agents_df[col] = free_agents_df[col].fillna(free_agents_df[col].median())

def weighted_gar_projection(row):
    weights = [0.20, 0.30, 0.50]  # 50/30/20 weighting for past three seasons
    return (
        row["GAR_22_23"] * weights[0] +
        row["GAR_23_24"] * weights[1] +
        row["GAR_24_25"] * weights[2]
    )

free_agents_df["Weighted_GAR_25_26"] = free_agents_df.apply(weighted_gar_projection, axis=1)

# Apply Aging Curve
def apply_aging_curve(row):
    if row["Age"] < 24:
        return row["Weighted_GAR_25_26"] * 1.05  # Boost for young players
    elif row["Age"] >= 31:
        return row["Weighted_GAR_25_26"] * 0.95  # Decline for older players
    return row["Weighted_GAR_25_26"]

free_agents_df["Final_GAR_25_26"] = free_agents_df.apply(apply_aging_curve, axis=1)

features = ["GAR_22_23", "GAR_23_24", "GAR_24_25", "TOI_22_23", "TOI_23_24", "TOI_24_25", "Weighted_GAR_25_26"]
free_agents_df = free_agents_df.dropna(subset=features + ["Final_GAR_25_26"])

X = free_agents_df[features]
y = free_agents_df["Final_GAR_25_26"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge Regression Model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)


y_pred = ridge_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f"📊 Projected GAR Model - MAE: {mae:.4f}, R²: {r_squared:.3f}")

y_final_pred = ridge_model.predict(scaler.transform(free_agents_df[features]))
free_agents_df["Final_Predicted_GAR_25_26"] = y_final_pred

output_path = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/GAR Stats/projected_gar_25-26_final.csv"
free_agents_df[["player_name", "Final_Predicted_GAR_25_26"]].to_csv(output_path, index=False)
print(f"✅ Final Projected GAR saved to: {output_path}")
