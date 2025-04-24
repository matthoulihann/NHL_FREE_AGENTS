import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ========== STEP 1: LOAD DATA ==========
# Load GAR Data from the past three seasons
gar_22_23 = pd.read_csv("/GAR Stats/projected_gar_22-23.csv")
gar_23_24 = pd.read_csv("/GAR Stats/projected_gar_23-24.csv")
gar_24_25 = pd.read_csv("/GAR Stats/projected_gar_24-25.csv")

# Load Free Agents Data
free_agents_df = pd.read_csv("/data/cleaned_free_agents_2025 (1).csv")

# ========== STEP 2: CLEAN & PREPROCESS DATA ==========
# Rename "Player" to "player_name" for consistency
gar_22_23 = gar_22_23.rename(columns={"Player": "player_name", "TOI_All": "TOI_22_23", "GAR": "GAR_22_23"})
gar_23_24 = gar_23_24.rename(columns={"Player": "player_name", "TOI_All": "TOI_23_24", "GAR": "GAR_23_24"})
gar_24_25 = gar_24_25.rename(columns={"Player": "player_name", "TOI_All": "TOI_24_25", "GAR": "GAR_24_25"})

# Merge GAR Data with Free Agents
free_agents_df = free_agents_df.merge(gar_22_23, on="player_name", how="left")
free_agents_df = free_agents_df.merge(gar_23_24, on="player_name", how="left")
free_agents_df = free_agents_df.merge(gar_24_25, on="player_name", how="left")

# Fill missing TOI values with the median TOI
for col in ["TOI_22_23", "TOI_23_24", "TOI_24_25"]:
    free_agents_df[col] = free_agents_df[col].fillna(free_agents_df[col].median())

# Create Weighted GAR Feature (Evolving Hockey's Approach)
def weighted_gar_projection(row):
    weights = [0.20, 0.30, 0.50]  # 50/30/20 weighting for past three seasons
    return (
        row["GAR_22_23"] * weights[0] +
        row["GAR_23_24"] * weights[1] +
        row["GAR_24_25"] * weights[2]
    )

free_agents_df["Weighted_GAR_25_26"] = free_agents_df.apply(weighted_gar_projection, axis=1)

# Apply Aging Curve Adjustments
def apply_aging_curve(row):
    if row["Age"] < 24:
        return row["Weighted_GAR_25_26"] * 1.05  # Boost for young players
    elif row["Age"] >= 31:
        return row["Weighted_GAR_25_26"] * 0.95  # Decline for older players
    return row["Weighted_GAR_25_26"]

free_agents_df["Final_GAR_25_26"] = free_agents_df.apply(apply_aging_curve, axis=1)

# Define Features for Model Training
features = ["GAR_22_23", "GAR_23_24", "GAR_24_25", "TOI_22_23", "TOI_23_24", "TOI_24_25", "Weighted_GAR_25_26"]
free_agents_df = free_agents_df.dropna(subset=features + ["Final_GAR_25_26"])

X = free_agents_df[features]
y = free_agents_df["Final_GAR_25_26"]

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Multiple Models
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=500, random_state=42),
    "Linear Regression": LinearRegression()
}

results = {}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    results[model_name] = {"MAE": mae, "R²": r_squared}
    print(f"📊 {model_name} - MAE: {mae:.4f}, R²: {r_squared:.3f}")

