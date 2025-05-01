import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


free_agents_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/data/Updated_Free_Agent_GAR_24-25.csv"
contracts_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/data/Cleaned_NHL_Contract_Data.csv"

free_agents_df = pd.read_csv(free_agents_file)
contracts_df = pd.read_csv(contracts_file)

if 'Age' not in contracts_df.columns:
    for col in contracts_df.columns:
        if 'Age' in col:
            contracts_df.rename(columns={col: 'Age'}, inplace=True)
            break
    else:
        raise KeyError("No column containing 'Age' found in contracts_df.")

def apply_optimized_weights(n1, n2, n3, w1=0.70, w2=0.20, w3=0.10):
    return (n1 * w1) + (n2 * w2) + (n3 * w3)


contracts_df["TOI%_Rolling_Avg"] = contracts_df[["TOI%_24_25", "TOI%_23_24", "TOI%_22_23"]].mean(axis=1)
contracts_df["GAR_Rolling_Avg"] = contracts_df[["GAR_24_25", "GAR_23_24", "GAR_22_23"]].mean(axis=1)
contracts_df["A1_Rolling_Avg"] = contracts_df[["A1_24_25", "A1_23_24", "A1_22_23"]].mean(axis=1)

for feature in ['TOI%', 'GAR', 'TOI', 'iCF', 'A1', 'Goals']:
    contracts_df[f'{feature}_Weighted_Optimized'] = apply_optimized_weights(
        contracts_df.get(f'{feature}_24_25', 0),
        contracts_df.get(f'{feature}_23_24', 0),
        contracts_df.get(f'{feature}_22_23', 0)
    )

def assign_age_tier(age):
    if age <= 22:
        return 1
    elif 23 <= age <= 24:
        return 2
    elif 25 <= age <= 26:
        return 3
    elif 27 <= age <= 29:
        return 4
    elif 30 <= age <= 34:
        return 5
    else:
        return 6

contracts_df['Age_Tier'] = contracts_df['Age'].apply(assign_age_tier)


label_encoder = LabelEncoder()
free_agents_df['contract_type'] = label_encoder.fit_transform(free_agents_df['contract_type'])

contracts_df = pd.get_dummies(contracts_df, columns=['position'], drop_first=True)
free_agents_df = pd.get_dummies(free_agents_df, columns=['position'], drop_first=True)


optimized_features = [
    'TOI%_Rolling_Avg', 'GAR_Rolling_Avg', 'A1_Rolling_Avg',
    'TOI%_Weighted_Optimized', 'TOI_Weighted_Optimized',
    'iCF_Weighted_Optimized', 'A1_Weighted_Optimized',
    'Goals_Weighted_Optimized', 'prev_cap_hit_pct'
] + [col for col in contracts_df.columns if col.startswith('position_')]

y_optimized = contracts_df['Contract_Length']
X_optimized = contracts_df[optimized_features]

X_train_optimized, X_test_optimized, y_train_optimized, y_test_optimized = train_test_split(
    X_optimized.fillna(X_optimized.median()), y_optimized, test_size=0.2, random_state=42
)

scaler_optimized = StandardScaler()
X_train_optimized_scaled = scaler_optimized.fit_transform(X_train_optimized)
X_test_optimized_scaled = scaler_optimized.transform(X_test_optimized)

# Train Random Forest model 
rf_model_optimized = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model_optimized.fit(X_train_optimized_scaled, y_train_optimized)


y_pred_optimized = rf_model_optimized.predict(X_test_optimized_scaled)

for feature in ['TOI%', 'GAR', 'TOI', 'iCF', 'A1', 'Goals']:
    free_agents_df[f'{feature}_Weighted_Optimized'] = apply_optimized_weights(
        free_agents_df.get(f'{feature}_24_25', 0),
        free_agents_df.get(f'{feature}_23_24', 0),
        free_agents_df.get(f'{feature}_22_23', 0)
    )

# Ensure missing features are accounted for
for col in optimized_features:
    if col not in free_agents_df.columns:
        free_agents_df[col] = 0

free_agents_df['Predicted_Contract_Term'] = rf_model_optimized.predict(
    scaler_optimized.transform(free_agents_df[optimized_features])
).round().astype(int)

# Contract limits based on age
def apply_max_contract(age, predicted_term):
    if age >= 35:
        return min(round(predicted_term), 3)
    elif 30 <= age <= 34:
        return min(round(predicted_term), 5)
    return round(predicted_term)

free_agents_df['Final_Contract_Term'] = free_agents_df.apply(
    lambda row: apply_max_contract(row['age'], row['Predicted_Contract_Term']), axis=1
)

corrected_contract_term_predictions_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/models/results/predicted_contract_term.csv"
free_agents_df.to_csv(corrected_contract_term_predictions_file, index=False)

filtered_contract_terms_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/models/results/filtered_contract_terms.csv"
free_agents_df[['player_name', 'age', 'contract_type', 'Final_Contract_Term']].to_csv(filtered_contract_terms_file, index=False)
