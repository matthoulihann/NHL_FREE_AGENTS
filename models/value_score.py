import pandas as pd
import os

cap_hit_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/models/results/predicted_aav_all_players_updated.csv"
gar_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/GAR Stats/projected_gar_25-26_final.csv"
term_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/models/results/filtered_contract_terms.csv"

df_cap_hit = pd.read_csv(cap_hit_file)
df_gar = pd.read_csv(gar_file)
df_term = pd.read_csv(term_file)

# Standardize player names
df_cap_hit['player_name'] = df_cap_hit['player_name'].str.lower().str.strip()
df_gar['player_name'] = df_gar['player_name'].str.lower().str.strip()
df_term['player_name'] = df_term['player_name'].str.lower().str.strip()

df = df_cap_hit.merge(df_gar, on='player_name', how='inner')
df = df.merge(df_term[['player_name', 'Final_Contract_Term', 'age']], on='player_name', how='inner')

if 'Predicted AAV' in df.columns and 'Final_Predicted_GAR_25_26' in df.columns:
    # Filter out invalid GAR values
    df = df[df['Final_Predicted_GAR_25_26'] > 0]
    df.dropna(subset=['Final_Predicted_GAR_25_26'], inplace=True)

    # Calculate League Cost Per GAR (Filtering out low-GAR players < 1)
    filtered_df = df[df['Final_Predicted_GAR_25_26'] > 1]
    total_AAV = filtered_df['Predicted AAV'].sum()
    total_GAR = filtered_df['Final_Predicted_GAR_25_26'].sum()
    league_cost_per_gar = total_AAV / total_GAR if total_GAR > 0 else 0

    # Apply an inflation factor based on market trends
    inflation_factor = 1.15  # Adjust based on historical cost growth
    league_cost_per_gar *= inflation_factor

    print(f"Adjusted League Cost Per GAR: ${league_cost_per_gar:,.2f}")

    # Adjust FMV
    df['Fair_Market_Value'] = df['Final_Predicted_GAR_25_26'] * league_cost_per_gar

    # Ensure Fair Market Value (FMV) and AAV are not lower than 775k (league minimum)
    df['Fair_Market_Value'] = df['Fair_Market_Value'].apply(lambda x: max(x, 775000)).round(0)
    df['Predicted AAV'] = df['Predicted AAV'].apply(lambda x: max(x, 775000)).round(0)

    # Calculate Value Rating
    df['Value_Rating'] = df['Fair_Market_Value'] - df['Predicted AAV']

    # Normalize Value Rating on a 0-100 scale using Min-Max Scaling (rounded numbers only)
    min_value = df['Value_Rating'].min()
    max_value = df['Value_Rating'].max()

    if max_value > min_value:  # Avoid division by zero
        df['Value_Score'] = ((df['Value_Rating'] - min_value) / (max_value - min_value) * 100)
    else:
        df['Value_Score'] = 50  # Default middle score if no variation

    # Clip scores to ensure they are within 0-100 range
    df['Value_Score'] = df['Value_Score'].clip(lower=0, upper=100).round(0).astype(int)

    # Categorize players based on value
    df['Value_Category'] = df['Value_Rating'].apply(lambda x: 'Undervalued' if x > 0 else 'Overvalued')

    output_file = "/Users/matthoulihan/PycharmProjects/NHLFreeAgentEvaluationModel/models/results/player_value_assessment.csv"
    df['player_name'] = df['player_name'].str.title()
    df_output = df[
        ['player_name', 'Predicted AAV', 'Final_Predicted_GAR_25_26', 'Fair_Market_Value', 'Value_Rating',
         'Value_Score', 'Value_Category', 'Final_Contract_Term', 'age']]

    print("Saving CSV to:", output_file)
    print("Output DataFrame Shape:", df_output.shape)
    print("Sample Output Data:", df_output.head())

    df_output.to_csv(output_file, index=False)

    if os.path.exists(output_file):
        print("CSV successfully saved!")
    else:
        print("Error: CSV not found after saving.")
else:
    print("Required columns not found in the dataset.")
