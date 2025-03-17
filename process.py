import pandas as pd

# Load the dataset
df = pd.read_csv("team_kd_results.csv")

# Convert to proper format
df_win = df[['game_id', 'team_name', 'team_KD_win']].dropna().rename(columns={'team_KD_win': 'team_KD'})
df_win['win'] = 1

df_loss = df[['game_id', 'team_name', 'team_KD_loss']].dropna().rename(columns={'team_KD_loss': 'team_KD'})
df_loss['win'] = 0

# Combine into a single dataset
df_final = pd.concat([df_win, df_loss], ignore_index=True)

# Save the formatted dataset
df_final.to_csv("formatted_team_kd_results.csv", index=False)
