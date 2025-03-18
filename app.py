import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Valorant Win Probability Predictor")
st.write("Enter each player's kills and deaths to calculate the team's K:D ratio and predict the win probability.")

def calculate_team_kd(kills, deaths):
    total_kills = sum(kills)
    total_deaths = sum(deaths)
    return total_kills / max(total_deaths, 1)  # Avoid division by zero

# Team A inputs
st.subheader("Team A")
team_a_kills = [st.number_input(f"Player {i+1} Kills (Team A)", min_value=0, step=1) for i in range(5)]
team_a_deaths = [st.number_input(f"Player {i+1} Deaths (Team A)", min_value=0, step=1) for i in range(5)]
team_a_kd = calculate_team_kd(team_a_kills, team_a_deaths)

# Team B inputs
st.subheader("Team B")
team_b_kills = [st.number_input(f"Player {i+1} Kills (Team B)", min_value=0, step=1) for i in range(5)]
team_b_deaths = [st.number_input(f"Player {i+1} Deaths (Team B)", min_value=0, step=1) for i in range(5)]
team_b_kd = calculate_team_kd(team_b_kills, team_b_deaths)

if st.button("Predict Win Probability"):
    # Scale each K:D separately
    team_a_kd_scaled = scaler.transform(np.array([[team_a_kd]]))
    team_b_kd_scaled = scaler.transform(np.array([[team_b_kd]]))

    # Predict win probabilities
    team1_prob = model.predict_proba(team_a_kd_scaled)[:, 1][0] * 100
    team2_prob = model.predict_proba(team_b_kd_scaled)[:, 1][0] * 100

    # Normalize probabilities so they sum to 100%
    total = team1_prob + team2_prob
    team1_final = (team1_prob / total) * 100
    team2_final = (team2_prob / total) * 100

    # Display results
    st.subheader("Win Probability")
    st.write(f"**Team A:** {team1_final:.2f}%")
    st.write(f"**Team B:** {team2_final:.2f}%")
