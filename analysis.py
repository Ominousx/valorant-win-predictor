import streamlit as st
import joblib
import numpy as np

def load_model():
    model = joblib.load("logistic_regression_model.pkl")  # Use joblib to load model
    scaler = joblib.load("scaler.pkl")  # Use joblib to load scaler
    return model, scaler

def predict_win_probability(model, scaler, kd_ratio):
    kd_ratio = np.array([[kd_ratio]])  # Reshape for model input
    kd_ratio_scaled = scaler.transform(kd_ratio)  # Apply scaling
    probability = model.predict_proba(kd_ratio_scaled)[0, 1]  # Probability of winning
    return probability

def main():
    st.title("Valorant Match Win Predictor")
    model, scaler = load_model()

    st.header("Enter Team Stats")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Team Name")
    with col2:
        st.subheader("Kills")
    with col3:
        st.subheader("Deaths")

    team_a = st.text_input("", placeholder="Enter Team A name", key="team_a")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(team_a)
    with col2:
        team_a_kills = st.number_input("", min_value=0, step=1, key="team_a_kills")
    with col3:
        team_a_deaths = st.number_input("", min_value=0, step=1, key="team_a_deaths")

    st.markdown("---")

    team_b = st.text_input("", placeholder="Enter Team B name", key="team_b")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(team_b)
    with col2:
        team_b_kills = st.number_input("", min_value=0, step=1, key="team_b_kills")
    with col3:
        team_b_deaths = st.number_input("", min_value=0, step=1, key="team_b_deaths")

    if st.button("Predict Win Probability"):
        team_a_kd = team_a_kills / max(1, team_a_deaths)
        team_b_kd = team_b_kills / max(1, team_b_deaths)

        team_a_win_prob = predict_win_probability(model, scaler, team_a_kd)
        team_b_win_prob = predict_win_probability(model, scaler, team_b_kd)

        st.subheader("Win Probability Predictions")
        st.write(f"**{team_a}:** {team_a_win_prob:.2%} chance to win")
        st.write(f"**{team_b}:** {team_b_win_prob:.2%} chance to win")

if __name__ == "__main__":
    main()
