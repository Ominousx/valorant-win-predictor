import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    model = joblib.load("logistic_regression_model.pkl")  # Use joblib to load model
    scaler = joblib.load("scaler.pkl")  # Use joblib to load scaler
    return model, scaler

def predict_win_probability(model, scaler, kd_ratio):
    kd_ratio = np.array([[kd_ratio]])  # Reshape for model input
    kd_ratio_scaled = scaler.transform(kd_ratio)  # Apply scaling
    probability = model.predict_proba(kd_ratio_scaled)[0, 1]  # Probability of winning
    return probability

def plot_win_probability_curve(model, scaler, team_a_kd, team_b_kd, team_a, team_b):
    kd_values = np.linspace(0.5, 2.5, 100)
    probabilities = [predict_win_probability(model, scaler, kd) for kd in kd_values]

    plt.figure(figsize=(8, 5))
    plt.plot(kd_values, probabilities, label="Win Probability", color="blue")

    # Thresholds
    kd_50_winrate = 1.01
    kd_unlosable = 1.08
    plt.axvline(x=kd_50_winrate, linestyle="dashed", color="green", label=f"50% Win Rate (K:D = {kd_50_winrate})")
    plt.axvline(x=kd_unlosable, linestyle="dashed", color="red", label=f"Unlosable Threshold (K:D = {kd_unlosable})")

    # Team K:D markers with dots
    plt.scatter([team_a_kd], [predict_win_probability(model, scaler, team_a_kd)], color="orange", label=f"{team_a} ({team_a_kd:.2f})", zorder=3)
    plt.scatter([team_b_kd], [predict_win_probability(model, scaler, team_b_kd)], color="purple", label=f"{team_b} ({team_b_kd:.2f})", zorder=3)

    plt.xlabel("Team K:D")
    plt.ylabel("Win Probability")
    plt.title("Win Probability vs. Team K:D")
    plt.legend()
    st.pyplot(plt)

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

    if st.button("🔮 Predict Win Probability", help="Click to predict win probability based on K:D ratios"):
        team_a_kd = team_a_kills / max(1, team_a_deaths)
        team_b_kd = team_b_kills / max(1, team_b_deaths)

        team_a_win_prob = predict_win_probability(model, scaler, team_a_kd)
        team_b_win_prob = predict_win_probability(model, scaler, team_b_kd)

        st.subheader("Win Probability Predictions")

        # Dynamic color coding
        if team_a_win_prob > 0.7:
            st.success(f"**{team_a}:** {team_a_win_prob:.2%} chance to win")
        elif team_a_win_prob < 0.3:
            st.error(f"**{team_a}:** {team_a_win_prob:.2%} chance to win")
        else:
            st.warning(f"**{team_a}:** {team_a_win_prob:.2%} chance to win")

        if team_b_win_prob > 0.7:
            st.success(f"**{team_b}:** {team_b_win_prob:.2%} chance to win")
        elif team_b_win_prob < 0.3:
            st.error(f"**{team_b}:** {team_b_win_prob:.2%} chance to win")
        else:
            st.warning(f"**{team_b}:** {team_b_win_prob:.2%} chance to win")

        st.subheader("Team K:D Ratios")
        st.write(f"**{team_a} K:D:** {team_a_kd:.2f}")
        st.write(f"**{team_b} K:D:** {team_b_kd:.2f}")

        # Plot win probability curve
        plot_win_probability_curve(model, scaler, team_a_kd, team_b_kd, team_a, team_b)

if __name__ == "__main__":
    main()
