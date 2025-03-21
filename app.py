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

def plot_win_probability_curve(model, scaler, team_a_kd, team_b_kd, team_a, team_b, map_name):
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
    plt.scatter([team_a_kd], [predict_win_probability(model, scaler, team_a_kd)], color="orange", label=f"{team_a} ({team_a_kd:.2f})")
    plt.scatter([team_b_kd], [predict_win_probability(model, scaler, team_b_kd)], color="purple", label=f"{team_b} ({team_b_kd:.2f})")

    plt.xlabel("Team K:D")
    plt.ylabel("Win Probability")
    plt.title(f"{map_name} Win Probability vs. Team K:D")
    plt.legend()
    st.pyplot(plt)

def main():
    st.title("Valorant Match Win Predictor")
    model, scaler = load_model()

    st.header("Enter Match Details")
    map_name = st.text_input("Map Name", placeholder="Enter map name", key="map_name")

    st.subheader("Enter Team Stats")
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

        st.subheader("Match Summary")
        match_summary = f"""
        | Team   | K:D Ratio | Win Probability |
        |--------|----------|----------------|
        | **{team_a}** | {team_a_kd:.2f} | {team_a_win_prob:.2%} |
        | **{team_b}** | {team_b_kd:.2f} | {team_b_win_prob:.2%} |
        """
        st.markdown(match_summary)

        # Plot win probability curve
        plot_win_probability_curve(model, scaler, team_a_kd, team_b_kd, team_a, team_b, map_name)

if __name__ == "__main__":
    main()
