# Valorant Win Probability Predictor

This is a Streamlit web app that predicts the win probability of a Valorant team based on their current Team K:D ratio.

## Features
- Input kills and deaths for each player on both teams.
- Automatically calculates the team's K:D ratio.
- Uses a trained machine learning model to predict the win probability.
- Displays the probability percentage for each team in real-time.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/valorant-win-predictor.git
   cd valorant-win-predictor
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the app:
   ```sh
   streamlit run app.py
   ```

## Deployment on Streamlit Cloud
1. Push the repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click 'New App' and select your repository.
4. Set the main file as `app.py`.
5. Click 'Deploy'.

## Model
The app uses a trained machine learning model (`team_kd_win_model.pkl`) to make predictions based on historical match data.

## License
MIT License

]
