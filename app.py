import streamlit as st
import pandas as pd
import pickle

# Load the model, scaler, and label encoders
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('le_team.pkl', 'rb') as le_file:
    le_team = pickle.load(le_file)

with open('le_opponent.pkl', 'rb') as le_file:
    le_opponent = pickle.load(le_file)

with open('le_venue.pkl', 'rb') as le_file:
    le_venue = pickle.load(le_file)

# Load and prepare data for calculating stats
data = pd.read_csv('matches.csv')

# Define a mapping of team names to standardized names
team_mapping = {
    'Manchester City': 'Manchester City',
    'Manchester United': 'Manchester United',
    'Liverpool': 'Liverpool',
    'Chelsea': 'Chelsea',
    'Leicester City': 'Leicester City',
    'West Ham United': 'West Ham United',
    'Tottenham Hotspur': 'Tottenham Hotspur',
    'Arsenal': 'Arsenal',
    'Leeds United': 'Leeds United',
    'Everton': 'Everton',
    'Aston Villa': 'Aston Villa',
    'Newcastle United': 'Newcastle United',
    'Wolverhampton Wanderers': 'Wolverhampton Wanderers',
    'Crystal Palace': 'Crystal Palace',
    'Southampton': 'Southampton',
    'Brighton and Hove Albion': 'Brighton and Hove Albion',
    'Burnley': 'Burnley',
    'Fulham': 'Fulham',
    'West Bromwich Albion': 'West Bromwich Albion',
    'Sheffield United': 'Sheffield United',
    'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford',
    'Nottingham Forest': 'Nottingham Forest',
    'Luton Town': 'Luton Town',
    'Watford': 'Watford',
    'Norwich City': 'Norwich City'
}

# Standardize opponent names
data['opponent'] = data['opponent'].replace(
    team_mapping).fillna(data['opponent'])

# Convert categorical columns to numeric using LabelEncoder
data['team'] = le_team.transform(data['team'])
data['opponent'] = le_opponent.transform(data['opponent'])
data['venue'] = le_venue.transform(data['venue'])

# Calculate mean statistics for home and away matches for all teams
home_stats = data[data['venue'] == le_venue.transform(
    ['Home'])[0]][['team', 'gf', 'ga', 'xg', 'xga', 'poss']].groupby('team').mean()
away_stats = data[data['venue'] == le_venue.transform(
    ['Away'])[0]][['team', 'gf', 'ga', 'xg', 'xga', 'poss']].groupby('team').mean()

# Streamlit UI
st.title('Football Match Result Prediction')

# Team selection
teams = list(le_team.classes_)
team1_name = st.selectbox('Select Team 1', teams)
team2_name = st.selectbox('Select Team 2', teams)

# Venue selection
venues = ['Home', 'Away']
venue = st.selectbox('Select Venue', venues)

# Predict button
if st.button('Predict Match Result'):
    # Convert team names to numeric
    team1 = le_team.transform([team1_name])[0]
    team2 = le_team.transform([team2_name])[0]

    # Convert venue to numeric
    venue_encoded = le_venue.transform([venue])[0]

    # Get the statistics for the teams
    team1_stats = home_stats.loc[team1] if venue == 'Home' else away_stats.loc[team1]
    team2_stats = home_stats.loc[team2] if venue == 'Home' else away_stats.loc[team2]

    # Prepare feature matrix for prediction
    future_match_features = {
        'team': team1,
        'opponent': team2,
        'gf': (team1_stats['gf'] + team2_stats['gf']) / 2,
        'ga': (team1_stats['ga'] + team2_stats['ga']) / 2,
        'xg': (team1_stats['xg'] + team2_stats['xg']) / 2,
        'xga': (team1_stats['xga'] + team2_stats['xga']) / 2,
        'poss': (team1_stats['poss'] + team2_stats['poss']) / 2
    }

    future_match_df = pd.DataFrame([future_match_features])

    # Scale features
    future_match_df_scaled = scaler.transform(future_match_df)

    # Predict future match result
    future_prediction = model.predict(future_match_df_scaled)
    result_map = {0: 'Loss', 1: 'Draw', 2: 'Win'}
    result = result_map[future_prediction[0]]

    st.write(
        f"Predicted result for the match between {team1_name} and {team2_name} ({venue}): {result}")
