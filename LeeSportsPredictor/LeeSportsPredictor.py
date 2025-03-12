
from curses.panel import bottom_panel
from multiprocessing import JoinableQueue
from os import error
from pickletools import pybool
from platform import libc_ver
from queue import Empty
import time
from turtle import st
from xml.sax import make_parser
import requests
import pandas as pd

API_KEY = 'your_api_key_here'
url = 'https://api.football-data.org/v4/competitions/PL/matches'
headers = {'X-Auth-Token': API_KEY}
response = requests.get(url, headers=headers)

if response.status_code ==200:
    data = respnse.json()
    matches = data['matches']
    df = pd.DataFrame(matches)
    print(df.head())
    else:
      print(''Error:'', response.status_code)

        import pandas as pd
        # Load dataset
        data= pd. read_csv('sports_data.csv')
        print(data.head())
        print(data.info())
        print(data.isnull().sum())
        # Fill missing data with 0
        data.fillna(0, inplace=True)


        import pandas as pd
        from sklearn.model_selection import train_test_split

        # Load dataset
        data= pd.read_csv('sports_data.csv')

        # Select features (X) and target (y)
        X = data[['possession', 'shots_on_target', 'home_goals', 'away_goals']]
        y = data['home_win']

        # Slipt data into training (70%) and testing (30%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ size=0.3, random_state=42)
        
        print(f''Training samples: {len(X_train)}, Testing samples: {len(X_test)}'')

        from sklearn.ensemble import
        RandomForestClassifier

        # Initialize the model
        model = 
        RandomForestClassifier(n_estimators=100, random_state=42)
        # Train the model
        model.fit(X_train, y_train)
        print(''Model training complete!'')

        from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
        # Maake predictions
        y_pred = model.predict(X_test)
        # Evaluate accuracy
        print(''Accuracy: '', accuracy_score(y_test, y_pred))
        # Detailed report
        print(classification_report(y_test, y_pred))

        # Example input: possession, shots_on_target, home_goals, away_goals new_match = [[60, 8, 2, 1]]

        # Predict the outcome (1= Home win, 0= Loss/Draw)
        prediction = model.predict(new_match)
        print(''Prediction:'', ''Home Win'' if prediction[0] == 1 else ''Away/Draw'')

        model =
        RandomForestClassifier(n_estimators=100, random_state=42)

        data['home_advantage'] =
        (data[home_team'] == ''Your Team'').astype(int)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler. transform(X_test)
        
        from sklearn.model_selection import
        RandomizedSearchCv
        from sklearn.ensemble import
        RandomForestClassifier
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10,20,None],
            'min_samples_split': [2, 5, 10],
        }

        model =
        RandomForestClassifier(random_state=42)

        # Randomized Search
        search = RandomizedSearchCV(model, param_grid, cv=5, n_iter=10, scoring= 'accuracy', n_jobs=-1) search.fit(X_train, y_train)
        print(''Best parameters:'', search.best_params_)

        from sklearn.model_selection import cross_val_score
        model=
        RandomForestClassifier(n_estimators=200, random_state=42)
        score= cross_val_score(model, X, y, cv=5)
        print(''Cross-Validation Accuracy:'', scores.mean())

        pip intall xgboost
        import xgboost as xgb
        model =
        xgb.XGBClassifier(n_estimators=200, max_depth=5)
        model.fit(X_train, y_train)
        print(''XGBoost Accuracy:'', model.score(X_test, y_test))
        
        from sklearn.metrics import roc_auc_score,confusion_matrix
        y_prob = model.predict_proba(X_test)[:, 1]
        print(''AUC Score:'', roc_auc_score(y_test,y_prob))
        print(''Confusion Matrix:/n'', confusion_matrix(y_test,y_pred))

        import requests
        import pandas as pd
        # Replace with your API key 
        API_KEY = ''YOUR_API_KEY''
        url = ''https://api.sportsdata.io/v4/soccer/scores/json/Live'' 
        headers = {'Ocp-Apim-Subscription-Key': API_KEY}
         
        # Fetch live scores
        response = requests.get(url, headers=headers)
        live_data = response.json()
        # Convert to DataFrame for easier handling
        live_df = pd.DataFrame(live_data)
        print(live_df.head())

        # Extract relevant features for prediction
        live_df['HomeTeamScore'] =
        live_df['HomeTeamScore'].fillna(0)
        live_df['AwayTeamScore'] =
        live_df['AwayTeamScore'].fillna(0)

        # Example: Add additional features if required
        live_df['VenueAdvantage'] =
        (live_df['HomeTeamId'] ==
       live_df['VenueId']).astype(int)
       live_features = live_df[['HomeTeamScore', 'AwayTeamScore', 'VenueAdvantage']]

       import Joblib
       # Load the trained model
       model =
      joblib.load (''sports_predictor.pkl'')

        # Make predictions
        live_predictions = model.predict(live_features)
        live_df['PredictWinner'] = live_predictions
        print(live_df[['HomeTeam', 'AwayTeam', 'PredictWinner']])
        
        from tensorflow.keras.models import load_model
        # Load trained keras mode
        dl_model = load_model('sports_predictor.h5')
        # Make predictions (probability > 0.5 means Home Win)
        live_probabilities = dl_model.predict (live_features)
        live_df['PredictedWinner'] = (live_probabilities > 0.5).astytype(int)
        print(live_df[['HomeTeamName', 'AwayTeamName', 'PredictedWinner']])

        import time
        while true:
            response = requests.get(url, headers=headers)
            live_data = response.json()
            live_df = pd.DataFrame(live_data)
            live_features = live_df[['HomeTeamScore' , 'AwayTeamScore']]
            predictions = model.predict(live_features)
            live_df['PredictWinner'] = predictions
            print(live_df[['HomeTeam', 'AwayTeam', 'PredictWinner']])
            # Wait 5 minutes before fetching new data
            time.sleep(300)

            pip install streamlit
            import streamlit as st
            import pandas as pd
            import requests
            import joblib

            # Load the trained prediction model
            model = joblib.load('sports_predictor.pkl')
            # API configuration (replace with your live API details)
            API_KEY = 'YOUR_API_KEY''
            Live_API_URL=
             'https://api.sportsdata.io/v4/soccer/scores/json/Live'
             headers = {'Ocp-Apim-Subscription-Key': API_KEY}

               # Fetch live sprts data
               def fetch_live_data():
                   response = requests.get(Live_API_URL, headers=headers)
                   if response.status_code == 200:
                       return
                       pd.DataFrame(response.json())
                       else
                       st.error(''Error fetching live data'')
                       return pd.DataFrame()
                   # Make predictions   
                   def make_predictions(df): if not df. empty:
df['HomeTeamScore'] = df['HomeTeamScore'].fillna(0) 
df['AwayTeamScore'] = df['AwayTeamScore'].fillna(0)

# Exact required features
features = df[['HomeTeamScore', 'AwayTeamScore']]
predictions = model.predict(features)
df['PredictWinner'] = predictions
return df

# Streamlit Dashboard Layout
st.title(''Live Sports Prediction Dashboard'')
st.sidebar.header(''Controls'')
refresh_rate = st.sidebar.slider(''Refresh Rate (seconds)'', 10, 300, 60)

# Live Data and Predictions Loop
st.write(''### Current Live Matches and Predictions'')
         while True:
             live_data = fecth_live_data()
             live_data = make_parser(live_data)
             if not live_data.empty:
display_data = live_data[['HomeTeamName', 'AwayTeamName', 'HomeTeamScore', 'AwayTeamScore', 'PredictWinner']]
display_data['PredictedWinner'] = display_data['PredictWinner'].map({1: 'Home Win' , 0: 'Away Win'})
st.write(display_data)
else:
    st.write(''No live matches at the moment'')
    st.experimental_rerun()

    streamlit run dashboard.py
    st.bar_chart(live_data['PredictedWinner']. value_counts())

    sports-prediction-bot
    dashboard.py
    sports_predictor.pkl
    requirements.txt
    README.md

    # Initialize Git and link to Github
    git init
    git remote add origin
    https://GitHub.com/LeratoBriget/LeeSportsPredictor.git
    # Add files and commit
    git add .
    git commit -m ''Initial commit''
    # Push to Github
    git branch -M main
    git push -u origin main
                   
                   

