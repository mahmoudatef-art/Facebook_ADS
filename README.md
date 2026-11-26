# AdPulse - Click Prediction Studio

Interactive Streamlit app that predicts whether a user will click a Facebook ad.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Keep `facebook_ads.csv` next to `app.py`. If `best_model.pkl` and `scaler.pkl` exist (from the notebook), they appear as an extra model in the app.

## Deploy
Push to GitHub, then deploy for free on https://share.streamlit.io (pick the repo and `app.py`).
