# Starbucks Calories Prediction — Streamlit Project

## Project idea
This app explores Starbucks nutrition data and predicts the estimated calories of a drink from:
- Beverage category
- Beverage name
- Preparation / size

The target column is `Calories`. The model is a Random Forest Regressor in a scikit-learn pipeline.

## Files
- `app.py` — Streamlit web app, data visualizations, prediction interface, and model evaluation.
- `ml.py` — corrected standalone exploratory data analysis (EDA) script.
- `starbucks.csv` — dataset.
- `requirements.txt` — Python packages needed.

## Run on your computer (Windows)
1. Install Python from https://www.python.org/downloads/ if it is not installed. During setup, tick **Add Python to PATH**.
2. Extract this ZIP file.
3. Open the extracted `starbucks_prediction_project` folder.
4. Click the folder address bar in File Explorer, type `cmd`, and press Enter.
5. Run these commands one by one:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

6. The app should open in your browser. If it does not, copy the local address shown in the terminal (usually `http://localhost:8501`) into your browser.

## Deploy online with Streamlit Community Cloud
1. Create/sign in to a GitHub account: https://github.com/
2. Create a **new public repository**, for example `starbucks-calories-prediction`.
3. Upload these three files to the repository root: `app.py`, `requirements.txt`, and `starbucks.csv`. You can also upload `ml.py` and this README.
4. Go to https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app** (or **Deploy an app**), select your repository and branch, and set the main file path to `app.py`.
6. Click **Deploy** and wait for installation to finish. Streamlit will provide a public app link.
7. If deployment fails, open the app's logs and check the first error message. Confirm that the CSV file is named exactly `starbucks.csv` and is in the same folder as `app.py`.

## How to explain the project to your teacher
“My project is a Starbucks Calories Prediction web app. I used a Starbucks nutrition dataset, cleaned the column names, explored the data with charts, and trained a Random Forest Regression model. The user selects a beverage category, drink name, and preparation type, and the app estimates the drink's calories. I evaluated it using a train-test split, Mean Absolute Error, and R² score. Since the dataset is small, predictions are educational estimates rather than guaranteed nutrition values.”

## Important limitations
- The supplied dataset has only 242 records, so this is a small classroom project.
- The model uses category/name/preparation, not the full nutrition profile. Its results are estimates and can be inaccurate for unseen drinks.
- Keep `app.py`, `requirements.txt`, and `starbucks.csv` in the same GitHub repository folder.
