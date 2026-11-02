import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Starbucks Calories Predictor", page_icon="☕", layout="wide")
st.title("☕ Starbucks Calories Prediction")
st.write("Explore Starbucks nutrition data and estimate a drink's calories using its category, name, and preparation type.")

@st.cache_data
def load_data():
    data = pd.read_csv("starbucks.csv")
    data.columns = data.columns.str.strip()
    needed = ["Calories", "Beverage_category", "Beverage", "Beverage_prep"]
    data["Calories"] = pd.to_numeric(data["Calories"], errors="coerce")
    data = data.dropna(subset=needed).copy()
    for col in ["Beverage_category", "Beverage", "Beverage_prep"]:
        data[col] = data[col].astype(str).str.strip()
    return data

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Make sure starbucks.csv is in the same GitHub folder as app.py.")
    st.stop()

features = ["Beverage_category", "Beverage", "Beverage_prep"]
X, y = df[features], df["Calories"]
preprocessor = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), features)])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

tab1, tab2, tab3 = st.tabs(["📊 Explore Data", "🔮 Predict Calories", "ℹ️ About"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Drink categories", f"{df['Beverage_category'].nunique():,}")
    c3.metric("Average calories", f"{df['Calories'].mean():.1f}")
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Average calories by beverage category")
    averages = df.groupby("Beverage_category", as_index=False)["Calories"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=averages, x="Calories", y="Beverage_category", ax=ax)
    ax.set_xlabel("Average calories")
    ax.set_ylabel("Beverage category")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Missing values")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)

with tab2:
    st.info("This is an educational estimate based on a small dataset; it may not match official nutrition information.")
    category = st.selectbox("Beverage category", sorted(df["Beverage_category"].unique()))
    drinks = sorted(df.loc[df["Beverage_category"] == category, "Beverage"].unique())
    beverage = st.selectbox("Beverage", drinks)
    preps = sorted(df.loc[(df["Beverage_category"] == category) & (df["Beverage"] == beverage), "Beverage_prep"].unique())
    prep = st.selectbox("Preparation / size", preps)
    if st.button("Predict calories", type="primary"):
        row = pd.DataFrame([{"Beverage_category": category, "Beverage": beverage, "Beverage_prep": prep}])
        estimate = max(0, float(model.predict(row)[0]))
        st.success(f"Estimated calories: **{estimate:.0f} kcal**")
        matched = df[(df["Beverage_category"] == category) & (df["Beverage"] == beverage) & (df["Beverage_prep"] == prep)]
        if not matched.empty:
            st.caption(f"Dataset recorded average for this selection: {matched['Calories'].mean():.0f} kcal.")
    st.subheader("Model evaluation")
    m1, m2 = st.columns(2)
    m1.metric("Mean Absolute Error", f"{mae:.1f} calories")
    m2.metric("R² score", f"{r2:.3f}")
    st.caption("Scores use a random 80/20 train-test split and are only a classroom demonstration.")

with tab3:
    st.markdown("""
    **Problem:** Estimate calories for a Starbucks drink.

    **Inputs:** Beverage category, beverage name, and preparation / size.  
    **Target:** Calories.  
    **Algorithm:** Random Forest Regression with one-hot encoding.  
    **Dataset:** The supplied Starbucks nutrition CSV.

    **Limitation:** The dataset is small. Predictions for unseen drinks may be inaccurate and should not replace official nutrition information.
    """)
