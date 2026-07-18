import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Student Performance AI", page_icon="🎓")

st.title("🎓 Student Performance AI Predictor")

# Load Dataset
12 df = pd.read_csv("student_data.csv")
13 df.columns = df.columns.str.strip()
14 df["reading score"] = pd.to_numeric(df["reading score"], errors="coerce")
15 df["writing score"] = pd.to_numeric(df["writing score"], errors="coerce")
16 df["math score"] = pd.to_numeric(df["math score"], errors="coerce")
17 df = df.dropna()

st.subheader("Dataset Preview")
st.dataframe(df)

# Encode categorical columns
data = df.copy()
encoders = {}

for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

27 X = data.drop("math score", axis=1)
28 y = data["math score"]
29 X = X.astype(float)
30 y = y.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

st.success(f"Model Accuracy : {round(r2_score(y_test, pred)*100,2)}%")

st.subheader("Predict Student Score")

gender = st.selectbox("Gender", df["gender"].unique())
race = st.selectbox("Race", df["race/ethnicity"].unique())
parent = st.selectbox(
    "Parental Education",
    df["parental level of education"].unique()
)
lunch = st.selectbox("Lunch", df["lunch"].unique())
prep = st.selectbox(
    "Test Preparation",
    df["test preparation course"].unique()
)

reading = st.number_input(
    "Reading Score",
    min_value=0,
    max_value=100,
    value=70
)

writing = st.number_input(
    "Writing Score",
    min_value=0,
    max_value=100,
    value=70
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parent],
        "lunch": [lunch],
        "test preparation course": [prep],
        "reading score": [reading],
        "writing score": [writing]
    })

    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    result = model.predict(input_data)

    st.success(
        f"🎯 Predicted Math Score : {round(result[0],2)}"
    )
