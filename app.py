import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Student Performance AI", page_icon="🎓")

st.title("🎓 Student Performance AI Predictor")

# Load Dataset
df = pd.read_csv("student_data.csv")
df.columns = df.columns.str.strip()
df = df.dropna()

st.subheader("Dataset Preview")
st.dataframe(df)

# Encode Categorical Columns
data = df.copy()

encoders = {}

for col in [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Convert score columns to numeric
for col in ["reading score", "writing score", "math score"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

# Features & Target
X = data[
    [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "reading score",
        "writing score",
    ]
]

y = data["math score"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = r2_score(y_test, pred)

st.success(f"Model Accuracy : {round(accuracy*100,2)} %")

st.subheader("Predict Student Math Score")

gender = st.selectbox("Gender", df["gender"].unique())

race = st.selectbox(
    "Race/Ethnicity",
    df["race/ethnicity"].unique()
)

parent = st.selectbox(
    "Parental Education",
    df["parental level of education"].unique()
)

lunch = st.selectbox(
    "Lunch",
    df["lunch"].unique()
)

prep = st.selectbox(
    "Test Preparation",
    df["test preparation course"].unique()
)

reading = st.slider(
    "Reading Score",
    0,
    100,
    70
)

writing = st.slider(
    "Writing Score",
    0,
    100,
    70
)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "gender":[gender],
        "race/ethnicity":[race],
        "parental level of education":[parent],
        "lunch":[lunch],
        "test preparation course":[prep],
        "reading score":[reading],
        "writing score":[writing]
    })

    for col in encoders:
        input_df[col] = encoders[col].transform(
            input_df[col].astype(str)
        )

    prediction = model.predict(input_df)

    st.success(
        f"🎯 Predicted Math Score : {round(prediction[0],2)}"
    )
