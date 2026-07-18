import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f172a,#2563eb,#7c3aed);
}

h1,h2,h3,p,label {
color:white !important;
}

.box{
background:rgba(255,255,255,0.15);
padding:20px;
border-radius:20px;
}

.stButton button{
background:#06b6d4;
color:white;
border-radius:15px;
padding:10px 25px;
}

</style>
""",unsafe_allow_html=True)



st.markdown(
"""
<div class="box">

<h1>🎓 Student Performance AI Predictor</h1>

<h3>
Machine Learning Based Student Score Prediction
</h3>

</div>
""",
unsafe_allow_html=True
)



@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")


df = load_data()


st.subheader("📊 Dataset Preview")

st.dataframe(
    df,
    use_container_width=True
)



data=df.copy()


encoder=LabelEncoder()


for col in data.columns:

    if data[col].dtype=="object":

        data[col]=encoder.fit_transform(data[col])



X=data.drop(
    "math score",
    axis=1
)


y=data["math score"]



X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model=RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)



prediction=model.predict(
    X_test
)



accuracy=r2_score(
    y_test,
    prediction
)


error=mean_absolute_error(
    y_test,
    prediction
)



st.subheader("🤖 Model Performance")


col1,col2=st.columns(2)


with col1:
    st.metric(
        "Accuracy",
        str(round(accuracy*100,2))+"%"
    )


with col2:
    st.metric(
        "Error",
        round(error,2)
    )



st.subheader("🔮 Predict Student Score")


inputs={}


for col in X.columns:

    inputs[col]=st.number_input(
        col,
        min_value=0,
        max_value=100,
        value=50
    )


input_df=pd.DataFrame(
    [inputs]
)



if st.button("Predict Score"):

    result=model.predict(
        input_df
    )

    st.success(
        "Predicted Math Score : "
        +str(round(result[0],2))
    )