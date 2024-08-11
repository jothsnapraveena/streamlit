import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
st.title("Multi Classification of IRIS Flower Species")
def load_data():
    iris=load_iris()
    df=pd.DataFrame(data=iris,columns=iris.feature_names)
    df['species']=iris.target
    return df,iris.target_names

df,target_names=load_data()



model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species'])

#sidebar
st.sidebar.title("Input Features")

#adding sliders

sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=4.3, max_value=7.9)
sepal_width = st.sidebar.slider('Sepal Width (cm)', min_value=2.0, max_value=4.4)
petal_length = st.sidebar.slider('Petal Length (cm)', min_value=1.0, max_value=6.9)
petal_width = st.sidebar.slider('Petal Width (cm)', min_value=0.1, max_value=2.5)

input_data=[[sepal_length,sepal_width,petal_length,petal_width]]

#prediction
prediction=model.predict(input_data)
predicted_species=target_names[prediction[0]]
prediction_proba = model.predict_proba(input_data)

st.subheader('prediction')
st.write(f'The predicted species is : {predicted_species}')
st.subheader("Prediction Probability")
# Create a DataFrame to map class names with probabilities
proba_df = pd.DataFrame(prediction_proba, columns=target_names)

# Display probabilities with class names
st.write(proba_df)
