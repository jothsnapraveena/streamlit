import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    wine=load_wine()
    df=pd.DataFrame(wine.data,columns=wine.feature_names)
    df['wine_class']=wine.target
    return df,wine.target_names

df,target_names=load_data()
df.rename(columns={'od280/od315_of_diluted_wines': 'od280_od315_of_diluted_wines'}, inplace=True)
model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['wine_class'])

# Sidebar title
st.sidebar.title("Input Features")

# Adding sliders 
alcohol = st.sidebar.slider('Alcohol', min_value=12.0, max_value=15.0)
malic_acid = st.sidebar.slider('Malic Acid', min_value=0.5, max_value=3.0)
ash = st.sidebar.slider('Ash', min_value=1.5, max_value=3.0)
alcalinity_of_ash = st.sidebar.slider('Alcalinity of Ash', min_value=10.0, max_value=30.0)
magnesium = st.sidebar.slider('Magnesium', min_value=80.0, max_value=160.0)
total_phenols = st.sidebar.slider('Total Phenols', min_value=2.0, max_value=4.0)
flavanoids = st.sidebar.slider('Flavanoids', min_value=1.0, max_value=4.0)
nonflavanoid_phenols = st.sidebar.slider('Nonflavanoid Phenols', min_value=0.1, max_value=0.5)
proanthocyanins = st.sidebar.slider('Proanthocyanins', min_value=1.0, max_value=4.0)
color_intensity = st.sidebar.slider('Color Intensity', min_value=2.0, max_value=8.0)
hue = st.sidebar.slider('Hue', min_value=0.8, max_value=1.2)
od280_od315_of_diluted_wines = st.sidebar.slider('OD280_OD315 of Diluted Wines', min_value=2.5, max_value=4.0)
proline = st.sidebar.slider('Proline', min_value=500.0, max_value=1500.0)

input_data = [[alcohol,
    malic_acid,
    ash,
    alcalinity_of_ash,
    magnesium,
    total_phenols,
    flavanoids,
    nonflavanoid_phenols,
    proanthocyanins,
    color_intensity,
    hue,
    od280_od315_of_diluted_wines,
    proline
]]

##Prediction
prediction=model.predict(input_data)
predicted_wine_class=target_names[prediction[0]]

st.write("Prediction")
st.write(f"The predicted class is: {predicted_wine_class}")