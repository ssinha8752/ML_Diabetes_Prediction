import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('diabetes.csv')

st.title('Diabetes Prediction App')

# Display Training Data Summary
st.subheader('Training Data Statistics')
st.write(df.describe())

# Split Features & Labels
x = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

# Sidebar Input Collection
features = {
    'Pregnancies': (0, 17, 3),
    'Glucose': (0, 200, 120),
    'BloodPressure': (0, 122, 70),
    'SkinThickness': (0, 100, 20),
    'Insulin': (0, 846, 79),
    'BMI': (0, 67, 20),
    'DiabetesPedigreeFunction': (0.0, 2.4, 0.47),
    'Age': (21, 88, 33)
}
user_report = {feat: st.sidebar.slider(feat, *limits) for feat, limits in features.items()}
user_data = pd.DataFrame(user_report, index=[0])

# Train Model
rf = RandomForestClassifier(random_state=42, n_estimators=200)
rf.fit(x_train, y_train)

# Predict User Data
user_result = rf.predict(user_data)

# Set Color for Visualization
color = 'blue' if user_result[0] == 0 else 'red'

# Visualization
st.subheader('Comparative Analysis')

for feature in features.keys():
    fig = plt.figure()
    ax = sns.scatterplot(x='Age', y=feature, data=df, hue='Outcome', palette='coolwarm')
    sns.scatterplot(x=user_data['Age'], y=user_data[feature], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.title(f"{feature} vs Age (0 - Healthy, 1 - Diabetic)")
    st.pyplot(fig)

# Display Prediction Result
st.subheader('Your Health Report:')
if user_result[0] == 0 :
    st.markdown(f"<h2 style='color: {color};'>Not Diabetic</h2>", unsafe_allow_html=True)
else:
    st.markdown(f"<h2 style='color: {color};'>Diabetic</h2>", unsafe_allow_html=True)

# Display Model Accuracy
st.subheader('Model Performance:')
st.write(f"Accuracy: {accuracy_score(y_test, rf.predict(x_test))*100:.2f}%")
