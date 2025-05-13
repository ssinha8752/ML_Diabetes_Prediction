import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- DATA LOADING & MODEL TRAINING ---

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

@st.cache_resource
def train_model(data):
    x = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, x.columns.tolist(), accuracy

# --- VISUALIZATION FUNCTIONS ---

def plot_feature_importance(model, feature_columns):
    st.subheader("\U0001F4CA Feature Importance")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    fig = plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
    plt.title("Random Forest Feature Importance")
    st.pyplot(fig)

def plot_outcome_distribution(df):
    st.subheader("\U0001F9EE Diabetes Outcome Distribution")
    fig = plt.figure(figsize=(5, 4))
    sns.countplot(x='Outcome', data=df, palette="coolwarm")
    plt.xticks([0, 1], ['Not Diabetic', 'Diabetic'])
    plt.title("Outcome Class Distribution")
    st.pyplot(fig)

def plot_correlation_heatmap(df):
    st.subheader("\U0001F9E0 Feature Correlation Heatmap")
    fig = plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    st.pyplot(fig)

def boxplots_by_outcome(df, features):
    st.subheader("\U0001F4E6 Feature Distribution by Outcome")
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(features):
                feature = features[i + j]
                with cols[j]:
                    fig = plt.figure(figsize=(4, 3))
                    sns.boxplot(x='Outcome', y=feature, data=df, palette='coolwarm')
                    plt.xticks([0, 1], ['Not Diabetic', 'Diabetic'])
                    plt.title(f"{feature} Distribution by Outcome")
                    st.pyplot(fig)

# --- SINGLE PATIENT ENTRY ---

def single_patient_input(feature_columns):
    st.header("\U0001F9FE Enter Patient Data")
    with st.form("patient_form"):
        user_input = {}
        for col in feature_columns:
            if col == "DiabetesPedigreeFunction":
                val = st.number_input(f"{col}", min_value=0.0, value=0.0, step=0.01)  # Allows decimal values
            else:
                val = st.number_input(f"{col}", min_value=0, value=0, step=1)  # Ensures integer input
            user_input[col] = val
        submitted = st.form_submit_button("Predict")
    return pd.DataFrame([user_input]) if submitted else None

# --- VISUALIZE SINGLE PATIENT ---

def visualize_patient(df, patient_df, features, result):
    st.subheader("Comparative Analysis")
    color = 'red' if result == 1 else 'blue'
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(features):
                feature = features[i + j]
                with cols[j]:
                    fig = plt.figure(figsize=(4, 3))
                    sns.scatterplot(x='Age', y=feature, data=df, hue='Outcome', palette='coolwarm', alpha=0.6)
                    sns.scatterplot(x=patient_df['Age'], y=patient_df[feature], color=color, s=100)
                    plt.title(f"{feature} vs Age")
                    st.pyplot(fig)

# --- BATCH UPLOAD & VISUALIZATION ---

def batch_prediction(model, features, df):
    empty_df = pd.DataFrame(columns=features)
    csv = empty_df.to_csv(index=False).encode("utf-8")

    # Provide a downloadable empty CSV template
    st.download_button(
        label="📥 Download Empty CSV Template",
        data=csv,
        file_name="diabetes_input_template.csv",
        mime="text/csv"
    )


    uploaded_file = st.file_uploader("\U0001F4C2 Upload CSV for Batch Prediction", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if not set(features).issubset(batch_df.columns):
            st.error(f"CSV must include: {', '.join(features)}")
            return

        # Predict and show table
        preds = model.predict(batch_df[features])
        batch_df['Prediction'] = preds
        st.subheader("\U0001F4CB Batch Predictions")
        st.dataframe(batch_df)

        # Dropdown for selecting patient (with "All" option)
        selection_options = ["All"] + list(batch_df.index)
        selected = st.selectbox("\U0001F50E Select a patient to visualize", selection_options, format_func=lambda x: f"Patient {x + 1}" if isinstance(x, int) else x)

        if selected == "All":
            st.markdown("### \U0001F9EA Visualizing all patients")
            for i in range(0, len(features), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(features):
                        feature = features[i + j]
                        with cols[j]:
                            fig = plt.figure(figsize=(4, 3))
                            sns.scatterplot(x='Age', y=feature, data=df, hue='Outcome', palette='coolwarm', alpha=0.5)
                            sns.scatterplot(x=batch_df['Age'], y=batch_df[feature], hue=batch_df['Prediction'],
                                            palette={0: 'blue', 1: 'red'}, s=80, alpha=0.8, legend=False)
                            plt.title(f"{feature} vs Age - All Patients")
                            st.pyplot(fig)
        else:
            selected_row = batch_df.loc[selected]
            st.markdown(f"### Patient {selected + 1}: {'\U0001F7E5 Diabetic' if selected_row['Prediction'] else '\U0001F7E6 Not Diabetic'}")
            color = 'red' if selected_row['Prediction'] else 'blue'

            for i in range(0, len(features), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(features):
                        feature = features[i + j]
                        with cols[j]:
                            fig = plt.figure(figsize=(4, 3))
                            sns.scatterplot(x='Age', y=feature, data=df, hue='Outcome', palette='coolwarm', alpha=0.6)
                            sns.scatterplot(x=[selected_row['Age']], y=[selected_row[feature]], color=color, s=100)
                            plt.title(f"{feature} vs Age")
                            st.pyplot(fig)

# --- MAIN APP ---

st.set_page_config(page_title="Diabetes Predictor", layout="wide")
st.title("\U0001FA7A Diabetes Prediction & Visualization App")

df = load_data()
model, feature_columns, model_accuracy = train_model(df)

# --- Sidebar and Visualization Options ---
mode = st.sidebar.radio("Choose Mode", ["Single Entry", "Batch Upload"])
st.sidebar.markdown(f"**Model Accuracy:** `{model_accuracy*100:.2f}%`")

st.sidebar.markdown("---")
st.sidebar.markdown("## EDA Visualizations")
if st.sidebar.checkbox("Show Feature Importance"):
    plot_feature_importance(model, feature_columns)
if st.sidebar.checkbox("Show Outcome Distribution"):
    plot_outcome_distribution(df)
if st.sidebar.checkbox("Show Correlation Heatmap"):
    plot_correlation_heatmap(df)
if st.sidebar.checkbox("Show Boxplots by Outcome"):
    boxplots_by_outcome(df, feature_columns)

# --- App Modes ---
if mode == "Single Entry":
    input_df = single_patient_input(feature_columns)
    if input_df is not None:
        prediction = model.predict(input_df)[0]
        st.subheader("Prediction:")
        st.markdown(f"<h2 style='color:{'red' if prediction else 'blue'};'>{'Diabetic' if prediction else 'Not Diabetic'}</h2>", unsafe_allow_html=True)
        visualize_patient(df, input_df, feature_columns, prediction)
else:
    batch_prediction(model, feature_columns, df)
