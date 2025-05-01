Diabetes Checkup Application
This is an interactive Diabetes Prediction web application developed using Streamlit, Pandas, and Scikit-learn. The app allows users to input patient data and predicts whether the patient is diabetic using a trained machine learning model (Random Forest Classifier). It also provides visualization tools to compare patient data with existing training data.
Features
- Patient Data Input: Input patient details like pregnancies, glucose level, blood pressure, BMI, etc., via an intuitive sidebar.
- Diabetes Prediction: The app predicts whether a patient is diabetic or not based on the provided details using a machine learning model.
- Training Data Statistics: Displays statistics for the training dataset to give an overview of its characteristics.
- Interactive Visualizations: Generates scatter plots comparing user data with training data for factors like age, blood pressure, glucose level, BMI, and more.
- Model Accuracy: Shows the prediction accuracy of the trained model.

Installation
- Clone the repository:git clone https://github.com/ssinha8752/ML_Diabetes_Prediction.git
cd diabetes-checkup

- Install the required libraries:pip install streamlit pandas sklearn matplotlib plotly seaborn

- Add the dataset diabetes.csv to the project folder.

Usage
- Run the application:streamlit run app.py

- Open the app in your browser (usually at http://localhost:8501).
- Adjust patient parameters using the sidebar and view the predictions and visualizations.

File Structure
- app.py: Main application script.
- diabetes.csv: Training dataset for the machine learning model.
- README.md: Documentation for the project.

Data Source
The dataset diabetes.csv used in this project contains patient data and their diabetes diagnosis outcomes. The features include:
- Pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI
- Diabetes pedigree function
- Age

The target variable is Outcome, where 0 represents a healthy individual, and 1 represents a diabetic individual.
Model Details
The app utilizes a Random Forest Classifier for prediction. The data is split into training (80%) and testing (20%) subsets. Model accuracy is displayed in the app and calculated using the testing data.
Visualizations
The app includes the following interactive graphs to analyze the patient's data against the training dataset:
- Age vs Pregnancies
- Age vs Glucose
- Age vs Blood Pressure
- Age vs Skin Thickness
- Age vs Insulin
- Age vs BMI
- Age vs Diabetes Pedigree Function (DPF)

Example Output
- Input Patient Data: Sidebar sliders allow users to input patient data.
- Prediction: "You are not Diabetic" or "You are Diabetic" displayed with visualizations and accuracy.

Screenshots


Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements.
