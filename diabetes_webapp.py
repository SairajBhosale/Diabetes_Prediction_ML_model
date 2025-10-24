import streamlit as st
import numpy as np
import pickle

# IMPORTANT: The custom class definition must be here for pickle to load the model
class Logistic_Regression():
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_hat = 1 / (1 + np.exp(- (self.X.dot(self.w) + self.b)))
        dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))
        db = (1/self.m)*np.sum(Y_hat - self.Y)
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        Y_pred = 1 / (1 + np.exp(- (X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred

# Load ONLY the saved model
try:
    loaded_model = pickle.load(open("trained_model.sav", "rb"))
except FileNotFoundError:
    st.error("Model file not found. Please run the 'train_model_unscaled.py' script first.")
    st.stop()

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # --- SCALING STEP HAS BEEN REMOVED ---
    # The prediction is now made on the raw, reshaped data
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

# Main function for the Streamlit UI
def main():
    st.title("Diabetes Prediction ML Model")
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        Glucose = st.text_input("Glucose Level")
        BloodPressure = st.text_input("Blood Pressure value")
        SkinThickness = st.text_input("Skin Thickness value")
    with col2:
        Insulin = st.text_input("Insulin Level")
        BMI = st.text_input("BMI value")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
        Age = st.text_input("Age of the Person")

    if st.button('Diabetes Test Result'):
        try:
            input_data_as_float = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            diagnosis = diabetes_prediction(input_data_as_float)
            st.success(diagnosis)
        except ValueError:
            st.error("Error: Please enter valid numbers for all fields.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


st.text("NOTE : The data on which this ML model has been trained on might not be true and could have faults")
st.link_button("https://www.dropbox.com/scl/fi/0uiujtei423te1q4kvrny/diabetes.csv?rlkey=20xvytca6xbio4vsowi2hdj8e&e=2&st=e9cxl0w0&dl=0")

if __name__ == "__main__":
    main()











