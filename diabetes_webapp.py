


# ai code

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


# Load the saved model and the scaler
try:
    # This relative path works everywhere, including on Streamlit Cloud
    loaded_model = pickle.load(open("trained_model.sav", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or scaler files not found. Please run the training script first to generate these files.")
    st.stop()
    
import pickle

# After training your model
pickle.dump(loaded_model, open('trained_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))


# Prediction function
def diabetes_prediction(input_data):
    """
    Takes user input, scales it using the pre-fitted scaler,
    and returns the model's prediction.
    """
    # Change the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # **CRITICAL STEP**: Standardize the reshaped data using the loaded scaler
    std_data = scaler.transform(input_data_reshaped)

    # Make the prediction
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return "The person is NOT diabetic"
    else:
        return "The person IS diabetic"


# Main function for the Streamlit UI
def main():
    """
    Defines the Streamlit user interface and handles user interaction.
    """
    # Giving a title
    st.title("Diabetes Prediction ML Model")

    # Getting the input data from the user in columns for a better layout
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        Glucose = st.text_input("Glucose Level")
        BloodPressure = st.text_input("Blood Pressure value")
        SkinThickness = st.text_input("Skin Thickness value")

    with col2:
        Insulin = st.text_input("Insulin Level")
        BMI = st.text_input("BMI value")
        DiabetesPedigreeFunction = st.text_input(
            "Diabetes Pedigree Function value")
        Age = st.text_input("Age of the Person")

    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        # The prediction logic should be inside the button's "if" block
        try:
            # Convert all text inputs to floating-point numbers
            input_data_as_float = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]

            # Call the prediction function with the list of numbers
            diagnosis = diabetes_prediction(input_data_as_float)
            st.success(diagnosis)

        except ValueError:
            # If conversion to float fails (e.g., user enters 'abc' or leaves a field empty)
            st.error("Error: Please enter valid numbers for all fields.")
        except Exception as e:
            # Catch any other potential errors
            st.error(f"An error occurred during prediction: {e}")


# This makes the script runnable
if __name__ == "__main__":
    main()

