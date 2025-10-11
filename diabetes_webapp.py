
'''
import streamlit as st
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# %%
from sklearn.linear_model import LogisticRegression

# %%


class Logistic_Regression():

    # declaring learning rate & number of iterations (Hyperparametes)
    def __init__(self, learning_rate, no_of_iterations):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    # fit function to train the model with dataset

    def fit(self, X, Y):

        # number of data points in the dataset (number of rows)  -->  m
        # number of input features in the dataset (number of columns)  --> n
        self.m, self.n = X.shape

        # initiating weight & bias value

        self.w = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # implementing Gradient Descent for Optimization

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):

        # Y_hat formula (sigmoid function)

        Y_hat = 1 / (1 + np.exp(- (self.X.dot(self.w) + self.b)))

        # derivatives

        dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))

        db = (1/self.m)*np.sum(Y_hat - self.Y)

        # updating the weights & bias using gradient descent

        self.w = self.w - self.learning_rate * dw

        self.b = self.b - self.learning_rate * db

    # Sigmoid Equation & Decision Boundary

    def predict(self, X):

        Y_pred = 1 / (1 + np.exp(- (X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred


# %%
dataset = pd.read_csv(r"C:\Users\Sairaj\Downloads\diabetes (1).csv")

# %%
dataset.head()

# %%
dataset.tail()

# %%
dataset.shape

# %%
dataset.isnull().sum()

# %%
# getting the statistical measures of the data
dataset.describe()

# %%
dataset["Outcome"].value_counts()

# %%
dataset.groupby("Outcome").mean()

# %%
# separating the data and labels
features = dataset.drop(columns='Outcome', axis=1)
target = dataset['Outcome']

# %%
print(features)

# %%
print(target)

# %%
scaler = StandardScaler()

# %%
scaler.fit(features)

# %%
standardized_data = scaler.transform(features)


# %%
print(standardized_data)

# %%
features = standardized_data
target = dataset['Outcome']

# %%
print(features)
print(target)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.2, random_state=2)

# %%
print(features.shape, X_train.shape, X_test.shape)

# %%
model = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)

# %%
model.fit(X_train, Y_train)

# %%
# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# %%
print('Accuracy score of the training data : ', training_data_accuracy)

# %%
# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# %%
print('Accuracy score of the test data : ', test_data_accuracy)

# %%
# Define the diabetes class
class diabetes():
    def __init__(self):
        self.Pregnancies = int(input("Enter the number of times pregnant: "))
        self.Glucose = int(input("Enter the glucose level: "))
        self.BloodPressure = int(input("Enter the blood pressure: "))
        self.SkinThickness = int(input("Enter the skin thickness: "))
        self.Insulin = int(input("Enter the insulin level: "))
        self.BMI = float(input("Enter the BMI: "))
        self.DiabetesPedigreeFunction = float(input("Enter the diabetes pedigree function: "))
        self.Age = int(input("Enter the age: "))

# Create an instance and collect input
patient = diabetes()

# Convert input to numpy array
diabetes_as_numpy_array = np.asarray([
    patient.Pregnancies,
    patient.Glucose,
    patient.BloodPressure,
    patient.SkinThickness,
    patient.Insulin,
    patient.BMI,
    patient.DiabetesPedigreeFunction,
    patient.Age
])

# Reshape the array for prediction
diabetes_reshaped = diabetes_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(diabetes_reshaped)
print(std_data)

# Make prediction
prediction = model.predict(std_data)
print(prediction)

# Output result
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')

# %% [markdown]
# SAVING THE TRAINED MODEL
#

# %%

# %%
filename = "trained_model.sav"
pickle.dump(model, open(filename, "wb"))

# %%
# Loading the saved model
loaded_model = pickle.load(open("trained_model.sav", "rb"))

# %%


def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


# %%

# %%

def main():

    # giving a title
    st.title("Diabetes Predicition ML model")

    # getting the input data from the user
    Pregnancies = st.text_input("Enter the number of preganancies: ")
    Glucose = st.text_input("Enter your Glucose level: ")
    BloodPressure = st.text_input("Enter your Blood Pressure: ")
    SkinThickness = st.text_input("Enter your Skin Thickness: ")
    Insulin = st.text_input("Enter your Insulin Level: ")
    BMI = st.text_input("Enter your BMI: ")
    DiabetesPedigreeFunction = st.text_input(
        "Enter the Diabetes Predigree Function: ")
    Age = st.text_input("Enter your Age: ")

    # code for prediction
    diagnosis = " "

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = ''  # Initialize an empty string for the result
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
        # If conversion to float fails (e.g., user enters 'abc')
        st.error("Error: Please enter valid numbers for all fields.")
    except Exception as e:
        # Catch any other potential errors
        st.error(f"An error occurred during prediction: {e}")

    # Call the correct function with the single list of inputs
    diagnosis = diabetes_prediction(input_data_as_float)

    st.success(diagnosis)


if __name__ == "__main__":
    main()
'''


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
