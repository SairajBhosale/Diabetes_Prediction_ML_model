# Diabetes Prediction with Custom Logistic Regression & Streamlit

This project is a comprehensive Machine Learning workflow that predicts whether a patient is diabetic based on diagnostic measures. It features a **custom implementation of Logistic Regression from scratch** (using Gradient Descent) and includes a **Streamlit web application** for real-time predictions.

##  Tech Stack
* **Python**: Core programming language.
* **Pandas & NumPy**: For data manipulation and matrix operations.
* **Scikit-Learn**: Used for `StandardScaler`, `train_test_split`, and metrics.
* **Streamlit**: For building the interactive web interface.
* **Pickle**: For saving and loading the trained model.

##  Dataset Overview
The dataset contains medical details for female patients. Key features include:

| Feature | Description |
| :--- | :--- |
| **Pregnancies** | Number of times pregnant |
| **Glucose** | Plasma glucose concentration |
| **BloodPressure** | Diastolic blood pressure (mm Hg) |
| **SkinThickness** | Triceps skin fold thickness (mm) |
| **Insulin** | 2-Hour serum insulin (mu U/ml) |
| **BMI** | Body mass index |
| **DiabetesPedigreeFunction** | Diabetes pedigree function |
| **Age** | Age (years) |
| **Outcome** | Target variable (0 = Non-Diabetic, 1 = Diabetic) |

##  Model Implementation
Unlike standard projects that strictly use libraries, this project implements **Logistic Regression from scratch** to demonstrate the underlying mathematics:
* **Optimization Algorithm**: Gradient Descent.
* **Activation Function**: Sigmoid Function.
* **Key Methods**: `fit()` (training loop) and `predict()` (inference).

## Project Link
