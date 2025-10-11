import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# NOTE: Your custom Logistic Regression class MUST be defined here
# so that pickle knows how to save the object.
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

# 1. Load Data
try:
    # Make sure the CSV file is named correctly and is in the same folder
    dataset = pd.read_csv("diabetes.csv") # Make sure your CSV is named this or change it
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please check the file name and path.")
    exit()

# 2. Separate Features and Target
features = dataset.drop(columns='Outcome', axis=1)
target = dataset['Outcome']

# 3. Standardize the data
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)
features = standardized_data

# 4. Split Data
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# 5. Train Model
model = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X_train, Y_train)

# 6. Save the Model AND the Scaler
pickle.dump(model, open("trained_model.sav", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Success: 'trained_model.sav' and 'scaler.pkl' have been created.")