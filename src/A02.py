from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

#Establishing Target Variable
df = housing.frame
X = df.drop(columns= ['MedHouseVal'])
Y = df['MedHouseVal']

#Splitting into Train, Val, Test

# 80% of data goes to training
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=.2, random_state=42)

# Rest of the 20% is split amongst val and test
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=.5, random_state=42)

#Scaling Data for Neural Networks

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

#MLRegressor Model: 

mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(10,5),
                   max_iter=500,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) 

mlp.fit(X_train_scaled, Y_train)


# Actual vs. Predicted

train_preds = mlp.predict(X_train_scaled)
test_preds = mlp.predict(X_test_scaled)

# for Training: Actual vs. Predict Plot

plt.figure(figsize=(8, 6))
plt.scatter(x = Y_train, y = train_preds)
plt.plot([0, 7], [0, 7], '--k') # 45 degree line
plt.axis('tight')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Train Results')

plt.savefig('figs/Training_Act_vs_Pred.png')
plt.show()


# for Test: Actual vs. Predict Plot
plt.figure(figsize=(8, 6))
plt.scatter(x = Y_test, y = test_preds)
plt.plot([0, 7], [0, 7], '--k') # 45 degree line
plt.axis('tight')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Results')

plt.savefig('figs/Test_Act_vs_Pred.png')
plt.show()
