import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

boston_df = pd.read_csv("Boston_Housing_Data.csv")

print("Linear Regression on Boston Housing Dataset")

print(boston_df.columns)

X = boston_df[['rm']]  
y = boston_df['medv']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

LR_model = LinearRegression()
LR_model.fit(X_train, y_train)

y_pred = LR_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

plt.scatter(X_test, y_test, color='green', label='Actual')

sorted_indices = X_test.squeeze().argsort()

plt.plot(
    X_test.iloc[sorted_indices],
    y_pred[sorted_indices],
    color='red',
    label='Predicted'
)

plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Price (MEDV)")
plt.title("Linear Regression on Boston Housing Dataset")
plt.legend()
plt.show()
