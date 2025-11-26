import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

# --- Test hello world ---
print("hello-world.")

# --- Load dataset ---
df = pd.read_csv("dataset01.csv")  # make sure dataset01.csv is in the same folder

# --- Task 1: number of entries in 'y' ---
n_y = df['y'].count()
print("Number of entries in y:", n_y)

# --- Task 2: mean of 'y' ---
mean_y = df['y'].mean()
print("Mean of y:", mean_y)

# --- Task 3: standard deviation ---
std_y = df['y'].std()
print("Standard deviation of y:", std_y)

# --- Task 4: variance ---
var_y = df['y'].var()
print("Variance of y:", var_y)

# --- Task 5: min and max ---
min_y = df['y'].min()
max_y = df['y'].max()
print("Min of y:", min_y, "Max of y:", max_y)

# --- Task 6: OLS model ---
X = df[['x']]           # independent variable(s)
X = sm.add_constant(X)  # adds intercept
y = df['y']             # dependent variable
model = sm.OLS(y, X).fit()
print(model.summary())

# --- Save the model ---
with open("OLS_model.pkl", "wb") as f:
    pickle.dump(model, f)
