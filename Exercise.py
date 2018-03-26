#import required libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy
import statsmodels.api as sm

#create new data frame for temperature and beer consumption
tb = pd.read_csv("Temp_and_Beer_Consump.csv")

#generate and display descriptive statistics for temperature and beer consumption
print("** Descriptive statistics for temperature **")
print(tb["Temp"].describe())
print("\n** Descriptive statistics for beer consumption **")
print(tb["Consumption"].describe())

#generate and display histogram for beer consumption
plt.hist(x=tb["Consumption"], bins=20)
plt.xlabel("Beer Consumption")
plt.ylabel("Frequency")
plt.title("Average Beer Consumption")
plt.show()

#generate and display histogram for temperature
plt.hist(x=tb["Temp"], bins=20)
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.title("Average Temperature")
plt.show()

#generate and display scatterplot for both variables with labels
plt.scatter(tb["Temp"], tb["Consumption"])
plt.xlabel("Temperature")
plt.ylabel("Beer Consumption")
plt.title("Beer Consumption per Capita by Average Temperature")
plt.show()

#conduct and display pearson correlation analysis
print("\n** Pearson Correlation Coefficient **")
print(pearsonr(tb["Temp"], tb["Consumption"]))

#create new variables that will store independent and dependent variable values
y = tb["Consumption"]
x = tb["Temp"]
x = sm.add_constant(x)

#create the model
mod = sm.OLS(y,x)

#estimate the model fit
results = mod.fit()

#display the summarized results
print("\n** Linear Regression Model **")
print(results.summary())