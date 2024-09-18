#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:50:10 2023

@author: austinmitchell
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import stats
import statistics as stat
from scipy.stats import linregress




data = pd.read_csv("abalone.csv")

d = pd.DataFrame(data)

column_names = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings"]
d.columns = column_names
print(column_names)
'''
column_names = d.columns
print(column_names)
print(d['Company\nLocation'])

mean_ratings = d.groupby('Company\nLocation')['Rating'].mean()

# Print the mean ratings
print("Mean Ratings by Country:")
print(mean_ratings)


usa_ratings = d[d['Company\nLocation'] == 'U.S.A.']['Rating']
france_ratings = d[d['Company\nLocation'] == 'France']['Rating']
canada_ratings = d[d['Company\nLocation'] == 'Canada']['Rating']

umean = usa_ratings.mean()
print(umean)

fmean = france_ratings.mean()
print(fmean)

cmean = canada_ratings.mean()
print(cmean)

# Perform ANOVA
f_statistic, p_value = stats.f_oneway(usa_ratings, france_ratings, canada_ratings)

# Print the results
print("ANOVA Results:")
print("F-statistic:", f_statistic)
print("P-value:", p_value)


column_name = 'Review\nDate'

low = d[column_name].min()
high = d[column_name].max()
print(low)
print(high)

f_statistic, p_value = stats.f_oneway(d[d['Review\nDate'] == 2006]['Rating'],
                                 d[d['Review\nDate'] == 2007]['Rating'],
                                 d[d['Review\nDate'] == 2008]['Rating'],
                                 d[d['Review\nDate'] == 2009]['Rating'],
                                 d[d['Review\nDate'] == 2010]['Rating'],
                                 d[d['Review\nDate'] == 2011]['Rating'],
                                 d[d['Review\nDate'] == 2012]['Rating'],
                                 d[d['Review\nDate'] == 2013]['Rating'],
                                 d[d['Review\nDate'] == 2014]['Rating'],
                                 d[d['Review\nDate'] == 2015]['Rating'],
                                 d[d['Review\nDate'] == 2016]['Rating'],
                                 d[d['Review\nDate'] == 2017]['Rating'])

print("F-statistic:", f_statistic)
print("P-value:", p_value)
'''

'''
# Extracting the variables
x = d['GNP']
y = d['Employed']

# Performing linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Print the results
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared value: {r_value**2}")
print(f"P-value: {p_value}")
print(f"Standard error: {std_err}")

# Plotting the regression line
plt.scatter(x, y, label='Data points')
plt.plot(x, intercept + slope * x, 'r', label='Regression line')
plt.xlabel('GNP')
plt.ylabel('Employed')
plt.legend()
plt.show()
'''
d['Sex'] = d['Sex'].astype('category').cat.codes


X = d.drop('Rings', axis=1)
y = d['Rings']

# Create and train the linear regression model
slope, intercept, r_value, p_value, std_err = linregress(X['Length'], y)

# Make predictions on the test set
y_pred = intercept + slope * X['Length']

# Calculate and print the mean squared error
mse = np.mean((y_pred - y) ** 2)
print(f"Mean Squared Error: {mse}")

# Optionally, plot the actual vs predicted values
plt.scatter(X['Length'], y, label='Actual Rings')
plt.plot(X['Length'], y_pred, color='red', label='Predicted Rings')
plt.xlabel("Length")
plt.ylabel("Rings")
plt.title("Actual vs Predicted Rings")
plt.legend()
plt.show()



