# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:00:08 2022

@author: Kshitij Tripathi
"""

import pandas as pd;
import numpy as np;
from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;
from sklearn.model_selection import train_test_split;
import statsmodels.api as sm;
import matplotlib.pyplot as plt; 

#reading input and storing the dependent and independent variables
path_to_file = input("Enter the path to the dataset: ");
new_path = "";
for i in path_to_file:
    new_path += i;
    if (i=="\\"):
        new_path += "\\";
print(new_path);
dataset = pd.read_csv(new_path);
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

ans = input("Does the data have categorical values ? (yes/no): ");
if (ans.lower() != "no"):
    col = int(input("Enter the column which has categorical values(0-based indices, indices start from 0): "));
    x_encoder = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[col])],remainder="passthrough");
    x = np.array(x_encoder.fit_transform(x));
    #ignoring the first variable for avoiding the dummy variable trap
    x = x[:,1:];

#adding a column of 1s to include the constant term, as the ols model doesn't do that
#for us.
x = np.append(arr=x,values=np.ones((len(x),1)).astype(int),axis=1);
x = np.array(x, dtype="float");
y = np.array(y, dtype="float");

#setting random_state to a fixed number so that result can be verified manually
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0);

#lines below does the backward elimination
flag = 1;
significance_level = float(input("Enter the significance level : "));
res = "";
ans = input("Display indices deleted at each stage?(yes/no): ");
debug_entire_process = True;
if (ans.lower()=="no"):
    debug_entire_process = False;
while flag == 1:
  ols_model = sm.OLS(y_train, x_train).fit();
  ols_summary = ols_model.summary();
  res = ols_summary.as_text();
  vals = ols_model.pvalues;
  max_value = max(vals);
  if max_value > significance_level:    
    index = np.where(vals==max_value);
    if debug_entire_process:
        print(index);
    x_train = np.delete(x_train, index, 1);
    x_test = np.delete(x_test, index, 1);
  else:
    flag = 0;
    y_train_pred = ols_model.predict(x_train);
    y_test_pred = ols_model.predict(x_test);

if debug_entire_process:
    print(res);

x_range = [];
for i in range(0,len(y_train)):
  x_range.append(i);

plt.plot(x_range, y_train_pred, 'red', label = "Predicted");
plt.plot(x_range, y_train, 'blue', label = "Actual");
plt.title("Testing data");
plt.legend(loc='best');