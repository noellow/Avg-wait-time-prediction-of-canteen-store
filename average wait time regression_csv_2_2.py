# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:20:30 2021

@author: Noel Low, Woo Jia Jun
"""

from matplotlib.pyplot import subplots, show
from matplotlib.dates import DateFormatter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import datetime as dt
import sys


###Do a function that converts 24h time into seconds###
def convert(time):
    time_in_string = str(time)
    'print(len(time_in_string))'

    if len(time_in_string) == 4 and time_in_string.isnumeric() is True:
         hours = int(time[0]+time[1]) #convert hours into integer
         minutes = int(time[2]+time[3]) #convert minutes into integer
         new_time_in_seconds = hours*3600+minutes*60
         #check if time is valid, proceed with conversion
         if 10<=hours<19 and 0<=minutes<60:

        ##time started at 1020hrs

             time_elapsed = new_time_in_seconds-(10*3600 + 20*60)
             if time_elapsed <=28200:
                return time_elapsed
             else:
                 print("Time is out of range")
                 sys.exit()
         else:
             print("Time provided is out of range, or invalid")
             sys.exit()
    else:
        print("Yo dude, enter time in (hhmm)")
        sys.exit()



#import csv file

dataset = pd.read_csv(r'.\Entrance and Exit 010321_CLEANED_2.csv',sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python',)


#check to see if dataset is gud after converting the entrance timestamp to a datetime format
dataset["Time Stamp (Conversion)"] = dataset["ENTRANCE Time"].apply(lambda x: dt.datetime(2020,3,1,10,20,0)+dt.timedelta(seconds=x))
#print(dataset, dataset.dtypes)

#Let X be the entrance time stamp and Y be the wait time in seconds

#split into training and testing set
X_train, X_test, Y_train, Y_test= train_test_split(dataset["ENTRANCE Time"].values.reshape(-1,1), dataset["Wait Time"].values.reshape(-1,1), test_size=0.1, random_state=0)


#generate polynomial regression model using scikit-learn pipeline
#steps taken: robust feature scaling (less effect from outlier data), polynomial feature generation, linear regression

poly_reg_model = Pipeline([("rob_scaler", RobustScaler()),("poly", PolynomialFeatures(degree=5)),("regression", LinearRegression())])
poly_reg_model.fit(X_train, Y_train)
#print(mymodel)

#generate graph space
fig, ax = subplots(1)
#give the graph a title
fig.suptitle("Observed Data With Regression Line")
#format the x-axis labels to fit the date/time
fig.autofmt_xdate()
#plot the data points and regression line
ax.scatter(dataset["Time Stamp (Conversion)"], dataset["Wait Time"], label="Data")
ax.plot(dataset["Time Stamp (Conversion)"], poly_reg_model.predict(dataset["ENTRANCE Time"].values.reshape(-1,1)), "r", label="Regression")
#set axis labels
ax.set_xlabel("Entrance Time")
ax.set_ylabel("Wait Time (s)")
#format the time labels used in the x-axis and the graph proper
custom_time_format = DateFormatter("%I:%M %p")
#extend x-axis range by rounding to nearest hours on either end of the range
ax.set_xlim([dataset.loc[dataset.index[0],"Time Stamp (Conversion)"].replace(minute=0, second=0, microsecond=0),dataset.loc[dataset.index[-1],"Time Stamp (Conversion)"].replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)])
ax.xaxis.set_major_formatter(custom_time_format)
ax.fmt_xdata = custom_time_format
#generate graph legend and display graph
ax.legend()
show()

"""
###PRINT R2 VALUE###
from sklearn.metrics import r2_score
r2_value = (r2_score(Y,mymodel(X)))
print("R2 value is: ", r2_value)
"""
##Close the graph to proceed to prediction section below
##Here we ask user to enter a time###
time_input = input("Enter the time in hhmm\n")
X_pred = convert(time_input)
av_wait_time = int(poly_reg_model.predict(np.array(X_pred).reshape(-1,1)))
print('Average wait time is: %i seconds' % av_wait_time)











