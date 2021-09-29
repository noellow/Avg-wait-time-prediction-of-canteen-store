# SUTD Canteen crowd prediction

The objective was to use linear regression to predict crowd levels in any store of our choice in the SUTD canteen during the store's operation. The code takes in user input and returns an approximate value for the crowd level at a user-specified time.

PROCESS:
1) Data was collected by positioning 2 PIR sensors at the entrance and exit of each store. The time taken for each person is determined by taking the difference in timing that the entrance and exit sensors were being activated. The time and count from each sensor was stored in a text file that was then later imported into a CSV file.

2) Data cleaning was done manually, as we received erronous data which did not make sense. The final CSV file has been uploaded in the repository.

3) The code was run in Python and the Polynomial Regression model had its terms tweaked iteratively to find the model which had the highest R-squared value, indicating that the model best fit our data points. 
