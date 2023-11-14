'''
Author: Eric Reschke
Cite: https://www.metricsnavigator.com/housing-cost-prediction/
Last Reviewed: 2023-11-14
License: Open to all
'''

import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# shave off features for ML setup
x_test = df_testing.iloc[:,:].values
x_train = df_training.iloc[:,0:-1].values
y_train = df_training.iloc[:,-1].values

# regression model (done on training set)
lin = sklearn.linear_model.LinearRegression()
lin.fit(x_train,y_train)

# ------------------------------- #
# regression on testing data
testPrediction = lin.predict(x_test)
TestingPrediction = []
for i in testPrediction:
    x = round(i,0)
    TestingPrediction.append(x)

# add prediction and ids to testing dataset 
y_export = pd.DataFrame(TestingPrediction)
y_export.columns=['Prediction']
df_testing['Prediction'] = TestingPrediction
df_testing['Id'] = testingIDs

# ------------------------------- #
# regression on training data
trainPrediction = lin.predict(x_train)
TrainingPrediction = []
for i in trainPrediction:
    x = round(i,0)
    TrainingPrediction.append(x)

y_train_export = pd.DataFrame(TrainingPrediction)
y_train_export.columns=['Prediction']
df_training['Prediction'] = y_train_export['Prediction']
df_training['Id'] = trainingIDs

# ------------------------------- #

## regression graph
df_training = df_training.sort_values('SalePrice')
x=0
SortIndex = []
for i in range(len(df_training)):
    x+=1
    SortIndex.append(x)
df_training['SortedValueIndex'] = SortIndex
plt.plot(df_training['SalePrice'],df_training['SortedValueIndex'],label='SalePrice',linestyle='-.',color='blue')
plt.plot(df_training['Prediction'],df_training['SortedValueIndex'],label='Prediction',linestyle='-.',color='green',alpha=0.3)
plt.legend()
plt.show()

# r-squared value
r2 = round(lin.score(x_train,y_train),4)
print('\nR2 Value:',r2)

# create dataframe for coefficients and intercept
coeff = []
for i in lin.coef_:
    x = round(i,4)
    coeff.append(x)

# list out the coefficients of the model
cNames = finalCols.copy()
cNames.remove('Id')
cNames.remove('index')
coeff = pd.DataFrame(coeff,index=cNames)
coeff.columns=['Results']
coeff.loc['Intercept'] = round(lin.intercept_,4)
print(coeff,'\n')

# calculate the RMSE between sales-actual and predictions
df_rmse = df_training.copy()
df_rmse = df_rmse[['SalePrice','Prediction']]
df_rmse['SaleLog'] = np.log(df_rmse['SalePrice'])
df_rmse['PredictionLog'] = np.log(df_rmse['Prediction'])

# mean-squared error
mse = round(mean_squared_error(y_true=df_rmse['SaleLog'],y_pred=df_rmse['PredictionLog']),4)
print('\n')
print('MSE:',mse)

# root mean-squared error
rmse = round(np.sqrt(mse),4)
print('RMSE:',rmse)


## end of script

