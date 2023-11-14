'''
Author: Eric Reschke
Cite: https://www.metricsnavigator.com/housing-cost-data-model/
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

housing_train_import = pd.read_csv(wd+'kaggle_train.csv')
housing_test_import = pd.read_csv(wd+'kaggle_test.csv')

df_training = housing_train_import.copy()
df_testing = housing_test_import.copy()

salePrice = df_training['SalePrice']
df_training = df_training.drop('SalePrice',axis=1)
df_training = pd.concat([df_training,df_testing],sort=False).reset_index()

'''
# confirming null values across columns
for col in df_training:
    # this reveals null values; I cut out anything missing over 50% of useable data next
    print(col,': Missing Percent =',round(df_training[col].isna().sum()/len(df_training),2))
'''

# remove features that have too many missing elements
droppedCols = ['Alley','PoolQC','Fence','MiscFeature']
df_training = df_training.drop(droppedCols,axis=1)

# impute (median) missing numerical values
# Note: I let GarageYrBlt be included in this impute-exercise for now
for col in df_training:
    if df_training[col].dtypes!='object':
        median = df_training[col].median()
        df_training[col].fillna(median,inplace=True)

# convert all float data to integers
for col in df_training:
    if df_training[col].dtypes!='object':
        df_training[col] = df_training[col].apply(np.int64)

'''
# code to confirm conversion went as expected
print(df_training.dtypes)
'''

# replace null values for categorical features with 'unknown'
df_training.fillna('unknown',inplace=True)

# add labels for categorical data tracking
labelencoder = LabelEncoder()
enc = OneHotEncoder()

# loop that automatically transforms data to binary and creates unique column names
for col in df_training:
    colNames = []
    if df_training[col].dtypes=='object':
      temp = pd.DataFrame(enc.fit_transform(df_training[[col]]).toarray())
      curCol = col
      for col in temp:
        tCol = (curCol+str(col))
        colNames.append(tCol)
      temp.columns = colNames
      df_training = df_training.join(temp)

# creating a Python list of columns to keep for the final setup
finalCols = []
for col in df_training:
    if df_training[col].dtypes!='object':
      finalCols.append(col)

df_training = df_training[finalCols]

# split the testing from the training dataset
df_testing = df_training.iloc[1460:,:].drop('index',axis=1)
df_training = df_training.iloc[0:1460,:].drop('index',axis=1)

# removing the ids from the datasets
trainingIDs = df_training['Id']
df_training = df_training.drop('Id',axis=1)

testingIDs = df_testing['Id']
df_testing = df_testing.drop('Id',axis=1)

# add back saleprice for the regression model
df_training['SalePrice'] = salePrice


## end of script

