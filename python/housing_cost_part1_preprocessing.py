'''
Author: Eric Reschke
Cite: https://metricsnavigator.org/housing_cost/
Last Reviewed: 2022-11-23
License: Open to all
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

NaN = np.nan

housing_train_import = pd.read_csv(wd+'kaggle_train.csv')
housing_test_import = pd.read_csv(wd+'kaggle_test.csv')

df_training = housing_train_import.copy()
df_testing = housing_test_import.copy()

# Note: SalePrice is not in the testing dataset

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
      enCodeCol = (curCol+'_Encoding')
      for col in temp:
        tCol = (curCol+str(col))
        colNames.append(tCol)
      temp.columns = colNames
      #temp[enCodeCol] = labelencoder.fit_transform(df_training[curCol])
      df_training = df_training.join(temp)

# creating a Python list of columns to keep for the final setup
finalCols = []
for col in df_training:
    if df_training[col].dtypes!='object':
      finalCols.append(col)

df_training = df_training[finalCols]

# removing the ids from the dataframe
trainingIDs = df_training['Id']
df_training = df_training.drop('Id',axis=1)

# placing target feature at the end of the dataframe
salePrice = df_training['SalePrice']
df_training = df_training.drop('SalePrice',axis=1)
df_training['SalePrice'] = salePrice


## end of script

