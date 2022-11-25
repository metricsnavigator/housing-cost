'''
Author: Eric Reschke
Cite: https://metricsnavigator.org/housing_cost/
Last Reviewed: 2022-11-23
License: Open to all
'''

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

NaN = np.nan
wd = 'C:/Users/ericr/Desktop/MetricsNavigator/posts/wip/housing_cost/'

# iterating the columns
def colReveal(x):
    for col in x.columns:
        print(col)

# z-score function
def zScores(df,col):
    if (len(df)<=1):
        print('df must have more than one row')
    else:
        samplePop = len(df)-1
        avg = np.mean(df[col])
        df['varEach'] = (df[col]-avg)**2
        var = sum(df['varEach'])/(samplePop)
        sd = np.sqrt(var)
        df[col+'zScore'] = (df[col]-avg)/sd
        df.drop('varEach',axis=1,inplace=True)
        return(df)

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


# ------------------------------- #
# begin ML setup

# shave off features for ML setup
x_train = df_training.iloc[:,0:-1].values
y_train = df_training.iloc[:,-1].values

# neural network
randState=23
classifier = MLPClassifier(hidden_layer_sizes=(300,200,150,50),
                           max_iter=500,activation = 'relu',solver='adam',
                           random_state=randState)

classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_train)
y_export = pd.DataFrame(y_pred)
y_export.columns=['y_pred']

df_training['Predicted_cohort'] = y_export['y_pred']

# export the initial run to csv
df_training.to_csv('housing_price_predictions.csv',index=False)

# accuracy check on part2 run
df_training['Accuracy'] = np.where(df_training['SalePrice']==df_training['Predicted_cohort'],1,0)
mlp_training_accuracy = round(np.sum(df_training['Accuracy'])/len(df_training),4)
print('\nInitial Run Accuracy:',mlp_training_accuracy,'\n')

# confusion matrix
cTrainingMatrix = confusion_matrix(df_training['SalePrice'],df_training['Predicted_cohort'])
cTrainingMatrix = pd.DataFrame(cTrainingMatrix)
RunError = round(cTrainingMatrix[0][1]/cTrainingMatrix[1][1],2)
    
# training confusion matrix
print('--- Confusion Matrix ---','\n')
print('Left Side Actuals; Headers Predictions\n\n',cTrainingMatrix,'\n')

# counts of accurate and missed cohort predictions
missedPred = cTrainingMatrix[0][1]+cTrainingMatrix[0][2]+cTrainingMatrix[1][0]+cTrainingMatrix[1][2]+cTrainingMatrix[2][0]+cTrainingMatrix[2][1]
print('Count Accurate:',len(df_training)-missedPred,'\nCount Missed:',missedPred)


## end of script

