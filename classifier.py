# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:45:13 2019

@author: WF465HJ
"""

import pandas as pd
import os, sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import RFECV
import gc 
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import resample
from lightgbm import LGBMClassifier
import warnings
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

os.chdir(r"C:\Users\WF465HJ\Documents\Raymond James")

## Do not show SettingWithCopyWarning warning
pd.options.mode.chained_assignment = None

## Do not show RuntimeWarning warning
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

## Do not show DeprecationWarning warning
warnings.simplefilter(action = "ignore", category = DeprecationWarning)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-fpr, index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

## For Sklearn categorical features!
def one_hot_encoding(train, test):
    for col in train.columns: 
        if train[col].dtype == 'object':
            print('encoding column - ', col)
            le = LabelEncoder()
            le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
            train[col] = le.transform(list(train[col].astype(str).values))
            test[col] = le.transform(list(test[col].astype(str).values))
    return train, test



## For tensorflow categorical features!
def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab))



def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn



def upsample(df, targetCol,frac):
    print('value count before upsampling - \n',df[targetCol].value_counts(normalize=True))
    zero = df[df[targetCol]==0]
    one = df[df[targetCol]==1]    
    # upsample minority
    min_upsampled = resample(one, replace=True, # sample with replacement
                             n_samples=int(frac*len(zero)), # match number in majority class
                             random_state=27) # reproducible results    
    # combine majority and upsampled minority
    df = pd.concat([zero, min_upsampled])
    
    # check new class counts
    print('value count after upsampling - \n', df[targetCol].value_counts(normalize=True))
    return df
    


def removeFeatures(df, targetCol):
    ## Features with only 1 unique value
    one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]    
    ## Features with more than 90% missing values
    ## isnull().sum() - counts number of null values in a particular column
    many_null_cols = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]        
    ## Features with the top value appears more than 90% of the time
    big_top_value_cols = [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    print('Features in dataset set with only 1 unique value -', one_value_cols)   
    print('Features in train set with more than 90% missing values -', many_null_cols)    
    print('Features in train set with the top value appears more than 90% of the time -', big_top_value_cols)
    cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))    
    cols_to_drop.remove(targetCol)
    print('features that will be dropped - ', cols_to_drop)   
    df = df.drop(cols_to_drop, axis=1)    
    return df



def trainModel(X_train, X_val, y_train, y_val, clf):
    print(clf)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)    
    print('Classification Report\n', classification_report(y_val, pred))
    print('ROC accuracy: {}'.format(roc_auc_score(y_val, pred)))    
    CM = confusion_matrix(y_val, pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print('Before calculating threshold')
    print('*'*30)
    print('True Positive - ',TP,'\nTrue Negative - ', TN, '\nFalse Positive - ', FP, '\nFalse Negative - ', FN )
    print('Error rate - ', ((FP+FN)/(TN+TP+FP+FP)*100),'%')
    print('False positive rate - ', ((FP)/(TN+FP)*100),'%')
    print('False negative rate - ', ((FN)/(TP+FN)*100),'%')    
    # Add prediction probability to dataframe
    X_train['pred_proba'] = clf.predict_proba(X_train)[:,1]
    X_val['pred_proba'] = clf.predict_proba(X_val)[:,1]    
    threshold = Find_Optimal_Cutoff(y_train, X_train['pred_proba'])
    print ('\nOptimal threshold value', threshold)    
    # # Find prediction to the dataframe applying threshold
    X_val['pred'] = X_val['pred_proba'].map(lambda x: 1 if x > threshold[0] else 0)
    CM = confusion_matrix(y_val, X_val['pred'])
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print('\nAfter calculating threshold')
    print('*'*30)
    print('True Positive - ',TP,'\nTrue Negative - ', TN, '\nFalse Positive - ', FP, '\nFalse Negative - ', FN )
    X_train.drop(['pred_proba'], axis=1, inplace=True)
    X_val.drop(['pred_proba', 'pred'], axis=1, inplace=True)    
    return clf


def norm(X_train, X_val):
    train_stats = X_train.describe()
    train_stats = train_stats.transpose()    
    return (X_train - train_stats['mean']) / train_stats['std'], (X_val - train_stats['mean']) / train_stats['std']



df = pd.read_csv(".\Data\logistic_output_train.csv", index_col=0)
df.drop(['LogisticScore', 'AdjustedScore'], axis=1, inplace=True)
print('Number of columns - ',df.shape[1],'\nNumber of rows - ',df.shape[0])

targetCol = 'IS_SAR'

print(df[targetCol].value_counts(normalize=True))

print('Columns with number of null values in them\n', df.isna().sum())

df = removeFeatures(df, targetCol)

## Reduce memory usage
df = reduce_mem_usage(df)

## Impute nulls    
df.fillna(-999, inplace=True)

## Split into train and validation set
test_size_ = 0.33
X_train, X_val, y_train, y_val = train_test_split(df, df[targetCol], test_size = test_size_, 
                                                  random_state=42, stratify=df[targetCol])

## Following lines only if upsampling is required - by duplicating positive values
frac = 0.6
X_train = upsample(X_train, targetCol, frac)
y_train = X_train[targetCol]


X_train = X_train.drop([targetCol],axis=1)
X_val = X_val.drop([targetCol],axis=1)


## Following lines only if upsampling is required - using SMOTE (didn't give good result)
sm = SMOTE(random_state=12, ratio = 0.3)
cols = X_train.columns
X_train, y_train = sm.fit_sample(X_train, y_train)
X_train = pd.DataFrame(X_train, columns = cols)
y_train = pd.Series(y_train)


#### One hot encoding for sklearn classifiers
X_train, X_val = one_hot_encoding(X_train, X_val)


#### For normalizing
X_train, X_val = norm(X_train, X_val)
####



clf = LogisticRegression(class_weight = 'balanced')

clf1 = trainModel(X_train, X_val, y_train, y_val, clf)


clf = LGBMClassifier(n_estimators=100, silent=False, random_state =94, max_depth=3,num_leaves=5,objective='binary',metrics ='auc')

clf2 = trainModel(X_train, X_val, y_train, y_val, clf)



NUM_EXAMPLES = len(y_train)
## Uncomment following code if there are categorical and numeric columns - make sure you have two lists 
## for CATEGORICAL_COLUMNS and NUMERIC_COLUMNS

# CATEGORICAL_COLUMNS = ['cat_col1','cat_col2']
# NUMERIC_COLUMNS = ['num_col1','num_col2']

# for feature_name in CATEGORICAL_COLUMNS:
#     # Need to one-hot encode categorical features
#     vocabulary = df[feature_name].unique()
#     feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

# for feature_name in NUMERIC_COLUMNS:
#     feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

feature_columns = []
for feature_name in list(X_train.columns):
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

train_input_fn = make_input_fn(X_train, y_train)
eval_input_fn = make_input_fn(X_val, y_val, shuffle=False, n_epochs=1)

############## Tensorflow linear classifier ##############
linear_est = tf.estimator.LinearClassifier(feature_columns)

# Train model.
linear_est.train(train_input_fn, max_steps=100)

# Evaluation.
result = linear_est.evaluate(eval_input_fn)

print(pd.Series(result))


############## Tensorflow boosted tree classifier ##############
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps=100)

# Eval
result = est.evaluate(eval_input_fn)
print(pd.Series(result))

pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])


#### Probability of '1' in histogram
probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()

#### ROC
fpr, tpr, _ = roc_curve(y_val, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()


############## Keras Basic Neural Network classifier ##############
input_dim_ = X_train.shape[1]
model = keras.Sequential([
    keras.layers.Dense(20,input_dim=input_dim_, activation='relu'),
    keras.layers.Dense(30,input_dim=input_dim_, activation='relu'),
    #keras.layers.Dense(20,input_dim=input_dim_, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])
    
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

_, accuracy = model.evaluate(X_val, y_val)
print('Accuracy: %.2f' % (accuracy*100))

pred = model.predict_classes(X_val).flatten()

CM = confusion_matrix(y_val, pred)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
print('True Positive - ',TP,'\nTrue Negative - ', TN, '\nFalse Positive - ', FP, '\nFalse Negative - ', FN )
print('False positive rate - ', ((FP)/(TN+FP)*100),'%')
print('False negative rate - ', ((FN)/(TP+FN)*100),'%')