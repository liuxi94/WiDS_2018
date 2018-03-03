# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:55:30 2018

@author: Snorlax & Xi
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

def combine_and_delete(df):
    df['if_from_town'] = np.where(df.AA5.isnull(), 0, 1)
    del df['DG12B_2']
    del df['DG12C_2']
    del df['DG13_7']
    del df['DG13_OTHERS']
    cols_to_combine = ['DG13', 'DL25', 'DL26', 'MT4']
    #xxxx = ['DG13']
    for col_to_combine in cols_to_combine:
        group_cols = [col for col in df if col.startswith(col_to_combine)]
        group_col_1 = group_cols[0]
        df[col_to_combine] = 0
        for group_col in group_cols:
            df[group_col] = df[group_col].replace(to_replace=2, value=0)
            df[col_to_combine] = df[col_to_combine] + df[group_col]
        if df[group_col_1].isnull().sum() > 0:
            new_col_name = 'if_' + col_to_combine + '_is_null'
            df[new_col_name] = np.where(df[col_to_combine].isnull(), 1, 0)
            
def test_feature_generation(df):
    df['Age'] = 2018 - df['DG1']
    del df['DG1']
    #positive
    #fraction
    df['if_GN1_by_spouse'] = np.where(df['GN1'] == 2, 1, 0)
    df['if_GN2_by_spouse'] = np.where(df['GN2'] == 2, 1, 0)
    df['if_GN3_by_spouse'] = np.where(df['GN3'] == 2, 1, 0)
    df['if_GN4_by_spouse'] = np.where(df['GN4'] == 2, 1, 0)
    df['if_GN5_by_spouse'] = np.where(df['GN5'] == 2, 1, 0)
    
    #medium
    df['if_low_education'] = np.where(df['DG4'] <= 2, 1, 0)
    df['if_other_main_income'] = np.where(df['DL0'] == 2, 1, 0)
    df['if_supported'] = np.where(df['DL5'] == 5, 1, 0)
    df['if_no_phone'] = np.where(df['MT2'] == 2, 1, 0)
    df['if_no_sim'] = np.where(df.MT15.isnull(), 0, 1)
    
    #high
    df['if_widow'] = np.where(df['DG3'] == 6, 1, 0)
    df['if_borrow_phone_from_spouse'] = np.where(df['MT7A'] == 1, 1, 0)
    #super
    df['if_spouse_househead'] = np.where(df['DG6'] == 2, 1, 0)
    df['if_housewife'] = np.where(df['DL1'] == 7, 1, 0)
    df['if_spouse_decide_phone'] = np.where(df['MT1A'] == 2, 1, 0)
    df['if_phone_bought_by_spouse'] = np.where(df['MT6'] == 2, 1, 0)
    
    #negative
    #fraction
    df['if_GN1_by_myself'] = np.where(df['GN1'] == 1, 1, 0)
    df['if_GN2_by_myself'] = np.where(df['GN2'] == 1, 1, 0)
    df['if_GN3_by_myself'] = np.where(df['GN3'] == 1, 1, 0)
    df['if_GN4_by_myself'] = np.where(df['GN4'] == 1, 1, 0)
    df['if_GN5_by_myself'] = np.where(df['GN5'] == 1, 1, 0)
    #low
    df['if_high_education'] = np.where(df['DG4'] >= 9, 1, 0)
    df['if_self_business'] = np.where((df['DL5'] == 6) | (df['DL5'] == 19) | (df['DL5'] == 20), 1, 0)
    df['if_have_phone'] = np.where(df['MT2'] == 1, 1, 0)
    #medium
    df['if_full_time'] = np.where((df['DL1'] == 1) | (df['DL1'] == 5) , 1, 0)
    df['if_myself_decide_phone'] = np.where(df['MT1A'] == 1, 1, 0)
    #high
    df['if_myself_main_income'] = np.where(df['DL0'] == 1, 1, 0)
    df['if_phone_bought_by_myself'] = np.where(df['MT6'] == 1, 1, 0)
    df['if_myself_househead'] = np.where(df['DG6'] == 1, 1, 0)
    df['if_have_drive_license'] = np.where(df['DG5_4'] == 1, 1, 0)
    
#read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
s1 = test['test_id']

#basic preprocessing
train["row_nulls"] = train.isnull().sum(axis=1)
test["row_nulls"] = test.isnull().sum(axis=1)
combine_and_delete(train)
combine_and_delete(test)
test_feature_generation(train)
test_feature_generation(test)
train = train[train.columns[train.isnull().mean() < 0.7]]

#prepare for train and test
col_names = list(train)
for col in col_names:
    if train[col].dtypes == 'object':
        del train[col]
        continue
    train[col] = train[col].astype('float32',copy=False)
train['is_female'] = train['is_female'].astype('int64')
tempX = train
Y = train['is_female']
del tempX['train_id']
Y_train = pd.DataFrame.as_matrix(Y)
target = 'is_female'
predictors = [x for x in tempX.columns if x not in [target]]

col_names_train = list(tempX)
col_names_test = list(test)
for col in col_names_test:
    if col not in col_names_train or test[col].dtypes == 'object':
        del test[col]
for col in list(test):
    test[col] = test[col].astype('float32',copy=False)

#####################################################################################

#########################################################################################
#train and predict with cv-auc = 0.9730*
final_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'seed':27,
    'num_leaves': 32, 'learning_rate': 0.01, 'max_depth': -1,  'metric':'auc',
    'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.8, 'bagging_freq':1, 'bagging_seed':72,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'min_split_gain': 0.05, 'min_child_weight': 1, 'min_child_samples': 20, 'scale_pos_weight': 0.862}
num_round =2775
train_data=lgb.Dataset(tempX[predictors], Y)
lgb_model = lgb.train(final_params,train_data,num_round)
ypred2=lgb_model.predict(test)
s2 = pd.Series(ypred2, name='is_female')
out_df = pd.concat([s1, s2], axis=1).reset_index()
del out_df['index']
out_df.to_csv('result\lgb_seed27_simplified_fe.csv', index=False)


#########################################################################################
#train and predict with cv-auc = 0.972988 
final_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'seed':12,
    'num_leaves': 49, 'learning_rate': 0.005, 'max_depth': -1,  'metric':'auc', 'gamma': 8.3548,
    'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.8, 'bagging_freq':1, 'bagging_seed':22,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.2970, 'reg_lambda': 0.0614,
    'min_split_gain': 0.1336, 'min_child_weight': 1, 'min_child_samples': 29, 'scale_pos_weight': 0.862}
num_round =3825
train_data=lgb.Dataset(tempX[predictors], Y)
lgb_model = lgb.train(final_params,train_data,num_round)
ypred2=lgb_model.predict(test)
s2 = pd.Series(ypred2, name='is_female')
out_df = pd.concat([s1, s2], axis=1).reset_index()
del out_df['index']
out_df.to_csv('result\lgb_seed12_simplified_fe.csv', index=False)

