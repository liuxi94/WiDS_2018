# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 22:34:17 2018

@author: Snorlax & Xi
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import MeanEncoder

def null_count(df):
    df["sum_nulls"] = df.isnull().sum(axis=1)
    df['sum_99'] = (df == 99).sum(axis=1)
    df = df.replace(to_replace=99, value=-1)
    
def combine_and_delete(df):
    df['if_from_town'] = np.where(df.AA5.isnull(), 0, 1)
    df.loc[df['AA5'].isnull(),'AA5'] = df['AA6']
    del df['AA6']
    del df['DG12B_2']
    del df['DG12C_2']
    del df['DG13_7']
    del df['DG13_OTHERS']   
    del df['DL4_99']
    del df['DL4_OTHERS'] 
    del df['FF14_OTHERS'] 
    del df['MM2_OTHERS'] 
    del df['MMP1_OTHERS'] 
    cols_to_combine = ['DG5', 'DG13', 'DL4', 'DL25', 'DL26', 'MT4', 'MT17', 'FF14', 'MM2', 'MM3', 'MM4', 'MMP1']
    for col_to_combine in cols_to_combine:
        group_cols = [col for col in df if col.startswith(col_to_combine)]
        group_col_1 = group_cols[0]
        df[col_to_combine] = 0
        for group_col in group_cols:
            if col_to_combine == 'MT17':
                df[group_col] = abs(df[group_col] - 6)
            else:
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
    df['if_illiterate'] = np.where(df['DG4'] ==1, 1, 0)
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
    #not decided and new
    df['if_not_working'] = np.where(df.DL2.isnull(), 1, 0)
    df['if_secondary_job'] = np.where(df['DL3'] == 1, 1, 0)
    df['if_phone_not_allowed_by_others'] = np.where(df['MT9'] == 3, 1, 0)
    df['if_know_mobile_money'] = np.where(df['MM1'] == 1, 1, 0)
    df['new_AA6'] = 10 * df['AA4'] + df['AA5']
    df['if_AA14_99999'] = np.where(df['AA14'] == 99999, 1, 0)
    df['DL7'] = df['DL7'].replace(to_replace=2, value=0)
    df['how_much_cultivated_by_yourself'] = df['DL7'] * df['DL8']

def prepare_train(train):
    col_names = list(train)
    for col in col_names:
        if train[col].dtypes == 'object':
            del train[col]
            continue
        train[col] = train[col].astype('float32',copy=False)
    train['is_female'] = train['is_female'].astype('int64')

def pp_and_fe(df):
    combine_and_delete(df)
    test_feature_generation(df)

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))
############################################################################################## 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
s1 = test['test_id']

pp_and_fe(train)
pp_and_fe(test)

train = train[train.columns[train.isnull().mean() < 0.7]]
prepare_train(train)

X = train
Y = train['is_female']
del X['train_id']
del X['is_female']
########################################################################################
single_cat = ['AA4']
cat_features = ['AA7', 'new_AA6']

temp_m1 = MeanEncoder.MeanEncoder(categorical_features = single_cat, n_splits=10, prior_weight_func = lambda x: 1 / (1 + np.exp((x - 100) / 0.2)))
tempX = temp_m1.fit_transform(X,Y)
del tempX['AA4_pred_0']
temp_m2 = MeanEncoder.MeanEncoder(categorical_features = cat_features, n_splits=5, prior_weight_func = lambda x: 1 / (1 + np.exp((x - 5) / 1)))
tempX = temp_m2.fit_transform(tempX,Y)
del tempX['AA7_pred_0']
del tempX['new_AA6_pred_0']

tempX['AA4_pred_1'] = add_noise(tempX['AA4_pred_1'], 0.01)
tempX['AA7_pred_1'] = add_noise(tempX['AA7_pred_1'], 0.01)
tempX['new_AA6_pred_1'] = add_noise(tempX['new_AA6_pred_1'], 0.01)

cat_col_index = []
cat_col = ['AA3', 'AA5', 'DG3', 'DG3A', 'DG4', 'DG6', 'DG14', 'DL1', 'DL2', 'DL5']
for i, col in enumerate(tempX.columns):
    if col in cat_col:
        cat_col_index.append(i)
        
new_test = temp_m1.transform(test)
new_test = temp_m2.transform(new_test)

new_test['AA4_pred_1'] = add_noise(new_test['AA4_pred_1'], 0.01)
new_test['AA7_pred_1'] = add_noise(new_test['AA7_pred_1'], 0.01)
new_test['new_AA6_pred_1'] = add_noise(new_test['new_AA6_pred_1'], 0.01)

col_names_train = list(tempX)
col_names_test = list(new_test)
for col in col_names_test:
    if col not in col_names_train:
        del new_test[col]
for col in list(new_test):
    new_test[col] = new_test[col].astype('float32',copy=False)
final_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'seed':66,
    'num_leaves': 40, 'learning_rate': 0.01, 'max_depth': -1, 'categorical_feature': cat_col_index, 'gamma':0.5,
    'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.8, 'bagging_freq':1, 'bagging_seed':72,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'min_split_gain': 0.05, 'min_child_weight': 1, 'min_child_samples': 20, 'scale_pos_weight': 0.862}
num_round =2150
train_data=lgb.Dataset(tempX, Y)
lgb_model = lgb.train(final_params,train_data,num_round)
ypred2=lgb_model.predict(new_test)
s2 = pd.Series(ypred2, name='is_female')
out_df = pd.concat([s1, s2], axis=1).reset_index()
del out_df['index']
out_df.to_csv('result\lgb_seed66_address_meanencoder1.csv',index=False)

########################################################################################
single_cat = ['AA4']
cat_features = ['AA7', 'new_AA6']

temp_m1 = MeanEncoder.MeanEncoder(categorical_features = single_cat, n_splits=10, prior_weight_func = lambda x: 1 / (1 + np.exp((x - 100) / 0.2)))
tempX = temp_m1.fit_transform(X,Y)
del tempX['AA4_pred_0']
temp_m2 = MeanEncoder.MeanEncoder(categorical_features = cat_features, n_splits=10, prior_weight_func = lambda x: 1 / (1 + np.exp((x - 10) / 0.5)))
tempX = temp_m2.fit_transform(tempX,Y)
del tempX['AA7_pred_0']
del tempX['new_AA6_pred_0']

tempX['AA4_pred_1'] = add_noise(tempX['AA4_pred_1'], 0.01)
tempX['AA7_pred_1'] = add_noise(tempX['AA7_pred_1'], 0.01)
tempX['new_AA6_pred_1'] = add_noise(tempX['new_AA6_pred_1'], 0.01)

cat_col_index = []
cat_col = ['AA3', 'AA5', 'DG3', 'DG3A', 'DG4', 'DG6', 'DG14', 'DL1', 'DL2', 'DL5']
for i, col in enumerate(tempX.columns):
    if col in cat_col:
        cat_col_index.append(i)
        
new_test = temp_m1.transform(test)
new_test = temp_m2.transform(new_test)

new_test['AA4_pred_1'] = add_noise(new_test['AA4_pred_1'], 0.01)
new_test['AA7_pred_1'] = add_noise(new_test['AA7_pred_1'], 0.01)
new_test['new_AA6_pred_1'] = add_noise(new_test['new_AA6_pred_1'], 0.01)

col_names_train = list(tempX)
col_names_test = list(new_test)
for col in col_names_test:
    if col not in col_names_train:
        del new_test[col]
for col in list(new_test):
    new_test[col] = new_test[col].astype('float32',copy=False)
final_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'seed':55,
    'num_leaves': 40, 'learning_rate': 0.01, 'max_depth': -1, 'categorical_feature': cat_col_index, 'gamma':0.5,
    'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.8, 'bagging_freq':1, 'bagging_seed':72,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'min_split_gain': 0.05, 'min_child_weight': 1, 'min_child_samples': 20, 'scale_pos_weight': 0.862}
num_round =2150
train_data=lgb.Dataset(tempX, Y)
lgb_model = lgb.train(final_params,train_data,num_round)
ypred2=lgb_model.predict(new_test)
s2 = pd.Series(ypred2, name='is_female')
out_df = pd.concat([s1, s2], axis=1).reset_index()
del out_df['index']
out_df.to_csv('result\lgb_seed55_address_meanencoder2.csv',index=False)