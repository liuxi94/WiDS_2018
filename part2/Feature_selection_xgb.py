# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:26:21 2018

@author: Snorlax & Xi
"""

import xgboost as xgb
import xgbfir
import pandas as pd
import numpy as np
from itertools import product

def null_count(df):
    df['sum_nulls'] = df.isnull().sum(axis=1)
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
    df['Age'] = 2016 - df['DG1']
    del df['DG1']
    #positive
    #fraction
    GN1 = np.where(df['GN1'] == 2, 1, 0)
    GN2 = np.where(df['GN2'] == 2, 1, 0)
    GN3 = np.where(df['GN3'] == 2, 1, 0)
    GN4 = np.where(df['GN4'] == 2, 1, 0)
    GN5 = np.where(df['GN5'] == 2, 1, 0)
    df['if_GN_by_spouse_score'] = GN1 + GN2 + GN3 + GN4 + GN5
    #medium
    df['if_illiterate'] = np.where(df['DG4'] ==1, 1, 0)
    df['if_supported'] = np.where(df['DL5'] == 5, 1, 0)
    
    #high
    df['if_widow'] = np.where(df['DG3'] == 6, 1, 0)
    df['if_borrow_phone_from_spouse'] = np.where(df['MT7A'] == 1, 1, 0)
    #super
    df['if_spouse_househead'] = np.where(df['DG6'] == 2, 1, 0)
    df['if_housewife'] = np.where(df['DL1'] == 7, 1, 0)
    df['if_spouse_decide_phone'] = np.where(df['MT1A'] == 2, 1, 0)
    df['if_phone_bought_by_spouse'] = np.where(df['MT6'] == 2, 1, 0)
    
    #low
    df['if_high_education'] = np.where(df['DG4'] >= 9, 1, 0)
    df['if_self_business'] = np.where((df['DL5'] == 6) | (df['DL5'] == 19) | (df['DL5'] == 20), 1, 0)
    #df['if_have_phone'] = np.where(df['MT2'] == 1, 1, 0)
    #medium
    df['if_full_time'] = np.where((df['DL1'] == 1) | (df['DL1'] == 5) , 1, 0)
    df['if_myself_decide_phone'] = np.where(df['MT1A'] == 1, 1, 0)
    #high
    #df['if_myself_main_income'] = np.where(df['DL0'] == 1, 1, 0)
    df['if_phone_bought_by_myself'] = np.where(df['MT6'] == 1, 1, 0)
    df['if_myself_househead'] = np.where(df['DG6'] == 1, 1, 0)
    #df['if_have_drive_license'] = np.where(df['DG5_4'] == 1, 1, 0)
    #not decided and new
    df['if_not_working'] = np.where(df.DL2.isnull(), 1, 0)
    #df['if_secondary_job'] = np.where(df['DL3'] == 1, 1, 0)
    df['if_phone_not_allowed_by_others'] = np.where(df['MT9'] == 3, 1, 0)
    #df['if_know_mobile_money'] = np.where(df['MM1'] == 1, 1, 0)
    df['new_AA6'] = 10 * df['AA4'] + df['AA5']
    df['if_AA14_99999'] = np.where(df['AA14'] == 99999, 1, 0)
    df['DL7'] = df['DL7'].replace(to_replace=2, value=0)
    df['how_much_cultivated_by_yourself'] = df['DL7'] * df['DL8']
    #language
    df['if_other_language_null'] = np.where(df.LN2_RIndLngBEOth.isnull(), 1, 0)
    df['if_bto_language'] = np.where((df['LN2_RIndLngBEOth'] == 'Bengali') | (df['LN2_RIndLngBEOth'] == 'Tamil') | (df['LN2_RIndLngBEOth'] == 'Oriya'), 1, 0)
    

def prepare_train(train):
    col_names = list(train)
    for col in col_names:
        if (col.endswith('OTHERS')) | (col.endswith('REC')) | (col.endswith('BEOth')):
            del train[col]
            continue
        train[col] = train[col].astype('float32',copy=False)
    train['is_female'] = train['is_female'].astype('int64')


def pp_and_fe(df):
    null_count(df)
    combine_and_delete(df)
    test_feature_generation(df)
    
def get_feature(file):
    f_set = set()
    data = pd.read_csv(file)
    for col in list(data):
        features = np.array(data[[col]]).tolist()
        for row_fe in features:
            singles = row_fe[0].split('|')
            f_set.update(singles)
    return list(f_set)
        
def get_stats(train, test, target_column, group_column):
    '''
    target_column: numeric columns to group with 
    group_column: categorical columns to group on 
    '''
    train_df = train.copy()
    test_df = test.copy()
    
    train_df['row_id'] = train_df['train_id']
    test_df['row_id'] = test_df['test_id']
    train_df['train'] = 1
    test_df['train'] = 0
    all_df = train_df[['row_id', 'train',  target_column, group_column]].append(test_df[['row_id','train', 
                                                                                        target_column, group_column]])
    grouped = all_df[[target_column, group_column]].groupby(group_column)
    the_size = pd.DataFrame(grouped.size()).reset_index()
    the_size.columns = [group_column, '%s_size_on_%s' % (target_column, group_column)]
    the_mean = pd.DataFrame(grouped.mean()).reset_index()
    the_mean.columns = [group_column, '%s_mean_on_%s' % (target_column, group_column)]
    the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
    the_std.columns = [group_column, '%s_std_on_%s' % (target_column, group_column)]
    the_median = pd.DataFrame(grouped.median()).reset_index()
    the_median.columns = [group_column, '%s_median_on_%s' % (target_column, group_column)]
    the_stats = pd.merge(the_size, the_mean)
    the_stats = pd.merge(the_stats, the_std)
    the_stats = pd.merge(the_stats, the_median)
    
    the_max = pd.DataFrame(grouped.max()).reset_index()
    the_max.columns = [group_column, '%s_max_on_%s' % (target_column, group_column)]
    the_min = pd.DataFrame(grouped.min()).reset_index()
    the_min.columns = [group_column, '%s_min_on_%s' % (target_column, group_column)]
    
    the_stats = pd.merge(the_stats, the_max)
    the_stats = pd.merge(the_stats, the_min)
    
    all_df = pd.merge(all_df, the_stats)
    
    
    selected_train = all_df[all_df['train'] == 1]
    selected_test = all_df[all_df['train'] == 0]
    selected_train.sort_values('row_id', inplace=True)
    selected_test.sort_values('row_id', inplace=True)
    selected_train.drop([target_column, group_column, 'train'], axis=1, inplace=True)
    selected_test.drop([target_column, group_column, 'train'], axis=1, inplace=True)
    selected_train.rename(columns={'row_id': 'train_id'}, inplace=True)
    selected_test.rename(columns={'row_id': 'test_id'}, inplace=True)
    
    
    tmp_train = pd.merge(train, selected_train)
    tmp_test = pd.merge(test, selected_test)
    return tmp_train, tmp_test 

   
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
s1 = test['test_id']

pp_and_fe(train)
pp_and_fe(test)

prepare_train(train)

X = train
Y = train['is_female']
    
tempX = X.copy()
del tempX['is_female']
del tempX['train_id']
Y_train = pd.DataFrame.as_matrix(Y)

xgdmat = xgb.DMatrix(tempX, Y_train)

our_params = {'eta': 0.01, 'seed':27, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 5, 'eval_metric':'auc', 
             'objective': 'binary:logistic', 'max_depth':7, 'min_child_weight':1, 'lambda': 0.1, 'scale_pos_weight':0.862} 
booster = xgb.train(our_params, xgdmat, num_boost_round = 2700)
######################################################################################
######################################################################################
#topK = 150
xgbfir.saveXgbFI(booster, TopK = 150, MaxTrees = 1000, MaxInteractionDepth = 5, OutputXlsxFile='xgb.xlsx')
#########################################################################################
#manually copy and paste features from xgb.xlsx and then read it 
feature_list = get_feature('features.csv')
feature_list.sort()
final_list = ['is_female', 'train_id']
final_list.extend(feature_list)

train_with_selected_features = X[final_list].copy()
train_with_selected_features.to_csv('new_train_150.csv',index=False)
final_test_list = ['test_id']
final_test_list.extend(feature_list)
test[final_test_list].to_csv('new_test_150.csv',index=False)
###########################################################################################
