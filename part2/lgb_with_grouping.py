# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:33:06 2018

@author: Snorlax & Xi
get_stat function are from kaggler https://www.kaggle.com/xiaozhouwang on page:https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32123
"""
import pandas as pd
import lightgbm as lgb
from itertools import product

def get_stats(train, test, target_column, group_column):
    '''
    target_column: numeric columns to group with 
    group_column: categorical columns to group on 
    '''
    train_df = train.copy()
    test_df = test.copy()
    #train_df['row_id'] = range(train_df.shape[0])
    #test_df['row_id'] = range(test_df.shape[0])
    
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

def chooser(tempX, Y):    
    test_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 
    'num_leaves': 42, 'learning_rate': 0.05, 'max_depth': -1, 'gamma':5,
    'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.8, 'bagging_freq':1, 'bagging_seed':72,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'min_split_gain': 0.2, 'min_child_weight': 1, 'min_child_samples': 6, 'scale_pos_weight': 0.862}
    
    lgb_params = test_params.copy()
    dset = lgb.Dataset(tempX, Y, silent=True)
    cv_results = lgb.cv(
        lgb_params, dset, num_boost_round=5000, nfold=10, stratified=True, shuffle=True, metrics=['auc'],
        early_stopping_rounds=100, verbose_eval=1000, show_stdv=True, seed=27)
    return int(len(cv_results['auc-mean']) * 10 / 9), cv_results['auc-mean'][-1]

#########################################################################################
train = pd.read_csv('new_train_150.csv')
test = pd.read_csv('new_test_150.csv')
s1 = test['test_id']
test['test_id'] = test['test_id'].astype('int64',copy=False)

X = train.copy()
Y = train['is_female']

######################################################################################
######################################################################################
#get_stat preprocessing
tempX = X.copy()
baseX = X.copy()
num_cols = ['Age', 'sum_nulls', 'sum_99']
cat_cols = ['AA14', 'DG3', 'DG4', 'DG6', 'DL1', 'DL15']
for target_column, group_column in product(num_cols, cat_cols):
    tempX, test = get_stats(tempX, test, target_column=target_column, group_column = group_column)
    new_name1 = target_column + '-' + target_column + '_mean_on_' + group_column
    tempX[new_name1] = tempX[target_column] - tempX['%s_mean_on_%s' % (target_column, group_column)]
    test[new_name1] = test[target_column] - test['%s_mean_on_%s' % (target_column, group_column)]
del tempX['train_id']
del tempX['is_female']
del baseX['train_id']
del baseX['is_female']
'''
#greedy grouping stat selection, takes a long time to run, but only several new columns are
#selected
new_cols = list(set(tempX) - set(baseX))
base_cols = list(baseX)
num_rounds, auc = chooser(baseX, Y)
for new_col in new_cols:
    new_features = base_cols.copy()
    new_features.append(new_col)
    tmp_df = tempX[new_features].copy()
    new_rounds, new_auc = chooser(tmp_df, Y)
    if new_auc > auc + 0.00001:
        auc = new_auc
        base_cols = new_features.copy()
        print('%s is selected and new auc is %f' % (new_col, auc))
    else:
        print('%s is removed' % (new_col))

print(auc)
print(list(set(base_cols) - set(baseX)))
#get final list and selected columns generated by grouping are ['sum_99-sum_99_mean_on_DL15',  'sum_99_max_on_DG3',  'Age_std_on_AA14',  'sum_nulls_std_on_DG6']
final_list = list(base_cols)
'''
final_list = list(baseX)
final_list.extend(['sum_99-sum_99_mean_on_DL15',  'sum_99_max_on_DG3',  'Age_std_on_AA14',  'sum_nulls_std_on_DG6'])
no_elimination_list = list(tempX)

#######################################################################################################
tempX = tempX[no_elimination_list]
test = test[no_elimination_list]

final_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'metric':'auc',
    'num_leaves': 44, 'learning_rate': 0.01, 'max_depth': -1, 'gamma':5, 'seed':77,
    'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.8, 'bagging_freq':1, 'bagging_seed':77,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'min_split_gain': 0.15, 'min_child_weight': 1, 'min_child_samples': 6, 'scale_pos_weight': 0.872}
num_round =2551
train_data=lgb.Dataset(tempX, Y)
lgb_model = lgb.train(final_params,train_data,num_round)
ypred2=lgb_model.predict(test)
s2 = pd.Series(ypred2, name='is_female')
out_df = pd.concat([s1, s2], axis=1).reset_index()
del out_df['index']
out_df.to_csv('result\lgb_seed77_top150_grouping_no_elimination.csv',index=False)
#######################################################################################################
tempX = tempX[final_list]
test = test[final_list]

final_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'metric':'auc',
    'num_leaves': 40, 'learning_rate': 0.01, 'max_depth': -1, 'gamma':5, 'seed':88,
    'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.8, 'bagging_freq':1, 'bagging_seed':72,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'min_split_gain': 0.15, 'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight': 0.862}
num_round =2551
train_data=lgb.Dataset(tempX, Y)
lgb_model = lgb.train(final_params,train_data,num_round)
ypred2=lgb_model.predict(test)
s2 = pd.Series(ypred2, name='is_female')
out_df = pd.concat([s1, s2], axis=1).reset_index()
del out_df['index']
out_df.to_csv('result\lgb_seed88_top150_selected_grouping.csv',index=False)
##################################################################################################
final_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'metric':'auc',
    'num_leaves': 44, 'learning_rate': 0.01, 'max_depth': -1, 'gamma':5, 'seed':99,
    'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.8, 'bagging_freq':1, 'bagging_seed':72,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 1, 'reg_lambda': 0.2,
    'min_split_gain': 0.05, 'min_child_weight': 1, 'min_child_samples': 2, 'scale_pos_weight': 0.862}
num_round =1674
train_data=lgb.Dataset(tempX, Y)
lgb_model = lgb.train(final_params,train_data,num_round)
ypred2=lgb_model.predict(test)
s2 = pd.Series(ypred2, name='is_female')
out_df = pd.concat([s1, s2], axis=1).reset_index()
del out_df['index']
out_df.to_csv('result\lgb_seed99_top150_selected_grouping.csv',index=False)