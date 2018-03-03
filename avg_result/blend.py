# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:46:48 2017

Scipt to rank (weight) average the 2 prediction files
produced from stacknet

It creates a file to be submitted on kaggle
"""
import pandas as pd
from scipy.stats import rankdata
#name of the prediction files produced by the 2 StackNets
avg1=pd.read_csv("avg1.csv")
avg3=pd.read_csv("avg3.csv")
avg1.sort_values('test_id', inplace=True)
avg3.sort_values('test_id', inplace=True)
#blend weights
weight_avg1_model=0.4
weight_avg3_model=0.6
##load data
avg1_preds=avg1['is_female'].ravel()
avg3_preds=avg3['is_female'].ravel()
#create ranks
avg1_preds=rankdata(avg1_preds, method='min') 
avg3_preds=rankdata(avg3_preds, method='min') 
#divide with length to make certain all values are between [0,1]
avg1_preds=avg1_preds/float(len(avg1_preds))
avg3_preds=avg3_preds/float(len(avg3_preds))
# rank average them based on the pre-defined weights
preds=avg1_preds*weight_avg1_model + avg3_preds*weight_avg3_model
#Create the submission file
submission_file=open("blend_avg1_avg3_with_rankdata.csv","w")
submission_file.write("test_id,is_female\n")
for i in range (0,len(preds)):
    submission_file.write("%d,%f\n" %(i,preds[i]))
submission_file.close()
print("Done!")



