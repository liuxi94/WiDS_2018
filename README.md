# WiDS_2018
First, we would like to give a reference list, which contains the source of codes written by others but are used by us in this competition.

I. MeanEncoder.py in folder ‘part1’. It is written by kaggler: https://www.kaggle.com/somnisight. And he posted the code of MeanEncoder.py in zhihu.com: https://zhuanlan.zhihu.com/p/26308272.

II. get_stats function in lgb_with_grouping.py in folder ‘part2’. It is written by kaggler: https://www.kaggle.com/xiaozhouwang. And the code is posted on: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32123.

III. kaggle_avg.py in folder ‘helper’ comes from github: https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/src/kaggle_avg.py.

IV. blend.py in folder ‘avg_result’ comes from github: https://github.com/kaz-Anova/StackNet/blob/master/example/example_amazon/blend_script.py.


Model Generation Procedure.
1. Part 1
Run lgb_simplified_fe.py in folder ‘part1’, then run lgb_using_meanencoder.py in the same folder. Four prediction files would be generated in folder ‘result’ inside folder ‘part1’. Run kaggle_avg.py to average the result from these four files(this github page shows how to run kaggle_avg.py https://github.com/MLWave/Kaggle-Ensemble-Guide). The generated file with averaging result is called ‘avg1’ in folder ‘avg_result’.

2. Part 2
Run Feature_selection_xgb.py in folder ‘part2’ to generate new train and test file. It takes about 20 minutes on a dual core laptop and you don’t have to run it again because we have also put the generated files inside folder ‘part2’. These two files are called new_train_150.csv and new_test_150.csv. Then run lgb_with_grouping.py and three prediction files are generated in ‘result’ folder inside folder ‘part2’.

3. Averaging and blending
First averaging ‘lgb_seed77_top150_grouping_no_elimination.csv’ with ‘lgb_seed99_top150_selected_grouping.csv’, which produces ‘avg2.csv’ in folder ‘avg_result’. Then averaging ‘avg2.csv’ with ‘lgb_seed88_top150_selected_grouping.csv’ and produces ‘avg3.csv’ in folder ‘avg_result’. At last, run ‘blend.py’ in folder ‘avg_result’ and the final submission file ‘blend_avg1_avg3_with_rankdata.csv’ is generated. 

By,

Xi and Snorlax
