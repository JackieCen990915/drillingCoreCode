from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler 


import pandas as pd


from openpyxl import Workbook

######################1.导入数据######################
df=pd.read_excel('../../../../0114_cx_整理数据_17_最终.xlsx',sheet_name='Sheet1')








######################2.提取特征变量######################
x=df.drop(columns='ROP ')
y=df['ROP ']


print("---------------x------------------")
print(x)
print(type(x))
print("---------------y------------------")
print(y)
print(type(y))






######################3.划分训练集和测试集######################
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)



#7378*0.8=5,902.4----5900吧，5900*0.5=2,950
##### 将异常值添加到训练集#####################################
##### 将异常值添加到前半部分的训练集---0，2949
random_numbers1=[ 100,1820,1547,17,1297,2275,1562,2205,318,678,2402,49,1372,791,2642,1735,2702,147,14,2303,487,2741,1121,91,669,657,473,71,159,1086,2246,707,954,1983,1224,846,2310,331,786,2525,385,1880,1159,2241,1072,2091,1223]
#print("前半部分未加噪声：",y_train[random_numbers1])
#随机倍数----相对误差大于30%。
random_numbers2=[0.2,0.65432606, 0.17937478, 0.22490822, 0.56186422, 0.22982631, 0.28179949, 0.23646807, 0.00178915, 0.12826151, 0.62982715, 0.60540484, 0.05067764, 0.53717476, 0.13289582, 0.12587697, 0.35522921, 0.32366063, 0.14896754, 0.18936439, 0.61110684, 0.28142689, 0.34374534, 0.09940323, 0.05802156, 0.28809496, 0.49137363, 0.57474047, 0.45294577, 0.02659484, 0.33497032, 0.44840771, 0.22114177, 0.17435798, 0.04240528, 0.41259738, 0.40259325, 0.68896999, 0.4820013 , 0.03723863, 0.34413224, 0.61365729, 0.66700671, 0.42810014, 0.31822617, 0.6876867 , 0.23067058]

y_train[random_numbers1]=random_numbers2*y_train[random_numbers1] 
#print("前半部分加噪声：",y_train[random_numbers1])


##### 将异常值添加到后半部分的训练集---2950，5900
random_numbers1=[4432,5895,3478,5635,4858,4430,4650,4577,5037,3091,5315,3212,3909,4731,3281,3577,5156,3869,5062,3207,4046,4717,4439,3582,3093,3328,2984,5795,4321,5469,3615,5459,3943,4817,5475,3645,3327,5084,5199,3717,3810,5857,5388,4427,5103,4469,5123,3109,4504,4599,5106,4364,4016]
#print("后半部分未加噪声：",y_train[random_numbers1])
#随机倍数----相对误差大于30%。
random_numbers2=[ 1.66245202, 1.58635779, 1.5076548 , 1.70339894, 1.67127236, 1.54832122, 1.91718196, 1.45179328, 1.62401413, 1.47706874, 1.77688032, 1.43676063, 1.54422687, 1.88961346, 1.95056438, 1.49308494, 1.5875819 , 1.91267702, 1.83997505, 1.56359015, 1.52631677, 1.52419464, 1.70757713, 1.32292622, 1.31323617, 1.33906655, 1.49019838, 1.47591269, 1.95776641, 1.41870291, 1.84148094, 1.91308036, 1.3694189 , 1.3773767 , 1.96814169, 1.5338931 , 1.33171993, 1.50378871, 1.49239457, 1.68174463, 1.30853376, 1.95309151, 1.36144316, 1.73630888, 1.39458107, 1.42731221, 1.37326974, 1.78478994, 1.87476055, 1.96727037, 1.45713648, 1.97292137, 1.49979058]
y_train[random_numbers1]=random_numbers2*y_train[random_numbers1] 
#print("后半部分未加噪声：",y_train[random_numbers1])



# 使用MinMaxScaler进行归一化------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

 
 


clf1=RandomForestRegressor(n_estimators=181,
                           max_depth=20,
                           max_features=4,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           random_state=90)
clf2=LGBMRegressor(n_estimators=201,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,
                   feature_fraction=0.5,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0.25,
                   reg_lambda=0.0,
                   min_split_gain=0,  
                   random_state=90)

clf3=xgb.XGBRegressor(
                  learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  reg_lambda=0.1,
                  random_state=90)


clf4=SVR(kernel='rbf',C=100,epsilon=0.8,gamma=1)

# 软投票
estimators=[ ('rf',clf1),('lgbm',clf2),( 'xgb',clf3),( 'svr',clf4)]
final_estimator=LinearRegression()

eclf=StackingRegressor(estimators=estimators,
                       final_estimator=final_estimator)



eclf.fit(x_train,y_train)
y_pred = eclf.predict(x_test)







######################6.评估模型######################
MSE=metrics.mean_squared_error(y_test, y_pred)
R2=metrics.r2_score(y_test, y_pred)
EV=metrics.explained_variance_score(y_test, y_pred)

print('MSE:', MSE)
print('r2_score:', R2)
print('EV:', EV)

