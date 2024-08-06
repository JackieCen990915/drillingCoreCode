
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor



from openpyxl import Workbook



######################1.导入数据######################
df=pd.read_excel('../../../0114_cx_整理数据_17_最终.xlsx',sheet_name='Sheet1')






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



# 使用MinMaxScaler进行归一化------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)






param_grid={
    'learning_rate': np.linspace(0,1,41),
}




#调参#
scorel=[]

k=5
#n_estimators
#{'n_estimators': 201}----0.9817533330914605
'''
regressor=LGBMRegressor(
                   learning_rate=0.1,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''




#{'max_depth': 18}----0.9818521630436967
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''





#{'num_leaves': 91}-----0.9872405075721262
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''




#{'min_data_in_leaf': 1}---0.988489817505646
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   num_leaves=91,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''


#{'max_bin': 90}---0.9889003050112304
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'feature_fraction': 0.5}---0.990569601079493
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''


#{'bagging_fraction': 0.1}---0.990569601079493
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,
                   feature_fraction=0.5,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''




#{'bagging_freq': 0}----0.990569601079493
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,
                   feature_fraction=0.5,
                   bagging_fraction=0.1,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'reg_alpha': 0.25}----0.9908192355263905
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,
                   feature_fraction=0.5,
                   bagging_fraction=0.1,
                   bagging_freq=0,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'reg_lambda': 0.0}---0.9908192355263905
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,
                   feature_fraction=0.5,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0.25,    
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'min_split_gain': 0.0}---0.9908192355263905
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,
                   feature_fraction=0.5,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0.25,
                   reg_lambda=0.0,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'learning_rate': 0.1}-----0.9908192355263905
'''
regressor=LGBMRegressor(n_estimators=201,
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
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''






######################5.使用测试集数据进行预测######################
regressor=LGBMRegressor(n_estimators=201,
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

regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)




######################6.评估模型(测试集)######################
MSE_test=metrics.mean_squared_error(y_test, y_test_pred)
R2_test=metrics.r2_score(y_test, y_test_pred)
EV_test=metrics.explained_variance_score(y_test, y_test_pred)

print('MSE_test:', MSE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)



