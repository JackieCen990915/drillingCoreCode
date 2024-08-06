
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt


from openpyxl import Workbook



#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np




from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler




param_grid={
        'learning_rate':np.linspace(0,1,21)
    }








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


#调参#
scorel=[]
k=5


#{'max_depth': 18}---0.9865271596936263
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=5)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''


#{'min_child_weight': 5}---0.9873603166097424
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                  max_depth=18,
                  random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=5)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''


#{'gamma': 0.0}---0.9873603166097424--默认
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=5)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'subsample': 0.6000000000000001}----0.9894673801009064
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=5)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'colsample_bytree': 0.25}-----0.9930519695855529
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=5)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''





#{'alpha': 0.0}---0.9930519695855529
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=5)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'reg_lambda': 0.1}---0.9933975300481566
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=5)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'learning_rate': 0.1}---0.9933975300481566
'''
regressor=xgb.XGBRegressor(
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  reg_lambda=0.1,
                  random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=5)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''












regressor=xgb.XGBRegressor(
                  learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  reg_lambda=0.1,
                  random_state=90)



regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)





######################6.评估模型(测试集)######################
MSE_test=metrics.mean_squared_error(y_test, y_test_pred)
R2_test=metrics.r2_score(y_test, y_test_pred)
EV_test=metrics.explained_variance_score(y_test, y_pred)

print('MSE_test:', MSE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)