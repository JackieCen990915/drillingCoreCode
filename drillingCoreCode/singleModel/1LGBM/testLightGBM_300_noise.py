
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






#7378*0.8=5,902.4----5900吧，5900*0.5=2,950
##### 将异常值添加到训练集#####################################
##### 将异常值添加到前半部分的训练集---0，2949
random_numbers1=[2480,888,1942,2,777,446,1349,1784,301,1913,1466,431,2338,1374,954,2716,129,1806,2433,198,2628,41,1894,1353,596,815,83,821,1543,2940,2331,401,8,2532,1614,2576,1333,760,2427,806,393,2194,1182,465,2695,1328,1657,2656,931,2481,2366,1019,2085,1300,1302,1257,2720,2518,1230,1110,2609,2221,69,1888,2531,2890,456,1687,737,969,1060,686,253,1142,2810,2784,1055,2745,2283,2884,2160,436,455,172,2498,2206,876,1835,1609,1436,1547,2050,271,184,678,2585,67,679,2400,2012,2207,214,1229,2914,180,1261,866,492,2586,2705,676,737,2673,1991,2122,39,575,927,1751,2797,351,1942,1266,2252,454,2142,1923,2798,1392,2783,2395,2376,1723,1618,468,2876,1827,1992,1814,2915,1301,1737,2899,2858,2159,1891,674,756,975,2400,390,950,1995,1980,1497,1686,1903,159,2941,1451]
#print("前半部分未加噪声：",y_train[random_numbers1])
#随机倍数----相对误差大于30%。
# 生成形状为 (160, 1) 的[0, 0.7)范围内的随机浮点数数组
random_numbers2=[3.23848973e-01,3.06742743e-01,4.27788030e-01,2.39080392e-01,4.95087356e-01,2.71297312e-01,2.22874749e-01,2.27020237e-01,3.11911356e-01,3.32265819e-01,1.64064571e-01,6.34122307e-01,5.56904176e-01,1.61513925e-01,1.32472894e-01,2.77801481e-01,3.63777924e-01,3.18368231e-01,7.18051702e-03,1.92173581e-01,1.15946191e-01,4.40784034e-01,2.80259928e-01,5.43862265e-01,1.68756184e-02,2.65771431e-01,4.81707897e-01,3.08653325e-01,1.16577460e-01,5.72697815e-01,5.36637662e-01,4.65311801e-01,2.13994429e-01,3.98724327e-01,6.84529448e-01,5.13530289e-01,4.61260749e-02,6.90739838e-01,2.52612665e-01,1.04481086e-01,4.24907845e-01,1.95523850e-01,5.64356728e-01,4.14935588e-01,6.45648681e-01,1.70663685e-01,4.78584835e-01,1.66551794e-01,4.69754321e-02,1.00954209e-01,2.03693407e-01,5.88171031e-01,1.00720830e-01,6.20342061e-01,5.72071911e-01,4.56241017e-01,5.86413256e-01,2.37246724e-01,5.82120118e-03,4.25126702e-01,1.92246028e-01,3.60698686e-01,7.83783020e-02,5.60086353e-01,5.42835814e-01,1.14671508e-01,2.57850938e-01,6.24219363e-01,1.06399375e-01,3.14745092e-02,6.88991114e-01,3.93478248e-01,4.16154765e-01,4.85158722e-01,4.89650143e-01,3.26492852e-01,6.76995503e-03,1.30027093e-01,1.84923722e-01,3.14774115e-01,6.25923263e-01,6.81474697e-01,5.20346236e-01,5.01441872e-01,6.06148823e-01,2.25750990e-01,3.37175380e-01,1.08549801e-01,5.26646308e-01,5.18364012e-01,5.07116757e-01,6.10172558e-01,5.98990576e-01,5.67740073e-01,6.42297247e-01,4.65368407e-02,5.85222684e-01,5.53884313e-01,2.95672718e-01,1.62892025e-01,6.95776486e-01,6.38024032e-01,3.67848559e-01,2.63758402e-01,2.97149257e-01,8.62095638e-04,5.74588748e-01,6.74638870e-04,1.01856810e-01,3.07743841e-01,4.64781840e-01,5.42833335e-02,1.33609373e-01,2.56455440e-01,1.48471008e-01,4.29799311e-01,3.54211604e-01,4.33062590e-01,1.38690907e-01,1.85543617e-02,9.23872532e-02,1.76667134e-01,5.43535070e-01,4.55348670e-01,6.19992229e-01,9.03036758e-02,6.46513113e-01,4.22874651e-01,5.66269548e-01,3.77662386e-01,5.16194149e-01,3.10418914e-01,3.35087615e-01,1.75828565e-01,4.41672034e-01,2.06300013e-01,3.85710297e-01,5.34978685e-01,5.67409635e-01,4.61249904e-01,3.35175096e-01,5.12532222e-01,5.36597998e-01,2.73545337e-01,2.69902097e-01,4.99628439e-01,3.51799926e-01,4.04386613e-01,1.72424568e-01,6.04389683e-01,6.28734282e-01,6.13691189e-01,4.42032631e-01,1.15802551e-01,6.72905615e-01,7.89853702e-02,3.00965823e-01,2.94514375e-01,4.30902285e-01,4.96740314e-01]
y_train[random_numbers1]=random_numbers2*y_train[random_numbers1] 
#print("前半部分加噪声：",y_train[random_numbers1])




##### 将异常值添加到后半部分的训练集---2950，5900
random_numbers1=[3729,4782,3683,3421,5522,3125,5588,5494,5635,5175,4973,4522,4117,4412,3759,4170,3964,3961,4109,3792,5186,3564,5136,4975,4523,3419,4417,4246,3749,5673,4131,4174,4935,3772,3250,3140,5442,5487,3851,3218,5209,4089,3717,4589,4144,4081,3810,5434,4432,5112,5417,5313,5163,3024,4659,4828,3891,4222,4174,5846,4210,3092,4619,5801,4052,3948,5325,4764,5413,5721,4403,5240,4670,3964,4585,5164,4480,4796,3454,4311,3884,5136,3327,3840,5257,3562,3026,3174,3155,4823,2996,5263,5458,3273,2952,4873,5431,3068,4017,5299,3800,4561,3533,3666,5591,5652,3021,5644,5432,3929,4974,4109,5818,4505,5682,3101,3722,4841,4977,4731,4125,5801,3204,3310,5700,4145,3752,3917,5829,5001,3374,4948,5889,5888,3998,3087,5100,5273,4227,4377]
#print("后半部分未加噪声：",y_train[random_numbers1])
#随机倍数----相对误差大于30%。
# 生成形状为 (140, 1) 的[0, 0.7)范围内的随机浮点数数组
random_numbers2= [ 1.68003583, 1.61848143, 1.82350095, 1.94697332, 1.91455209, 1.92829947, 1.835378  , 1.48468755, 1.58772934, 1.72997722, 1.93512538, 1.70241231, 1.89685594, 1.85571571, 1.80284681, 1.52973089, 1.96666582, 1.91121743, 1.45785166, 1.34908474, 1.86432511, 1.72153563, 1.95246345, 1.8049673 , 1.75952941, 1.75093239, 1.57997531, 1.82618466, 1.99195931, 1.34166641, 1.79705681, 1.81191651, 1.77728055, 1.9512211 , 1.91595681, 1.3932703 , 1.74746985, 1.52413926, 1.86043782, 1.44614056, 1.33466042, 1.90054033, 1.51477524, 1.81733449, 1.69502303, 1.69220414, 1.58603361, 1.8460315 , 1.97511163, 1.77369609, 1.80012263, 1.50857263, 1.6843795 , 1.32103244, 1.64389902, 1.72282239, 1.57226613, 1.92913742, 1.66219351, 1.49886378, 1.82793226, 1.52608059, 1.65350036, 1.9376756 , 1.71687123, 1.88808754, 1.43453765, 1.36159155, 1.83432129, 1.39408016, 1.56409287, 1.96388401, 1.44254169, 1.92809533, 1.71783431, 1.55887224, 1.80652867, 1.30836573, 1.76019637, 1.60494556, 1.54049033, 1.32915577, 1.42426428, 1.73139758, 1.55729706, 1.39214999, 1.53570475, 1.43044059, 1.51473227, 1.53023626, 1.72639175, 1.62000778, 1.34123776, 1.84847664, 1.91001952, 1.9375036 , 1.8646492 , 1.8719561 , 1.72283157, 1.55783949, 1.44123603, 1.89144653, 1.94597092, 1.36630285, 1.59121322, 1.74290838, 1.99591493, 1.76573934, 1.82311904, 1.98772126, 1.6812243 , 1.55806218, 1.71309165, 1.59269718, 1.42917829, 1.8067828 , 1.99931802, 1.46178011, 1.9450033 , 1.38395181, 1.8751847 , 1.46345825, 1.43835529, 1.74149792, 1.38593215, 1.79337244, 1.90787809, 1.99543535, 1.44207112, 1.65592863, 1.80011069, 1.36656167, 1.80095207, 1.82227447, 1.31165131, 1.4223794 , 1.35985004, 1.73335289, 1.71407402, 1.53085092]
y_train[random_numbers1]=random_numbers2*y_train[random_numbers1] 
#print("后半部分未加噪声：",y_train[random_numbers1])










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
regressor=LGBMRegressor(learning_rate=0.1,
                   n_estimators=201,
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
y_pred = regressor.predict(x_test)
print("---------------x_train------------------")
print(x_train)
print("---------------y_train------------------")
print(y_train)
print("---------------x_test------------------")
print(x_test)
print("---------------y_test------------------")
print(y_test)
print("---------------y_pred------------------")
print(y_pred)






######################6.评估模型######################
MSE=metrics.mean_squared_error(y_test, y_pred)
R2=metrics.r2_score(y_test, y_pred)
EV=metrics.explained_variance_score(y_test, y_pred)

print('MSE:', MSE)
print('r2_score:', R2)
print('EV:', EV)
