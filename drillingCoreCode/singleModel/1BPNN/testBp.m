%% 清空环境变量
warning off % 关闭报警信息
close all % 关闭开启的图窗
clear % 清空变量
clc % 清空命令行


%% 设置随机种子
rng(100); % 在这里使用任何整数，100只是一个示例种子值




%% 导入数据
x_train_data=xlsread('D:\A研2项目\课题论文\3钻井预测\实战\预测代码\单模型\1BPNN\0203前\实现BP\实战matlab\Data\x_train.xlsx');
y_train_data=xlsread('D:\A研2项目\课题论文\3钻井预测\实战\预测代码\单模型\1BPNN\0203前\实现BP\实战matlab\Data\y_train.xlsx');
x_test_data=xlsread('D:\A研2项目\课题论文\3钻井预测\实战\预测代码\单模型\1BPNN\0203前\实现BP\实战matlab\Data\x_test.xlsx');
y_test_data=xlsread('D:\A研2项目\课题论文\3钻井预测\实战\预测代码\单模型\1BPNN\0203前\实现BP\实战matlab\Data\y_test.xlsx');




%%  划分训练集和测试集
train_feature_num=16;

% 训练集
x_train=x_train_data(1:end,1:train_feature_num)';
y_train=y_train_data(1:end,1)';
M=size(x_train,2);

% 测试集
x_test=x_test_data(1:end,1:train_feature_num)';
y_test=y_test_data(1:end,1)';
N=size(x_test,2);




%% 数据归一化

[x_train_MinMaxScaler,ps_input]=mapminmax(x_train,0,1)
[y_train_MinMaxScaler,ps_output]=mapminmax(y_train,0,1)

x_test_MinMaxScaler=mapminmax('apply',x_test,ps_input)
y_test_MinMaxScaler=mapminmax('apply',y_test,ps_output)




%% 创建网络
% sqrt(16+1)+1~10=6~15
s1=13; %隐藏层结点个数
net=newff(x_train_MinMaxScaler,y_train_MinMaxScaler,s1,{'tansig','purelin'},'trainlm');

%% 参数设置
net.trainParam.epochs=1000; % % % % % % 最大迭代次数
net.trainParam.goal=1e-6;% % % % % % % 目标训练误差
net.trainParam.lr=0.001;% 学习率

%% 训练网络
net=train(net,x_train_MinMaxScaler,y_train_MinMaxScaler);

%% 仿真测试
y_pred_MinMaxScaler=sim(net,x_test_MinMaxScaler);

%% 数据反归一化
y_pred=mapminmax('reverse',y_pred_MinMaxScaler,ps_output);

disp(['y_test: ', num2str(y_test)]);
disp(['y_pred: ', num2str(y_pred)]);


%% 性能评价
MSE=mean((y_pred-y_test).^2);
ssres=sum((y_pred-y_test).^2);
sstotal=sum((y_test-mean(y_test)).^2);
R2=1-ssres/sstotal;


disp(['均方误差 (MSE): ', num2str(MSE)]);
disp(['决定系数 (R2): ', num2str(R2)]);

% 计算Explanatory Variance
ev = explanatoryVariance(y_test, y_pred);
disp(['Explanatory Variance: ', num2str(ev)]);


function ev = explanatoryVariance(y_true, y_pred)
    % 计算Residual Variance
    residual_variance = var(y_true - y_pred);

    % 计算Total Variance
    total_variance = var(y_true);

    % 计算Explanatory Variance
    ev = 1 - residual_variance / total_variance;
end
