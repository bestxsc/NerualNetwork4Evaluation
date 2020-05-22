# -*- encoding: utf-8 -*-
# @MoudleName: runMe.py
# @Function : run the nerual network (for testing)
# @Author : XsC
# @Time : 2020/4/28 15:17
import dataLoader
import xscnetwork
import data

# 读取数据
data.getData('datas.csv')
training_data, validation_data, test_data = dataLoader.load_data_wrapper()
# 设置一个具有14个输入层节点，6个输出节点，15个一层隐藏层节点的神经网络并进行训练,训练完成后保存
"""
net = xscnetwork.Network([14,15,6])
net.SGD(training_data,300,10,0.01, lmbda=10.0,
        evaluation_data=validation_data, monitor_evaluation_accuracy=True,
        monitor_training_cost=True) #用validation_data作为评价数据，并且监控评估的精度
net.save("xscNetwork")
#读取神经网络
"""
net = xscnetwork.load("xscNetwork")
net.getResults(test_data)
data.getResult('datas.csv')
