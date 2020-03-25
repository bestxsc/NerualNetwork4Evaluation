# -*- encoding= utf-8 -*-
import dataLoader
import xscnetwork
import setnparray
import formatData
#读取数据
setnparray.f()
formatData.f()
training_data, validation_data, test_data = dataLoader.load_data_wrapper()
#设置一个具有14个输入层节点，6个输出节点，29个一层隐藏层节点的神经网络
net = xscnetwork.Network([14,7,6])
net.SGD(training_data,30,1,1.0, lmbda=1.0,
        evaluation_data=validation_data, monitor_evaluation_accuracy=True) #用validation_data作为评价数据，并且监控评估的精度