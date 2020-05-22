# -*- encoding: utf-8 -*-
# @MoudleName: dataLoader.py
# @Function : Sorting the data for NN
# @Author : XsC
# @Time : 2020/04/18 08:25
"""
加载数据的库。
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    """
    将数据作为包含训练数据、验证数据和测试数据的元组返回。
    “训练数据”作为包含两个项的元组返回。
    第一个项包含实际训练数据。
    “training_data”元组中的第二个条目是元组第一个条目中包含的对应数据的评价值（0-5）。
    “验证数据”和“测试数据”类似.
    这是一种很好的数据格式，但是对于神经网络来说，稍微修改一下“训练数据”的格式是有帮助的。
    这是在包装函数“load_data_wrapper（）”中完成的，请参见下文。
    """
    details = np.loadtxt('details.txt', dtype=int, delimiter=',')
    scores = np.loadtxt('format_scores.txt', dtype=int, delimiter=',')
    training_data = (details, scores)
    validation_data = training_data
    test_data = training_data
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    返回包含`（训练数据、验证数据、测试数据）``的元组。
    基于“加载数据”，但该格式更便于在我们的神经网络实现中使用。
    尤其是，“training_data”是一个2元组“(x，y)”的列表。
    x是一个包含输入数据的numpy.ndarray。y是一个6维numpy.ndarray，表示对应于``x``的评价值的单位向量。
    验证数据``和``测试数据``是类似地（x，y）的列表。在每种情况下，
    x是包含输入数据numpy.ndarry，
    y是相应的分类，即对应于``x``的评价值（整数）。
    显然，这意味着我们对培训数据和验证/测试数据使用的格式略有不同。
    这些格式在我们的神经网络代码中被证明是最方便使用的。
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (14, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (14, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (14, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    返回一个10维单位向量，在第j位上有1.0，其它为0.用于转换数字0~5到相应期望输出网络
    """
    e = np.zeros((6, 1))
    e[j] = 1.0
    return e
