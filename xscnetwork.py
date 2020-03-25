"""xscnetwork.py
~~~~~~~~~~~~~~

实现了前馈神经网络的梯度下降学习算法。改进了包括交叉熵成本函数、正则化，更好地初始化网络权值。

"""

# 标准库
import json
import random
import sys

# 第三方库
import numpy as np


#### 定义了二次函数和交叉熵函数

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """
        返回输出a和所需输出y之间的差距（或着说相关联的成本)
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """
        从输出层返回错误的增量
        """
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        返回与输出“a”和所需输出“y”的差距。
        请注意，np.nan_to_num用于确保数值的稳定性。特别是如果在a和y均有1.0，
        则表达式(1-y)*np.log(1-a)返回nan。np.nan_to_num确保转换为正确的值(0.0)。
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """
        从输出层返回错误增量。注意，该方法不适用参数“z”。它包含在参数中，以便使接口与其他类的delta一致（之前旧版中的delta方法）
        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        列表“size”包含网络各层中的神经元数量。例如[2,3,1]表示一个三层网络。第一层包含2个神经元，第二次3个，第三次1个。
        网络的偏置和权重是随机初始化的，使用“self.default_weight_initializer”
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """
        使用均值为0，标准差为1的高斯分布（正态分布）初始化每个位于连接到同一个神经元的权重数的平方根上的权重和偏置。
        注意，第一层为输入层，不初始化偏置。
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        使用均值为0，标准差为1的高斯分布（正态分布）初始化权重w和偏置b。
        注意，第一层假设为输入层，所以我们不会为这些神经元设置任何偏置。
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):
        """
        利用小批量随机梯度下降训练神经网络。
        ”training_data“是表示训练输入和期望输出元组“(x,y)”的列表。
        其他非可选参数是字节是的，正则化参数lmbda也是如此。
        该方法还接受”evaluation-data“，通常是验证数据或测试数据。
        我们可以通过设置适当的表示来监控评估数据或训练数据的成本和准确性。
        该方法返回一个包含四个列表的元组：评估数据的（每次训练）成本、评估数据的精度、训练数据的成本和训练数据的精度。
        所有的价值都是在每个训练结束阶段评估的。
        比如，我们训练30次，那么元组的第一个元素将是30个元素的列表，其中包含每个阶段结束时评估数据的成本。
        注意，如果未设置相应标志，列表为空。
        """
        # early stopping functionality:
        best_accuracy=1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy=0
        no_accuracy_change=0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        通过使用反向传播对单个子数据集应用梯度下降来更新网络的权重和偏差。
        “mini_batch”是一个元组列表“(x,y)”，“eta”是学习率，“lmbda”是正则化参数，“n”是训练数据集的大小
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        返回一个元组”(nabla_b,nabla_w)“，表示成本函数C_x的梯度。
        “nabla_b”和“nabla_w”是numpy数组的逐层列表，类似”self.biasses“和”self.weights“
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        返回网络输出正确的“data”中的数。神经网络的输出认定为最后一层活性最高的那一个。
        如果数据集是验证或测试数据（默认情况），就把convert标志设置为False，如果是训练数据就设置为True。
        需要这个标志是因为结果y在不同的数据集中的含义不同。
        尤其是它标记着我们是否需要在不同的表示之间进行转换。
        对不同的数据集使用不同的表示可能看起来很奇怪，为什么不对三个数据集使用相同的表示呢？
        这是处于效率的考虑。程序通常会评估训练数据的成本和其他数据集的准确性。
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """
        返回数据集”data“的总开销。
        如果数据集是训练数据，应该把convert设置为False（默认为false），把convert设置为Ture如果是验证或测试数据。
        参见上面类似的（事实上是相反的）”accuracy“方法
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

    def save(self, filename):
        """
        把神经网络保存为“filename”
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### 读取一个神经网络
def load(filename):
    """
    通过”filename“来读取一个神经网络。
    返回一个神经网络实例
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### 其他函数(初期测试使用)
def vectorized_result(j):
    """
    返回一个10维单位向量，在第j位上有1.0，其它为0.用于转换数字0~5到相应期望输出网络
    """
    e = np.zeros((6, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """sigmoid函数"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """对sigmoid函数求导"""
    return sigmoid(z)*(1-sigmoid(z))
