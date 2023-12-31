import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
from tools.features import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        # 预处理完后的数据，mean 值，标准差 std 值
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        # 列 为 特征
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        实际迭代模块，会迭代 num_iterations 次
        :param alpha 学习率
        :param  num_iterations 迭代次数
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            # 计算并存储 损失值
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新计算方法，注意是矩阵运算
        """
        # 样本个数
        num_examples = self.data.shape[0]
        # 预测值
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        # 公式中： 预测值 - 真实值   得到残差
        delta = prediction - self.labels
        # 计算 theta 并更新 theta
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
        损失计算方法
        """
        num_examples = data.shape[0]
        # 公式中： 预测值 - 真实值   得到残差
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        # 预测值是 data * thera 公式
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        """
        测试数据损失
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions
