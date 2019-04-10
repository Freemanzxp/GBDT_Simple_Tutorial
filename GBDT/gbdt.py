"""
Created on ：2019/03/28
@author: Freeman
"""
import abc
import math
import logging
import pandas as pd
from GBDT.decision_tree import Tree
from GBDT.loss_function import SquaresError, BinomialDeviance
from GBDT.treeplot import printtree
logging.basicConfig(level= logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class BaseGradientBoosting(AbstractBaseGradientBoosting):

    def __init__(self, loss_function, learning_rate, n_trees, max_depth, is_log=False, is_plot=False):
        super().__init__()
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.features = None
        self.trees = {}
        self.f_0 = None
        self.is_log = is_log
        self.is_plot = is_plot

    def fit(self, data):
        """
        :param data: pandas.DataFrame, the features data of train training   
        """
        # 掐头去尾， 删除id和label，得到特征名称
        self.features = list(data.columns)[1: -1]
        # 初始化 f_0(x)
        # 对于平方损失来说，初始化 f_0(x) 就是 y 的均值
        self.f_0 = self.loss_function.initialize_f_0(data)
        # 对 m = 1, 2, ..., M
        logger.setLevel(logging.INFO if self.is_log else logging.CRITICAL)
        for iter in range(1, self.n_trees+1):
            # 计算负梯度--对于平方误差来说就是残差
            logger.info(('-----------------------------构建第%d颗树-----------------------------' % iter))
            self.loss_function.calculate_residual(data, iter)
            self.trees[iter] = Tree(data, self.max_depth, self.features, self.loss_function, iter, logger)
            self.loss_function.update_f_m(data, self.trees, iter, self.learning_rate, logger)
            if self.is_plot:
                printtree(self.trees[iter])


class GradientBoostingRegressor(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth, is_log=False):
        super().__init__(SquaresError(), learning_rate, n_trees, max_depth, is_log)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees+1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_value'] = data[f_m_name]


class GradientBoostingClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth, is_log=False):
        super().__init__(BinomialDeviance(), learning_rate, n_trees, max_depth, is_log)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees + 1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_value'] = data[f_m_name].apply(lambda x: 1 / (1 + math.exp(-x)))