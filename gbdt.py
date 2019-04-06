"""
Created on ：2019/03/28
@author: Freeman
"""
import pandas as pd
from loss_function import SquaresError
from decision_tree import Tree
import logging
logging.basicConfig(level= logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class DecisionTreeRegressor:

    def __init__(self, loss_function, learning_rate, n_trees, max_depth, verbose=0, is_log = True):

        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.verbose = verbose
        self.features = None
        self.trees = {}
        self.f_0 = None
        self.is_log = is_log

    def fit(self, data):
        '''
        :param x: pandas.DataFrame, the features data of train training  
        :param y: list, the label of training
        '''
        # 去头掐尾， 删除id和label，得到特征名称
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
            self.trees[iter] = Tree(data, self.max_depth, self.features, iter,logger)
            self.loss_function.update_f_m(data, self.trees, iter, self.learning_rate)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees+1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_value'] = data[f_m_name]


if __name__ == '__main__':

    data = pd.DataFrame(data=[[1, 5, 20, 1.1],
                        [2, 7, 30, 1.3],
                        [3, 21, 70, 1.7],
                        [4, 30, 60, 1.8],
                        ], columns=['id', 'age', 'weight', 'label'])
    loss_function = SquaresError()
    model = DecisionTreeRegressor(loss_function=loss_function, learning_rate=0.1, n_trees=10, max_depth=2)
    model.fit(data)
    logger.info(data)
    test_data = pd.DataFrame(data=[[5, 25, 65],
                      ], columns=['id', 'age', 'weight'])
    model.predict(test_data)
    print(test_data['predict_value'])