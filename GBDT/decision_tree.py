"""
Created on ：2019/03/30
@author: Freeman
"""


class Node:
    def __init__(self, data_index, logger=None, split_feature=None, split_value=None, is_leaf=False, loss_function=None, target_name =None, deep = None ):
        self.loss_function = loss_function
        self.split_feature = split_feature
        self.split_value = split_value
        self.data_index = data_index
        self.is_leaf = is_leaf
        self.predict_value = None
        self.left_child = None
        self.right_child = None
        self.logger = logger
        self.name = target_name
        self.deep = deep

    def update_predict_value(self, targets, y):
        self.predict_value = self.loss_function.update_leaf_values(targets, y)
        self.logger.info(('叶子节点预测值：', self.predict_value))

    def get_predict_value(self, instance):
        if self.is_leaf:
            self.logger.info(('predict:', self.predict_value))
            return self.predict_value
        if instance[self.split_feature] < self.split_value:
            return self.left_child.get_predict_value(instance)
        else:
            return self.right_child.get_predict_value(instance)


class Tree:
    def __init__(self, data, max_depth, min_samples_split, features, loss_function, iter, logger):
        self.loss_function = loss_function
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = features
        self.logger = logger
        self.target_name = 'res_' + str(iter)
        self.remain_index = [True] * len(data)
        self.leaf_nodes = []
        self.root_node = self.build_tree(data, self.remain_index, target_name = self.target_name,depth=0)

    def build_tree(self, data, remain_index, target_name, depth=0):
        now_data = data[remain_index]
        # 如果 深度没有到达最大 以及 节点样本数>=min_samples_split 就继续生长
        # 否则 停止生长 变成叶子节点
        if depth < self.max_depth and len(now_data) >= self.min_samples_split:
            mse = None
            split_feature = None
            split_value = None
            left_index_of_now_data = None
            right_index_of_now_data = None
            self.logger.info(('--树的深度：%d' % depth))
            for feature in self.features:
                self.logger.info(('----划分特征：', feature))
                feature_values = now_data[feature].unique()
                for fea_val in feature_values:
                    # 尝试划分
                    left_index = list(now_data[feature] < fea_val)
                    right_index = list(now_data[feature] >= fea_val)
                    left_mse = self._calculate_mse(now_data[left_index][self.target_name])
                    right_mse = self._calculate_mse(now_data[right_index][self.target_name])
                    sum_mse = left_mse + right_mse
                    self.logger.info(('------划分值:%.3f,左节点损失:%.3f,右节点损失:%.3f,总损失:%.3f' %
                          (fea_val, left_mse, right_mse, sum_mse)))
                    if mse is None or sum_mse < mse:
                        split_feature = feature
                        split_value = fea_val
                        mse = sum_mse
                        left_index_of_now_data = left_index
                        right_index_of_now_data = right_index
            self.logger.info(('--最佳划分特征：', split_feature))
            self.logger.info(('--最佳划分值：', split_value))

            node = Node(remain_index, self.logger, split_feature, split_value, target_name=target_name, deep=depth)
            # trick for DataFrame, index revert
            left_index_of_all_data = []
            for i in remain_index:
                if i:
                    if left_index_of_now_data[0]:
                        left_index_of_all_data.append(True)
                        del left_index_of_now_data[0]
                    else:
                        left_index_of_all_data.append(False)
                        del left_index_of_now_data[0]
                else:
                    left_index_of_all_data.append(False)

            right_index_of_all_data = []
            for i in remain_index:
                if i:
                    if right_index_of_now_data[0]:
                        right_index_of_all_data.append(True)
                        del right_index_of_now_data[0]
                    else:
                        right_index_of_all_data.append(False)
                        del right_index_of_now_data[0]
                else:
                    right_index_of_all_data.append(False)

            node.left_child = self.build_tree(data, left_index_of_all_data, target_name, depth + 1)
            node.right_child = self.build_tree(data, right_index_of_all_data, target_name, depth + 1)
            return node
        else:
            node = Node(remain_index, self.logger, is_leaf=True, loss_function=self.loss_function,target_name=target_name, deep=depth)
            node.update_predict_value(now_data[self.target_name], now_data['label'])
            self.leaf_nodes.append(node)
            return node

    def _calculate_mse(self, label):
        mean = label.mean()
        error = 0
        for y in label:
            error += (y - mean) * (y - mean)
        return error
