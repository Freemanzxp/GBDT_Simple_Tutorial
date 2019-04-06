"""
Created on ：2019/03/30
@author: Freeman
"""


class SquaresError:

    def initialize_f_0(self, data):
        data['f_0'] = data['label'].mean()
        return data['label'].mean()

    def calculate_residual(self, data, iter):
        res_name = 'res_' + str(iter)
        f_prev_name = 'f_' + str(iter - 1)
        data[res_name] = data['label'] - data[f_prev_name]

    def update_f_m(self, data, trees, iter, learning_rate,logger):
        f_prev_name = 'f_' + str(iter - 1)
        f_m_name = 'f_' + str(iter)
        data[f_m_name] = data[f_prev_name]
        for leaf_node in trees[iter].leaf_nodes:
            data.loc[leaf_node.data_index, f_m_name] += learning_rate * leaf_node.predict_value
        # 打印每棵树的 train loss
        self._get_train_loss(data['label'], data[f_m_name], iter,logger)

    def _get_train_loss(self, y, f, iter,logger):
        loss = ((y - f) ** 2).sum()
        logger.info(('第%d棵树: mse_loss:%.4f' % (iter, loss)))
