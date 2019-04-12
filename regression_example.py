"""
Created on ：2019/04/07
@author: Freeman
"""
import logging
import pandas as pd
from GBDT.gbdt import GradientBoostingRegressor
from GBDT.loss_function import SquaresError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

if __name__ == '__main__':

    data = pd.DataFrame(data=[[1, 5, 20, 1.1],
                        [2, 7, 30, 1.3],
                        [3, 21, 70, 1.7],
                        [4, 30, 60, 1.8],
                        ], columns=['id', 'age', 'weight', 'label'])
    loss_function = SquaresError()
    model = GradientBoostingRegressor(learning_rate=0.1, n_trees=10, max_depth=3,
                                      min_samples_split=2, is_log=False, is_plot=True)
    model.fit(data)
    logger.removeHandler(logger.handlers[-1])
    logger.addHandler(logging.FileHandler('results/result.log'.format(iter),mode='w',encoding='utf-8'))
    logger.info(data)
    test_data = pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])
    model.predict(test_data)
    logger.setLevel(logging.INFO)
    logger.info((test_data['predict_value']))