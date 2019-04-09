"""
Created on ：2019/04/09
@author: Freeman
"""
import logging
import pandas as pd
from GBDT.gbdt import GradientBoostingClassifier
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if __name__ == '__main__':

    data = pd.DataFrame(data=[[1, 5, 20, 0],
                        [2, 7, 30, 0],
                        [3, 21, 70, 1],
                        [4, 30, 60, 1],
                        ], columns=['id', 'age', 'weight', 'label'])
    model = GradientBoostingClassifier(learning_rate=0.1, n_trees=10, max_depth=2, is_log=False)
    model.fit(data)
    logger.info(data)
    test_data = pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])
    model.predict(test_data)
    logger.setLevel(logging.INFO)
    logger.info((test_data['predict_value']))