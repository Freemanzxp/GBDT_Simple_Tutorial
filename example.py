import argparse
import pandas as pd
from GBDT.gbdt import GradientBoostingRegressor, GradientBoostingBinaryClassifier, GradientBoostingMultiClassifier
import logging
import os
import shutil
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.removeHandler(logger.handlers[0])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)



def getdata(model):

    dic = {}
    dic['regression'] = [pd.DataFrame(data=[[1, 5, 20, 1.1],
                                  [2, 7, 30, 1.3],
                                  [3, 21, 70, 1.7],
                                  [4, 30, 60, 1.8],
                                  ], columns=['id', 'age', 'weight', 'label']),
                         pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]
    dic['binary_cf'] = [pd.DataFrame(data=[[1, 5, 20, 0],
                        [2, 7, 30, 0],
                        [3, 21, 70, 1],
                        [4, 30, 60, 1],
                        ], columns=['id', 'age', 'weight', 'label']),
                        pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]
    dic['multi_cf'] = [pd.DataFrame(data=[[1, 5, 20, 0],
                        [2, 7, 30, 0],
                        [3, 21, 70, 1],
                        [4, 30, 60, 1],
                        [4, 30, 60, 3],
                        [4, 30, 70, 3],
                        ], columns=['id', 'age', 'weight', 'label']),
               pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]
    return dic[model]


def run(args):
    model =None
    data = getdata(args.model)[0]
    test_data = getdata(args.model)[1]
    if not os.path.exists('results'):
        os.makedirs('results')
    if len(os.listdir('results')) > 0:
        shutil.rmtree('results')
        os.makedirs('results')
    if args.model == 'regression':
        model = GradientBoostingRegressor(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
                                          min_samples_split=args.count, is_log=args.log, is_plot=args.plot)
    if args.model == 'binary_cf':
        model = GradientBoostingBinaryClassifier(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
                                                 is_log=args.log, is_plot=args.plot)
    if args.model == 'multi_cf':
        model = GradientBoostingMultiClassifier(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth, is_log=args.log,is_plot=False)
    model.fit(data)
    logger.removeHandler(logger.handlers[-1])
    logger.addHandler(logging.FileHandler('results/result.log'.format(iter), mode='w', encoding='utf-8'))
    logger.info(data)
    model.predict(test_data)
    logger.setLevel(logging.INFO)
    # logger.info((test_data['predict_value']))
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GBDT-Simple-Tutorial')
    parser.add_argument('--model', default='regression', help='the model you want to use',
                        choices=['regression', 'binary_cf', 'multi_cf'])
    parser.add_argument('--lr', default=0.1, type=int, help='learning rate')
    parser.add_argument('--trees', default=10, type=int, help='the number of decision trees')
    parser.add_argument('--depth', default=3, type=int, help='the max depth of decision trees')
    # 非叶节点的最小数据数目，如果一个节点只有一个数据，那么该节点就是一个叶子节点，停止往下划分
    parser.add_argument('--count', default=2, type=int, help='the min data count of a node')
    parser.add_argument('--log', default=True, type=bool, help='whether to print the log on the console')
    parser.add_argument('--plot', default=False, type=bool, help='whether to plot the decision trees')
    args = parser.parse_args()
    run(args)
    pass
