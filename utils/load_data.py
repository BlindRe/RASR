import pickle
import os
import numpy as np

def load_stock_data(data_path, stock_rel_file, stock_id_file, market_name='CASE'):
    # data_path: ../data/CASE/preprocess/
    # stock_rel_file: ../data/CASE/preprocess/stock_relation.npy
    # stock_id_file: ../data/CASE/preprocess/stock2index.pkl
    stock_features = []
    masks = []
    return_ratios = []
    stock2index = pickle.load(open(stock_id_file, 'rb'))
    relation_matrix = np.load(stock_rel_file)
    cnt = 0
    for s_name, sid in stock2index.items():
        cur_stock = np.genfromtxt(
            os.path.join(data_path, s_name+'.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if cnt == 0:
            print('single stock data shape:', cur_stock.shape)
            stock_features = np.zeros([len(stock2index), cur_stock.shape[0], cur_stock.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(stock2index), cur_stock.shape[0]], dtype=np.float32)
            return_ratios = np.zeros([len(stock2index), cur_stock.shape[0]], dtype=np.float32)
            base_price = np.zeros([len(stock2index), cur_stock.shape[0]], dtype=np.float32)
        for row in range(cur_stock.shape[0]):
            if market_name == 'CASE':
                if cur_stock[row][1] < 1e-8:
                    masks[sid][row] = 0.0
                else:
                    return_ratios[sid][row] = cur_stock[row][-2] * 0.2
            else:
                if abs(cur_stock[row][-1] + 1234) < 1e-8:
                    masks[sid][row] = 0.0
                elif row > 0 and abs(cur_stock[row - 1][-1] + 1234) > 1e-8:
                    return_ratios[sid][row] = (cur_stock[row][-2] - cur_stock[row-1][-2]) / cur_stock[row-1][-2]
                for col in range(cur_stock.shape[1]):
                    if abs(cur_stock[row][col] + 1234) < 1e-8:
                        cur_stock[row][col] = 1.1
        
        stock_features[sid, :, :] = cur_stock[:, 1:]
        base_price[sid, :] = cur_stock[:, -2]
        cnt += 1
    
    # stock_features: (stock_num, date_len, feature_num=5)
    # relation_matrix: (stock_num, valid_industry_num)
    # masks: (stock_num, date_len)
    # return_ratios: (stock_num, date_len)
    return stock_features, relation_matrix, masks, return_ratios, base_price