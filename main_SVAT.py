import argparse
import torch

from experiments.exp_svat import Exp_SVAT
from utils.optimize import set_seed

gpus = 0 if torch.cuda.is_available() else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
                        help="name of dataset",
                        type=str)
    parser.add_argument("hist_len",
                        help="length of history",
                        type=int)
    sys_args = parser.parse_args()
    sys_data = sys_args.dataset
    sys_hist_len = sys_args.hist_len
    
    d_path = './data/' + sys_data + '/preprocess/'
    rel_file = './data/' + sys_data + '/relation/stock_relation.npy'
    id_file = './data/' + sys_data + '/preprocess/stock2index.pkl'
    stock2num = {'NASDAQ': 1026, 'NYSE': 1737, 'CASE': 4465}
    stock2adv_eps = {'NASDAQ': 0.005, 'NYSE': 0.001, 'CASE': 0.005}
    stock2kl_lambdas = {'NASDAQ': 0.5, 'NYSE': 1.0, 'CASE': 1.0}
    

    param_dict = dict(
        data_path = [d_path],
        stock_rel_file = [rel_file],
        stock_id_file = [id_file],
        market_name = [sys_data],
        checkpoint_dir = ['./SVAT_checkpoints/' + sys_data + '/'],
        history_len = [sys_hist_len],
        num_stocks = [stock2num[sys_data]],
        fea_dim = [5],
        hid_size = [32],
        drop_rate = [0.1],
        z_dim = [32],
        adv_hid_size = [128],
        learning_rate = [0.0001],
        adv_lr = [0.0001],
        adv_eps = [stock2adv_eps[sys_data]],
        kl_lambda = [1.],
        reg_alpha = [0.5],
        adv = ['Attention'],
        valid_index = [1008],
        text_index = [None],
        epochs = [500],
        lradj = ['decayed'],
        patience = [2],
        clip = [10.],
        sample_size = [50],
        rank_sign = [True]
    )

    seed = 42
    set_seed(seed)
    exp = Exp_SVAT(param_dict, gpus=gpus) # set experiments
    exp.train()

    torch.cuda.empty_cache()