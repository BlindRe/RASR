import math
import copy
import heapq
from tqdm import tqdm
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score
from empyrical.stats import max_drawdown, downside_risk, calmar_ratio

def evaluate(prediction, ground_truth, mask, topk=5):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask) ** 2. / np.sum(mask)
    bt_long5 = 1.0
    sharpe_li5 = []
    ndcg_score_top5 = []

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top5 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top5) < topk:
                gt_top5.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])
        pre_top5 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < topk:
                pre_top5.add(cur_rank)
        if topk == 1:
            ndcg_score_top5.append(0.)
        else:
            ndcg_score_top5.append(ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1)))
        
        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= topk
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)
    
    performance['ndcg_score_top5'] = np.mean(np.array(ndcg_score_top5))
    performance['btl5'] = bt_long5 - 1
    sharpe_li5 = np.array(sharpe_li5)
    performance['btl5_array'] = sharpe_li5
    performance['sharpe5'] = ((np.mean(sharpe_li5) - (0.018/365.0))/np.std(sharpe_li5))*15.87 #To annualize
    return performance

def ranking_entropy(rank_records, M):
    # rank_records: (stock_num, stock_num)
    rank_prob = rank_records / M
    stock_num = rank_records.shape[0]
    entropy = np.zeros((stock_num,))
    for stock in range(stock_num):
        prob_index = np.argwhere(rank_prob[stock, :] > 1e-8)
        cur_en = rank_prob[stock, prob_index] * np.log(rank_prob[stock, prob_index])
        entropy[stock] = -1. * np.sum(cur_en)
    
    return entropy

def rank_entropy_evaluate(prediction, ground_truth, mask, risk_free_return=0.018):
    # prediction: (sample_num, stock_num, date_len)
    # ground_truth: (stock_num, date_len)
    # mask: (stock_num, date_len)
    assert ground_truth.shape == prediction[0].shape, 'shape mis-match'
    stock_num, date_len = prediction[0].shape
    stock_risk_data = np.zeros((date_len, stock_num, 2))
    top_5_stocks = []
    performance = {}
    bt_long5 = 1.0
    sharpe_li5 = []
    stock2rr_uncs = []
    sample_size = prediction.shape[0]

    for day in tqdm(range(date_len)):
        stock_risk_data[day, :, 0] = ground_truth[:, day]
        inf_index = np.argwhere(mask[:, day] < 0.5)
        cur_pre = prediction[:, :, day]
        cur_pre[:, inf_index] = -float('inf')

        # compute average returns of the top-5 stocks
        #real_pre_top5 = heapq.nlargest(5, range(len(cur_pre[0])), cur_pre[0].take)
        rank_pre = np.argsort(cur_pre[0])
        real_pre_top5 = []
        for j in range(1, stock_num + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][day] < 0.5:
                continue
            if len(real_pre_top5) < 5:
                real_pre_top5.append(cur_rank)
        top_5_stocks.append(real_pre_top5)
        real_ret_rat_top5 = 0
        for pre in real_pre_top5:
            real_ret_rat_top5 += ground_truth[pre, day]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)

        # compute the entropy of each stock
        rank_records = np.zeros((stock_num, stock_num))    # {stock: rankings}
        for s in range(1, sample_size):
            tmp_rank = np.argsort(cur_pre[s])
            for j in range(1, stock_num + 1):
                cur_rank = tmp_rank[-1 * j]
                if mask[cur_rank][day] >= 0.5:
                    rank_records[cur_rank][j-1] += 1.
        entropy = ranking_entropy(rank_records, sample_size-1)
        stock_risk_data[day, :, 1] = entropy

    performance['top_5_stocks'] = top_5_stocks
    performance['stock_risk_data'] = stock_risk_data
    performance['btl5'] = bt_long5 - 1.
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = ((np.mean(sharpe_li5) - (risk_free_return/365.0))/np.std(sharpe_li5))*15.87 #To annualize
    return performance

def rank_entropy_1day(prediction, ground_truth, mask, cross=True):
    # prediction: (sample_num, stock_num)
    # ground_truth: (stock_num)
    # mask: (stock_num)
    assert ground_truth.shape == prediction[0].shape, 'shape mis-match'
    stock_num = prediction.shape[1]
    stock_risk_data = np.zeros((stock_num, 2))
    top_5_stock = []
    performance = {}
    sample_size = prediction.shape[0]

    inf_index = np.argwhere(mask < 0.5)
    prediction[:, inf_index] = -float('inf')
    rank_pre = np.argsort(prediction[0])
    for j in range(1, stock_num + 1):
        cur_rank = rank_pre[-1 * j]
        if mask[cur_rank] < 0.5:
            continue
        if len(top_5_stock) < 5:
            top_5_stock.append(cur_rank)
    real_ret_rat_top5 = 0
    for pre in top_5_stock:
        real_ret_rat_top5 += ground_truth[pre]
    real_ret_rat_top5 /= 5
    bt_long5 = real_ret_rat_top5

    # compute the entropy of each stock
    stock_risk_data[:, 0] = ground_truth
    rank_records = np.zeros((stock_num, stock_num))    # {stock: rankings}
    for stock in range(stock_num):
        for s in range(1, sample_size):
            if cross:
                cur_pre = copy.copy(prediction[0])
                cur_pre[stock] = prediction[s, stock]
            else:
                cur_pre = prediction[s]
            tmp_rank_pre = np.argsort(cur_pre)
            for j in range(1, stock_num + 1):
                cur_srank = tmp_rank_pre[-1 * j]
                if cur_srank == stock:
                    rank_records[stock][j-1] += 1.
                    break
            del cur_pre
    entropy = ranking_entropy(rank_records, sample_size-1)
    stock_risk_data[:, 1] = entropy

    performance['top_5_stock'] = top_5_stock
    performance['stock_risk_data'] = stock_risk_data
    performance['btl5'] = bt_long5
    return performance

def rank_prob_evaluate(prediction, ground_truth, mask):
    # prediction: (sample_num, stock_num, date_len)
    # ground_truth: (stock_num, date_len)
    # mask: (stock_num, date_len)
    assert ground_truth.shape == prediction[0].shape, 'shape mis-match'
    performance = {}
    bt_long5 = 1.0
    sharpe_li5 = []
    ndcg_score_top5 = []
    uncertainties = []
    market_stds = []
    pre_top5_freqs = []
    sample_size = prediction.shape[0]

    for day in range(ground_truth.shape[1]):
        inf_index = np.argwhere(mask[:, day] < 0.5)
        cur_gt = ground_truth[:, day]
        cur_std = np.std(np.delete(cur_gt, inf_index))
        market_stds.append(cur_std)
        cur_gt[inf_index] = -float('inf')
        gt_top5 = heapq.nlargest(5, range(len(cur_gt)), cur_gt.take)

        cur_pre = prediction[:, :, day]
        cur_pre[:, inf_index] = -float('inf')
        top5_dict = {}
        for s in range(sample_size):
            tmp_top5 = heapq.nlargest(5, range(len(cur_pre[s])), cur_pre[s].take)
            for idx in tmp_top5: top5_dict[idx] = top5_dict.get(idx, 0) + 1
        pre_top5 = heapq.nlargest(5, top5_dict.keys(), lambda x: top5_dict[x])
        pre_top5_freqs.append(top5_dict)
        
        unc = np.array([top5_dict[idx] / sample_size for idx in pre_top5])
        unc = np.sum(unc) / 5.
        uncertainties.append(unc)
        ndcg_score_top5.append(ndcg_score(np.array(gt_top5).reshape(1,-1), np.array(pre_top5).reshape(1,-1)))
        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][day]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)
    
    performance['market_stds'] = market_stds
    performance['uncertainty_array'] = uncertainties
    performance['ndcg_score_top5_array'] = ndcg_score_top5
    performance['ndcg_score_top5'] = np.mean(np.array(ndcg_score_top5))
    performance['pre_top5_freqs'] = pre_top5_freqs
    performance['btl5'] = bt_long5 - 1
    performance['btl5_array'] = sharpe_li5
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = ((np.mean(sharpe_li5) - (0.005/365.0)) / np.std(sharpe_li5)) * 15.87 #To annualize
    return performance

def rank_prob_evaluate2(prediction, ground_truth, mask):
    # prediction: (sample_num, stock_num, date_len)
    # ground_truth: (stock_num, date_len)
    # mask: (stock_num, date_len)
    assert ground_truth.shape == prediction[0].shape, 'shape mis-match'
    performance = {}
    bt_long5 = 1.0
    sharpe_li5 = []
    ndcg_score_top5 = []
    stock2rr_uncs = []
    sample_size = prediction.shape[0]

    for day in range(ground_truth.shape[1]):
        inf_index = np.argwhere(mask[:, day] < 0.5)
        cur_gt = ground_truth[:, day]
        cur_gt[inf_index] = -float('inf')
        gt_top5 = heapq.nlargest(5, range(len(cur_gt)), cur_gt.take)

        cur_pre = prediction[:, :, day]
        cur_pre[:, inf_index] = -float('inf')
        real_pre_top5 = heapq.nlargest(5, range(len(cur_pre[0])), cur_pre[0].take)
        top5_dict = {}
        for sid in real_pre_top5: top5_dict[sid] = 1
        for s in range(1, sample_size):
            tmp_top5 = heapq.nlargest(5, range(len(cur_pre[s])), cur_pre[s].take)
            for sid in real_pre_top5:
                if sid in tmp_top5:
                    top5_dict[sid] += 1

        ndcg_score_top5.append(ndcg_score(np.array(gt_top5).reshape(1,-1), np.array(real_pre_top5).reshape(1,-1)))
        sid2rr_unc = {}
        real_ret_rat_top5 = 0
        for sid in real_pre_top5:
            sid2rr_unc[sid] = [ground_truth[sid][day], top5_dict[sid] / sample_size]
            real_ret_rat_top5 += ground_truth[sid][day]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)
        stock2rr_uncs.append(sid2rr_unc)

    performance['uncertainty_array'] = stock2rr_uncs
    performance['ndcg_score_top5'] = np.mean(np.array(ndcg_score_top5))
    performance['btl5'] = bt_long5 - 1
    performance['btl5_array'] = sharpe_li5
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = ((np.mean(sharpe_li5) - (0.005/365.0))/np.std(sharpe_li5))*15.87 #To annualize
    return performance