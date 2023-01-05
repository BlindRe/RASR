import copy
import numpy as np
from tqdm import tqdm
from scipy import sparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import utils
import torch.optim as optim
import torch.distributions as ds
from torch.nn.utils import clip_grad_norm_

from utils.load_data import load_stock_data
from utils.runtime_logger import get_logger
from utils.args import Args
from utils.metric import evaluate
from utils.optimize import Adjust_LR, init_model
from models.svat import SVAT, AdvSampler, AdvGenerator
from experiments.exp_basic import Exp_Basic

def tensor_clean(data, r_num):
    data = torch.where(torch.isnan(data), torch.full_like(data, r_num), data)
    data = torch.where(torch.isinf(data), torch.full_like(data, r_num), data)

    return data

class Exp_SVAT(Exp_Basic):
    def __init__(self, param_dict, gpus=0):
        """
        Experiment class for the Adversial HGAT model

        :params:
            - param_dict: dict      hyperparameters defined by user
            - gpus: int or list     GPU number             
        """
        super(Exp_SVAT, self).__init__(param_dict, gpus)
        args = Args(self.params[0])
        self.stock_fea, self.rel_mat, self.masks, self.gt_rr, self.base_data = load_stock_data(
            args.data_path,
            args.stock_rel_file, 
            args.stock_id_file,
            args.market_name
        )
        self.total_stock_num = len(self.masks)
        self.trade_dates = self.masks.shape[1]

    def _build_model(self, args):
        """
        Build the model for experiments

        :params:
            - args: utils.args.Args   Args object storing hyperparameters
        """
        model = SVAT(
            args.num_stocks,
            args.fea_dim,
            args.hid_size,
            args.drop_rate,
            args.history_len
        )
        z_prior_sampler = AdvSampler(
            args.hid_size,
            args.z_dim,
            args.adv_hid_size
        )
        z_post_sampler = AdvSampler(
            args.hid_size * 2,
            args.z_dim,
            args.adv_hid_size
        )
        adv_generator = AdvGenerator(
            args.hid_size + args.z_dim,
            args.hid_size
        )

        model, self.device = init_model(model, self.gpus)
        z_prior_sampler, _ = init_model(z_prior_sampler, self.gpus)
        z_post_sampler, _ = init_model(z_post_sampler, self.gpus)
        adv_generator, _ = init_model(adv_generator, self.gpus)

        return model, z_prior_sampler, z_post_sampler, adv_generator

    def _get_logger(self, args):
        logger = get_logger(args.checkpoint_dir, 'SVAT', 'SVAT-'+args.market_name)
        return logger

    def _get_optimizer(self, model, lr, op_type='adam'):
        if op_type == 'adam':
            model_optim = optim.Adam(model.parameters(), lr=lr)
        elif op_type == 'sgd':
            model_optim = optim.SGD(model.parameters(), lr=lr)
        elif op_type == 'momentum':
            model_optim = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif op_type == 'rmsprop':
            model_optim = optim.RMSprop(model.parameters(), lr=lr)
        
        return model_optim

    def _get_rank_loss(self, args, pred, ground_truth, base_data, stock_num, is_adv=False):
        if args.market_name == 'CASE':
            reg_loss = nn.MSELoss(reduction='none')(pred, base_data)
        else:
            pred_rr = torch.div((pred - base_data), base_data)
            reg_loss = nn.MSELoss(reduction='none')(pred_rr, ground_truth)

        all_ones = torch.ones(stock_num,1).to(self.device)
        pre_pw_dif = (
            torch.matmul(pred, torch.transpose(all_ones, 0, 1)) - 
            torch.matmul(all_ones, torch.transpose(pred, 0, 1))
        )
        gt_pw_dif = (
            torch.matmul(ground_truth, torch.transpose(all_ones, 0,1)) -
            torch.matmul(all_ones, torch.transpose(ground_truth,0,1))
        )
        mask_pw = torch.sign(gt_pw_dif) if args.rank_sign else gt_pw_dif
        if not is_adv:
            reg_loss = torch.mean(reg_loss)
            rank_loss = torch.mean(F.relu(-1. * pre_pw_dif * mask_pw))
        else:
            reg_loss = torch.mean(reg_loss * ground_truth)
            rank_loss = torch.mean(F.relu(-1. * pre_pw_dif * mask_pw) * ground_truth)
        total_loss = args.reg_alpha * reg_loss + rank_loss

        del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
        return total_loss, pred

    def _get_kl_loss(self, prior_mu, prior_std, post_mu, post_std):
        prior_pdf = ds.normal.Normal(prior_mu, prior_std)
        post_pdf = ds.normal.Normal(post_mu, post_std)
        kl_loss = ds.kl.kl_divergence(post_pdf, prior_pdf)

        return kl_loss.mean()

    def _get_batch(self, args, offset, select_all=False):
        mask_batch = self.masks[:, offset: offset + args.history_len + 1]
        mask_batch = np.min(mask_batch, axis=1)
        if select_all:
            stock_idx = np.arange(0, self.total_stock_num, dtype=np.int64)
        else:
            stock_idx = np.argwhere(mask_batch > 1e-8)
            stock_idx = stock_idx.reshape(len(stock_idx))

        if args.market_name == 'CASE':
            base_data = np.expand_dims(self.base_data[stock_idx, offset+args.history_len], axis=1)
        else:
            base_data = np.expand_dims(self.base_data[stock_idx, offset+args.history_len-1], axis=1)
        return self.stock_fea[stock_idx, offset:offset+args.history_len, :-1], \
               self.stock_fea[stock_idx, offset:offset+args.history_len+1, -1], \
               self.rel_mat[stock_idx, :], \
               np.expand_dims(self.gt_rr[stock_idx, offset+args.history_len], axis=1), \
               base_data, \
               stock_idx, \
               np.expand_dims(mask_batch, axis=1)

    def valid(self, args, start_idx, end_idx):
        with torch.no_grad():
            self.model.eval()
            self.z_prior_sampler.eval()
            self.z_post_sampler.eval()
            self.adv_generator.eval()
            cur_valid_pred = np.zeros([self.total_stock_num, end_idx-start_idx], dtype=float)
            cur_valid_gt = np.zeros([self.total_stock_num, end_idx-start_idx], dtype=float)
            cur_valid_mask = np.zeros([self.total_stock_num, end_idx-start_idx], dtype=float)
            val_rank_loss = 0.0
            rel_sparse = sparse.coo_matrix(self.rel_mat)
            incidence_edge = utils.from_scipy_sparse_matrix(rel_sparse)
            hyp_input = incidence_edge[0].to(self.device)
            for cur_offset in range(start_idx-args.history_len, end_idx-args.history_len):
                fea_batch, TRs, _, gt_batch, base_data, _, mask_batch = self._get_batch(args, cur_offset, True)
                att_fea, hg_fea, output = self.model(torch.FloatTensor(fea_batch).to(self.device), hyp_input)
                cur_rank_loss, curr_rank_score = self._get_rank_loss(
                    args,
                    output.reshape((self.total_stock_num,1)), 
                    torch.FloatTensor(gt_batch).to(self.device),
                    torch.FloatTensor(base_data).to(self.device),
                    self.total_stock_num,
                    False
                )
                val_rank_loss += cur_rank_loss.detach().cpu().item()
                cur_valid_gt[:, cur_offset-(start_idx-args.history_len)] = copy.copy(gt_batch[:, 0])
                cur_valid_mask[:, cur_offset-(start_idx-args.history_len)] = copy.copy(mask_batch[:, 0])
                curr_rank_score = curr_rank_score.detach().cpu().numpy().reshape((self.total_stock_num,1))
                cur_valid_pred[:, cur_offset-(start_idx-args.history_len)] = copy.copy(curr_rank_score[:, 0])
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)

            return val_rank_loss / (end_idx - start_idx), cur_valid_perf

    def train(self):
        best_perf = {'sharpe5': -np.Inf}
        for k, param in enumerate(self.params):
            args = Args(param)
            tr_logger = self._get_logger(args)
            tr_logger.info('============== start training ' + str(k) + '-th parameters ==============')
            tr_logger.info(args)

            self.model, self.z_prior_sampler, self.z_post_sampler, self.adv_generator = self._build_model(args)
            model_optim = self._get_optimizer(self.model, args.learning_rate, 'adam')
            z_prior_optim = self._get_optimizer(self.z_prior_sampler, args.adv_lr, 'adam')
            z_post_optim = self._get_optimizer(self.z_post_sampler, args.adv_lr, 'adam')
            adv_gen_optim = self._get_optimizer(self.adv_generator, args.adv_lr, 'adam')
            lr_adjuster = Adjust_LR(patience=args.patience)
            batch_offsets = np.arange(start=0, stop=args.valid_index, dtype=int)
            val_start_idx = args.valid_index
            val_end_idx = args.text_index if args.text_index is not None else self.trade_dates
            for i in range(args.epochs):
                self.model.train()
                self.z_prior_sampler.train()
                self.z_post_sampler.train()
                self.adv_generator.train()
                np.random.shuffle(batch_offsets)
                tra_rank_loss = 0.0
                for j in tqdm(range(args.valid_index - args.history_len)):
                    fea_batch, _, rel_mat, gt_batch, base_data, stock_idx, _ = self._get_batch(args, batch_offsets[j])
                    rel_sum = np.sum(rel_mat, axis=0)
                    rel_id = np.argwhere(rel_sum > 1.)[:,0]
                    rel_mat = rel_mat[:, rel_id]
                    rel_sparse = sparse.coo_matrix(rel_mat)
                    incidence_edge = utils.from_scipy_sparse_matrix(rel_sparse)
                    hyp_input = incidence_edge[0].to(self.device)
                    batch_size = len(fea_batch)

                    self.model.eval()
                    z_prior_optim.zero_grad()
                    z_post_optim.zero_grad()
                    adv_gen_optim.zero_grad()
                    att_fea, hg_fea, output = self.model(torch.FloatTensor(fea_batch).to(self.device), hyp_input, stock_idx)
                    fea_con = att_fea if args.adv == 'Attention' else hg_fea
                    loss_for_adv, _ = self._get_rank_loss(
                        args,
                        output.reshape((batch_size,1)), 
                        torch.FloatTensor(gt_batch).to(self.device),
                        torch.FloatTensor(base_data).to(self.device),
                        batch_size,
                        False
                    )
                    z_prior_mu, z_prior_std = self.z_prior_sampler(fea_con.detach())
                    z_prior_mu = tensor_clean(z_prior_mu, 0.)
                    z_prior_std = tensor_clean(z_prior_std, 1.)
                    grad = torch.autograd.grad(loss_for_adv, [fea_con])[0].detach()
                    norm_grad = args.adv_eps * nn.functional.normalize(grad, p=2, dim=1)
                    z_post_mu, z_post_std = self.z_post_sampler(torch.cat([fea_con.detach(), norm_grad], 1))
                    z_post_mu = tensor_clean(z_post_mu, 0.)
                    z_post_std = tensor_clean(z_post_std, 1.)
                    z_eps = torch.randn_like(z_post_std)
                    z_post = z_post_mu + z_post_std * z_eps
                    delta_g = self.adv_generator(torch.cat([fea_con.detach(), z_post], 1))
                    delta = args.adv_eps * nn.functional.normalize(delta_g, p=2, dim=1)

                    self.model.train()
                    model_optim.zero_grad()
                    att_fea, hg_fea, output = self.model(torch.FloatTensor(fea_batch).to(self.device), hyp_input, stock_idx)
                    fea_con = att_fea if args.adv == 'Attention' else hg_fea
                    e = hyp_input if args.adv == 'Attention' else None
                    x_adv = fea_con + delta
                    adv_output = self.model.adv(x_adv, e)

                    origin_loss, _ = self._get_rank_loss(
                        args,
                        output.reshape((batch_size,1)), 
                        torch.FloatTensor(gt_batch).to(self.device),
                        torch.FloatTensor(base_data).to(self.device),
                        batch_size,
                        False
                    )
                    tra_rank_loss += origin_loss.item()
                    adv_loss, _ = self._get_rank_loss(
                        args,
                        adv_output.reshape((batch_size,1)), 
                        torch.FloatTensor(gt_batch).to(self.device),
                        torch.FloatTensor(base_data).to(self.device),
                        batch_size,
                        True
                    )
                    kl_loss = self._get_kl_loss(z_prior_mu, z_prior_std, z_post_mu, z_post_std)
                    total_loss = origin_loss + adv_loss + args.kl_lambda * kl_loss

                    total_loss.backward()
                    model_optim.step()
                    z_prior_optim.step()
                    z_post_optim.step()
                    adv_gen_optim.step()
                tra_rank_loss /= (args.valid_index - args.history_len)
                tr_logger.info('Train Rank Loss: {}, Total Loss: {}'.format(tra_rank_loss, total_loss))
                lr_adjuster(model_optim, tra_rank_loss, i+1, args, tr_logger)

                cur_valid_loss, cur_valid_perf = self.valid(args, val_start_idx, val_end_idx)
                if cur_valid_perf['sharpe5'] > best_perf['sharpe5']:
                    best_perf = cur_valid_perf
                    self.best_args = args
                    model_path = os.path.join(args.checkpoint_dir, 'SVAT-'+args.market_name+".pth")
                    torch.save(self.model.state_dict(), model_path)
                    decoder_path = os.path.join(args.checkpoint_dir, 'SVAT_z_sampler-'+args.market_name+".pth")
                    torch.save(self.z_prior_sampler.state_dict(), decoder_path)
                    generator_path = os.path.join(args.checkpoint_dir, 'SVAT_adv_generator-'+args.market_name+".pth")
                    torch.save(self.adv_generator.state_dict(), generator_path)

                tr_logger.info('Valid Rank Loss: {}'.format(cur_valid_loss))
                tr_logger.info('====> Valid preformance: ndcg_score_top5: {}, btl5: {}, sharpe5: {}, mse: {}'.format(
                    cur_valid_perf['ndcg_score_top5'], cur_valid_perf['btl5'], cur_valid_perf['sharpe5'], cur_valid_perf['mse']
                ))
        tr_logger.info('============== The Best parameters ==============')
        tr_logger.info(self.best_args)
        tr_logger.info('====> Best valid preformance:')
        tr_logger.info('{}'.format(best_perf))
