import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pickle
from multiprocessing import Pool, Process
# import threading
# from concurrent.futures import ThreadPoolExecutor
# from tqdm_multi_thread import TqdmMultiThreadFactory

import utils
from task.GeneralTask import GeneralTask
from task.TopK import init_ranking_report, TopK
from reader.BaseReader import worker_init_func
    
def calculate_batch_ranking_metric(eval_dict, report = {}):
    '''
    @input:
    - pos_pred: (B,R)
    - all_pred: (B,N)
    - mask: (B,R)
    - k_list: e.g. [1,5,10,20,50]
    - report: {"HR@1": 0, "P@1": 0, ...}
    '''
        
    pos_pred, neg_pred, pos_mask = eval_dict['pos_preds'], eval_dict['neg_preds'], eval_dict['pos_mask']
    k_list = eval_dict['k_list']
    b, ap, gt_position = eval_dict['b'], eval_dict['ap'], eval_dict['gtp']
    max_k = max(k_list)
    B,R = pos_pred.shape # (B,R)
    N = neg_pred.shape[1]

    if len(report) == 0:
        report = init_ranking_report(k_list)
        
    pos_pred = pos_pred * pos_mask
    all_pred = torch.cat((pos_pred, neg_pred), dim = 1).view(B,-1) # (B,L)
    pos_length = torch.sum(pos_mask, dim = 1)
#     print(pos_pred.shape, neg_pred.shape)
#     print(pos_length.shape)

    rank = torch.sum(pos_pred.view(B,R,1) < all_pred.view(B,1,R+N), dim = 2) + 1
    rank = rank * pos_mask
#     print(rank.shape, pos_mask.shape)
    values, indices = torch.topk(all_pred, k = max_k, dim = 1)
    hit_map = (indices < R).to(torch.float)
    tp = torch.zeros_like(hit_map) # true positive
    tp[:,0] = hit_map[:,0]
    dcg = torch.zeros_like(hit_map) # DCG
    dcg[:,0] = hit_map[:,0]
    idcg = torch.zeros_like(hit_map)
    flip_mask = torch.flip(pos_mask, dims = [1])
#     print(hit_map.shape, flip_mask.shape)
    K = min(max_k, R)
    idcg[:,:K] = flip_mask[:,:K]
    idcg = idcg * b.view(1,-1)
    for i in range(1,max_k):
        tp[:,i] = tp[:,i-1] + hit_map[:,i]
        dcg[:,i] = dcg[:,i-1] + hit_map[:,i] * b[i]
        idcg[:,i] = idcg[:,i-1] + idcg[:,i]
    hr = tp.clone()
    hr[hr>0] = 1
    precision = (tp / ap)
    recall = (tp / pos_length.view(-1,1))
    f1 = (2*tp / (ap + pos_length.view(-1,1))) # 2TP / ((TP+FP) + (TP+FN))
    ndcg = (dcg / idcg)

    # mean rank
    report['MR'] += torch.sum(torch.sum(rank, dim = 1) / pos_length)
    # mean reciprocal rank
    mrr = torch.sum(pos_mask / (rank + 1e-6), dim = 1)
    report['MRR'] += torch.sum(mrr / pos_length)
    # hit rate
    hr = torch.sum(hr, dim = 0)
    # precision
    precision = torch.sum(precision, dim = 0)
    # recall
    recall = torch.sum(recall, dim = 0)
    # f1
    f1 = torch.sum(f1, dim = 0)
    # ndcg
    ndcg = torch.sum(ndcg, dim = 0)
    # auc
    rank[rank == 0] = R+N+1
    sorted_rank, _ = torch.sort(rank, dim = 1)
    level_width = sorted_rank - gt_position
    level_width = level_width * flip_mask
    auc = torch.sum(level_width, dim = 1) / pos_length
    auc = auc / N
    report['AUC'] += torch.sum(1 - auc)

#     input()
    for k in k_list:
        report[f'HR@{k}'] += hr[k-1]
        report[f'P@{k}'] += precision[k-1]
        report[f'RECALL@{k}'] += recall[k-1]
        report[f'F1@{k}'] += f1[k-1]
        report[f'NDCG@{k}'] += ndcg[k-1]
    return report, B

class ColdStartTopK(GeneralTask):
    
    @staticmethod
    def parse_task_args(parser):
        parser = GeneralTask.parse_task_args(parser)
        parser.add_argument('--at_k', type=int, nargs='+', default=[1,5,10,20,50], 
                            help='specificy a list of k for top-k performance')
        return parser

    def __init__(self, args, reader):
        self.at_k_list = args.at_k
        super().__init__(args, reader)
        max_k = max(self.at_k_list)
        # b in IDCG, [1/log2(2), 1/log2(3), 1/log2(4), ...]
        self.b = 1 / torch.log2(torch.arange(2,max_k+2))
        # [1,2,3,4,...,max_k]
        self.ap = torch.arange(1, max_k+1).to(torch.float).view(1,-1)
                                
    def set_device(self, device):
        self.b = self.b.to(device)
        self.ap = self.ap.to(device)
    
    def log(self):
        super().log()
        print(f"\tat_k: {self.at_k_list}")
    
    def do_epoch(self, model, epoch_id):
        return super().do_epoch(model, epoch_id)

    def do_eval(self, model):
        """
        Evaluate the results for an eval dataset.
        @input:
        - model: GeneralRecModel or its extension
        
        @output:
        - resultDict: {metric_name: metric_value}
        """

        print("Evaluating...")
        print("Sample p = " + str(self.eval_sample_p))
        model.eval()
        if model.loss_type == "regression": # rating prediction evaluation
#             report = self.evaluate_regression(model)
            raise NotImplemented
        else: # ranking evaluation
            report = self.evaluate_userwise_ranking(model)
        print("Result dict:")
        print(str(report))
        return report

    def evaluate_userwise_ranking(self, model):
        '''
        Calculate ranking metrics

        @input:
        - model: transfer model
        
        @output:
        - resultDict:
        {
            "mr": mean rank
            "mrr": mean reciprocal rank
            "auc": area under the curve
            "hr": [1 if a hit @k] of size (L+1), take the mean over all records
            "p": precision
            "recall": recall
            "f1": f1-score
            "ndcg": [DCG score] since IDCG=1 for each record. DCG = 1/log(rank) after rank, =0 before rank
            "metric": auc
        }
        '''
        eval_data = model.reader.get_eval_dataset()
        eval_loader = DataLoader(model.reader, batch_size = self.eval_batch_size, 
                                 shuffle = False, pin_memory = False, 
                                 num_workers = self.n_worker)
        report = init_ranking_report(self.at_k_list)
        n_user_tested = 0
        pbar = tqdm(total = len(model.reader))
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                # sample user with record in eval data
                if np.random.random() <= self.eval_sample_p and \
                        "no_item" not in batch_data:
                    # predict
                    user_report, B = self.get_transfer_performance(model, batch_data)
                    # metrics
                    for k,v in user_report.items():
                        report[k] += v
                    n_user_tested += B
                pbar.update(self.eval_batch_size)
        pbar.close()
        for key, value in report.items():
            report[key] = (report[key] / n_user_tested).cpu().numpy() + 0.
        return report
    
    def get_transfer_performance(self, model, batch_data):
        feed_dict = model.wrap_batch(batch_data)
        transfer_out_dict = model.forward(feed_dict)
        target_domain_out = model.reader.get_target_domain_predictions(feed_dict, transfer_out_dict)
        target_domain_out['k_list'] = self.at_k_list
        target_domain_out['b'] = self.b # b in IDCG, [1/log2(2), 1/log2(3), 1/log2(4), ...]
        target_domain_out['ap'] = self.ap # [1,2,3,4,...,max_k]
        return calculate_batch_ranking_metric(target_domain_out)
    

    