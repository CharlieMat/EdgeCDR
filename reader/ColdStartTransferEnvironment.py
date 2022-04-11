import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from argparse import Namespace

from reader.reader_utils import *
from reader.BaseReader import BaseReader
from reader.NextItemReader import sample_negative, padding_and_cut
from data.preprocess import DOMAINS
from utils import read_line_number, extract_args, init_ranking_report

def get_common_user_list(df, domains):
    CU = {}
    for source_domain in tqdm(domains):
        CU[source_domain] = {}
        subset = df[df[source_domain]>0]
        for target_domain in domains:
            CU[source_domain][target_domain] = subset[subset[target_domain]>0]['UserID'].values
    return CU

# def calculate_batch_ranking_metric(pos_pred, all_pred, pos_mask, k_list, report = {}):
#     '''
#     @input:
#     - pos_pred: (B,R)
#     - all_pred: (B,N)
#     - mask: (B,R)
#     - k_list: e.g. [1,5,10,20,50]
#     - report: {"HR@1": 0, "P@1": 0, ...}
#     '''
# #     if len(report) == 0:
# #         report = init_ranking_report(k_list)
#     pos_pred = pos_pred.view(-1,1) # (B,1)
#     all_pred = all_pred.view(pos_pred.shape[0],-1) # (B,L)
#     B,L = all_pred.shape[0],all_pred.shape[1]
#     max_k = max(k_list)
#     assert max_k <= L
    
#     diff = all_pred - pos_pred
#     diff[diff>0] = 1
#     diff[diff<0] = 0
#     rank = torch.sum(diff, dim = 1) + 1 # (B,)
#     # normalized mean rank
#     report["MR"] = torch.sum(rank) + report["MR"]
#     # mean reciprocal rank
#     report["MRR"] = torch.sum(1.0/rank) + report["MRR"]
#     # auc
#     report["AUC"] = torch.sum((-rank+L) / (L-1)) + report["AUC"]
#     # hit map of each position
#     hitMap = torch.zeros_like(all_pred)
#     positions = torch.arange(1,L+1,device=all_pred.device).view(1,-1)
#     hitMap[positions >= rank.view(-1,1)] = 1 # (B,L)
#     # hit rate
#     hr = torch.sum(hitMap, dim = 0)
#     # precision
#     precision = hitMap / positions # (B,L)
#     # recall
#     recall = hitMap # (B,L)
#     # f1
#     f1 = (precision * recall * 2) / (precision + recall + 1e-6) # (B,L)
#     precision = torch.sum(precision, dim = 0) # (B,)
#     recall = torch.sum(recall, dim = 0) # (B,)
#     f1 = torch.sum(f1, dim = 0) # (B,)
#     # ndcg
#     idcg = 1.
#     dcg = hitMap / torch.log2(rank+1).view(B,1)
#     ndcg = torch.sum(dcg / idcg, dim = 0) # (B,)
#     for k in k_list:
#         report["HR@%d"%k] = hr[k-1].detach() + report["HR@%d"%k] 
#         report["P@%d"%k] = precision[k-1].detach() + report["P@%d"%k] 
#         report["RECALL@%d"%k] = recall[k-1].detach() + report["RECALL@%d"%k] 
#         report["F1@%d"%k] = f1[k-1].detach() + report["F1@%d"%k] 
#         report["NDCG@%d"%k] = ndcg[k-1].detach() + report["NDCG@%d"%k] 
#     return report, B
    
# #############################################################################
# #                              Dataset Class                                #
# #############################################################################

# class UserTransferEvalReader(IterableDataset):
#     def __init__(self, reader, L, n_worker = 1):
#         super().__init__()
#         self.reader = reader
#         self.L = L
        
#         self.n_worker = max(n_worker, 1)
#         self.worker_id = None
        
#     def __iter__(self):
#         for idx in tqdm(range(self.L)):
#             if idx % self.n_worker == self.worker_id:
#                 uid = self.reader.bufferred_users[self.reader.phase][idx]
#                 record = self.reader.get_user_eval_info(uid)
#                 row_id = self.reader.user_row[uid]
#                 if "no_item" in record:
#                     record['UniID'] = row_id + 1
#                     yield record
#                 transfer_info = self.reader.data['train'].iloc[row_id]
#                 for d in self.reader.domains:
#                     record[d] = transfer_info[d]
#                 record['UniID'] = row_id + 1
#                 for d in self.reader.domains:
#                     record['emb_' + d] = self.reader.user_embeddings[d][record[d]]
#                     record['mask_' + d] = 1. if record[d] > 0 else 0.
#                 yield record

class ColdStartTransferEnvironment(BaseReader):
    
    @staticmethod
    def parse_data_args(parser):
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--domain_model_file', type=str, required=True, 
                            help='file path for environment models')
        parser.add_argument('--target', type=str, default='None', 
                            help='specificy a target domain')
        parser.add_argument('--n_neg_val', type = int, default = 100, 
                            help = 'number of negative per user for validation set, set to -1 if sample all items')
        parser.add_argument('--n_neg_test', type = int, default = -1, 
                            help = 'number of negative per user for test set, set to -1 if sample all items')
        return parser
    
    
    def log(self):
        super().log()
        print(f"\ttarget: {self.target_domain}")
        print(f"\tdomain_dim: {self.user_emb_size}")
        print(f"\t#training samples: {len(self.transfer_for_train)}")
        print(f"\t#user: {len(self.transfer_for_train)}/{len(self.users)}")
        print(f"\tn_neg_val : {self.n_neg_val}")
        print(f"\tn_neg_test : {self.n_neg_test}")
        print(f"\tdata_dir : {self.data_dir}")
            
    def __init__(self, args):
        '''
        - phase: one of ["train", "val", "test"]
        
        - domain_model_path: {domain_name: domain_model_path}
        - domains: [domain_name]
        - user_embeddings: {domain_name: embedding_tensor}
        - user_emb_size: {domain_name: scalar}
        - self.data: {'train': [UserID, D1_idx, ... , DN_idx], 
                        'val': [UserID, ItemID, Response, Timestamp], 
                        'test': [UserID, ItemID, Response, Timestamp]}
        - self.data_dir: data_path
        - common_user_list: {source_domain: {target_domain: [UserID]}}
        - user_row: {UserID: row_id in self.data['train']}
        - users: [UserID]
        '''
        self.n_neg_val = args.n_neg_val
        self.n_neg_test = args.n_neg_test
        super(ColdStartTransferEnvironment, self).__init__(args)
        
        '''
        - target_domain
        - target_model
        - data['val'] and data['test']: DataFrame with [UserID, ItemID, Response, Timestamp]
        - transfer_for_eval: {UserID: [domain_id]}
        - transfer_for_train: DataFrame with [UserID, domain1_id, ... , domainN_id] 
        - all_item_candidates: [idx] of target domain items
        - bufferred_positive_sample: {'val': {UserID: [iid]}, 'test': {UserID: [iid]}}
        - bufferred_users: {'val': [UserID], 'test': [UserID]}
        - bufferred_negative_sample: {'val': {UserID: [iid]}}
        '''
        self.set_target(args.target)
        
#         from utils import init_ranking_report
#         b = 1 / torch.log2(torch.arange(2,self.max_k+2).to(self.target_model.device))
#         ap = torch.arange(1, max_k+1).to(torch.float).to('cuda:4').view(1,-1)
#         gt_position = torch.arange(1,n_pos+1).to('cuda:4').view(1,-1)

        
    def _read_data(self, args):
        '''
        - domain_model_path: {domain_name: domain_model_path}
        - domains: [domain_name]
        - user_embeddings: {domain_name: embedding_tensor}
        - user_emb_size: {domain_name: scalar}
        - self.data: {'train': [UserID, D1_idx, ... , DN_idx]}
        - self.data_dir: data_path
        - common_user_list: {source_domain: {target_domain: [UserID]}}
        - user_row: {UserID: row_id in self.data['train']}
        - users: [UserID]
        '''
        self.do_print("Load domain model embeddings")
        # {domain_name: domain_model_log_path}
        self.domain_model_path = eval(open(args.domain_model_file, 'r').readline())
        # [domain_name]
        self.domains = list(self.domain_model_path.keys())
        # {domain_name: embedding_tensor}
        self.user_embeddings = {}
        for d in tqdm(self.domains):
            log_path = self.domain_model_path[d]
            argstr = read_line_number(log_path, 1)
            model_args = eval(argstr) # evaluate Namespace
            checkpoint = torch.load(model_args.model_path + ".checkpoint", 
                                    map_location = 'cpu')['model_state_dict']
            uEmb = checkpoint['uEmb.weight']
            uBias = checkpoint['uBias.weight']
            self.user_embeddings[d] = torch.cat([uEmb, uBias],dim=1)
        # {domain_name: scalar}
        self.user_emb_size = {d: EMB.shape[1] for d,EMB \
                              in self.user_embeddings.items()}
        # data
        # - train: [UserID, D1_idx, D2_idx, ..., DN_idx]
        # - val & test: [UserID, ItemID, Response, Timestamp]
        self.do_print("Load transfer data")
        self.data = {}
        self.data_dir = args.data_file
        df = pd.read_csv(args.data_file + 'id_train.tsv', header = 0, sep = '\t',
                         names = ["UserID"] + self.domains + ["All"])
        df = df[["UserID"] + self.domains]
        self.data['train'] = df
        # {source_domain: {target_domain: [uid]}}
        self.common_user_list = get_common_user_list(df, self.domains)
        # {uid: row_id}
        self.user_row = {uid: idx for idx,uid in enumerate(df['UserID'].values)}
        # [uid]
        self.users = list(self.user_row.keys())
        
        
    def set_target(self, target_domain):
        '''
        - target_domain
        - target_model
        - data['val'] and data['test']: DataFrame with [UserID, ItemID, Response, Timestamp]
        - transfer_for_eval: {UserID: [domain_id]}
        - transfer_for_train: DataFrame with [UserID, domain1_id, ... , domainN_id] 
        - all_item_candidates: [idx] of target domain items
        - bufferred_positive_sample: {'val': {UserID: [iid]}, 'test': {UserID: [iid]}}
        - bufferred_users: {'val': [UserID], 'test': [UserID]}
        - bufferred_negative_sample: {'val': {UserID: [iid]}}
        '''
        self.do_print("Load target domain environment")
        self.target_domain = target_domain if target_domain != 'None' else self.domains[0]
        # target model
        model_args = extract_args(self.domain_model_path[target_domain])
        self.target_model = torch.load(model_args.model_path, 
                                       map_location='cpu')
        # cold-start evaluation data
        self.data['val'] = pd.read_csv(self.data_dir+target_domain+"_val.tsv", index_col = 0)
        self.data['test'] = pd.read_csv(self.data_dir+target_domain+"_test.tsv", index_col = 0)
        self.transfer_for_eval = {}
        for phase in ['val','test']:
            for uid in self.data[phase]['UserID'].unique():
                self.transfer_for_eval[uid] = []
        for uid, *domain_uids in self.data['train'].values:
            if uid in self.transfer_for_eval:
                self.transfer_for_eval[uid] = domain_uids
        self.do_print("#User in eval: ", len(self.transfer_for_eval))
        # filter users that interacted with the target domain
        df = self.data['train']
        train_subset = df[df[target_domain]>0]
        condition_str = ' | '.join([f"(train_subset['{d}']>0)" \
                                    for d in self.domains if d != target_domain])
        self.do_print(f"Target domain: {target_domain}")
        self.do_print("Filtering condition: " + condition_str)
        train_subset = train_subset[eval(condition_str)]
        self.transfer_for_train = list(train_subset.index)
        # buffer positive and negative records in target domain
        self._buffer_samples()
        # position indices of user history
        self.gt_position = {phase: torch.arange(1,self.max_hist_len[phase]+1).view(1,-1) \
                            for phase in ['val', 'test']} # [1,2,3,4,...,|H|]
        
    def set_target_device(self, device):
        self.do_print(f"move target model at {self.target_model.device} to {device}:")
        self.do_print(self.target_model)
        self.target_model.to(device)
        self.target_model.device = device
        for k,v in self.gt_position.items():
            self.gt_position[k] = v.to(device)
        
    def _buffer_samples(self):
        '''
        - bufferred_positive_sample: {'val': {UserID: [ItemID]}, 'test': {UserID: [ItemID]}}
        - buffered_users: {'val': [UserID], 'test': [UserID]}
        - bufferred_negative_sample: {'val': {UserID: [ItemID]}, 'test': {UserID: [ItemID]}}
        - max_hist_len: {'val': scalar, 'test': scalar}
        - all_item_candidates: set(ItemID)
        '''
        self.all_item_candidates = self.target_model.reader.all_item_candidates
        self.do_print("Buffer validation and test samples")
        self.bufferred_positive_sample = {'val':{}, 'test':{}}
        self.bufferred_users = {}
        tgt_reader = self.target_model.reader
        for key in ['val','test']:
            df = self.data[key]
            user_hist = {}
            for idx in tqdm(range(len(df))):
                row = df.iloc[idx]
                uid, iid, resp, t = row['UserID'], row['ItemID'], row['Response'], row['Timestamp']
                if uid not in user_hist:
                    user_hist[uid] = []
                user_hist[uid].append((tgt_reader.get_item_feature(iid, "ItemID"),resp))
            self.bufferred_positive_sample[key] = user_hist
            self.bufferred_users[key] = list(user_hist.keys())

        self.do_print("Buffer negative sample")
        self.bufferred_negative_sample = {'val': {}, 'test': {}}
        max_hist_len = {'val': 0, 'test': 0}
        for uid, user_hist in tqdm(self.bufferred_positive_sample['val'].items()):
            pos_items = [iid for iid, resp in user_hist]
            self.bufferred_negative_sample['val'][uid] = np.array(sample_negative(
                                            pos_items, self.all_item_candidates, 
                                            n_neg = self.n_neg_val, replace = False))
            max_hist_len['val'] = max(max_hist_len['val'], len(pos_items))
        for uid, user_hist in tqdm(self.bufferred_positive_sample['test'].items()):
            pos_items = [iid for iid, resp in user_hist]
            if self.n_neg_test > 0:
                self.bufferred_negative_sample['test'][uid] = sample_negative(
                                            pos_items, self.all_item_candidates, 
                                            n_neg = self.n_neg_test, replace = False)
            max_hist_len['test'] = max(max_hist_len['test'], len(pos_items))
        self.max_hist_len = max_hist_len
        self.do_print(f"max user history length: {self.max_hist_len}")

    def get_statistics(self):
        stats = {'n_user': len(self.users), 'n_domains': len(self.domains), 
                 'emb_size': [self.user_emb_size[d] for d in self.domains]}
        return stats
        
#     def get_train_dataset(self):
#         return self
    
#     def get_eval_dataset(self):
#         return UserTransferEvalReader(self, len(self.bufferred_users[self.phase]), 
#                                       n_worker = self.n_worker)
    
    def __len__(self):
        if self.phase == 'train':
            return len(self.transfer_for_train)
        else:
            return len(self.bufferred_users[self.phase])
    
    def __getitem__(self, idx):
        '''
        {
            'UserID': ...,
            domain1: uid,
            ...
            domainD: uid,
            emb_domain1: (dim, ),
            ...
            emb_domainD: (dim, ),
            mask_domain1: 1/0,
            ...
            mask_domainD: 1/0
            
            # for val and test
            item_resp: (R,)
            item_ItemID: (R,)
            item_mask: (R,)
            negi_ItemID: (N,)
        }
        '''
        if self.phase == 'train':
            # domain record row_id in data['train']
            row_id = self.transfer_for_train[idx]
            # [UserID, D1_idx, ... , DN_idx]
            transfer_info = self.data['train'].iloc[row_id].to_dict()
            record = {}
        else:
            uid = self.bufferred_users[self.phase][idx]
            record, transfer_info = self.get_user_eval_info(uid)
        for d in self.domains:
            record[d] = transfer_info[d]
            record['emb_' + d] = self.user_embeddings[d][record[d]]
            record['mask_' + d] = 1. if record[d] > 0 else 0.
            
        return record
    
    def get_user_eval_info(self, uid):
        # user's domain id for transfer
        domain_uid = self.transfer_for_eval[uid]
        transfer_info = {d: domain_uid[i] for i,d in enumerate(self.domains)}
        # positive and negative samples
        items = [iid for iid,r in self.bufferred_positive_sample[self.phase][uid]]
        resp = [r for iid,r in self.bufferred_positive_sample[self.phase][uid]]
        if self.phase == 'test' and self.n_neg_test == -1:
            neg_items = sample_negative(items, self.all_item_candidates, -1)
        else:
            neg_items = self.bufferred_negative_sample[self.phase][uid]
        L = len(items)
        items = padding_and_cut(items, L, self.max_hist_len[self.phase], default = 0)
        resp = padding_and_cut(resp, L, self.max_hist_len[self.phase], default = 0)
        mask = [0] * (self.max_hist_len[self.phase] - L) + [1] * L
        # construct data
        user_data = {"item_resp": np.array(resp),
                     "item_ItemID": np.array(items),
                     "item_mask": np.array(mask),
                     "negi_ItemID": np.array(neg_items)}
        return user_data, transfer_info
    
    
    def get_target_domain_predictions(self, feed_dict, transfer_dict):
        '''
        @input:
        - transfer_model
        - feed_dict: {
            ${domain_name}: (1,), # [uid]
            emb_${domain_name}: (1,dim),
            mask_${domain_name}: (1,), # [0,1]-mask for existence in domain
            item_ItemID: (1,P)
            negi_ItemID: (1,N)
            resp: (1,P)
        }
        '''
        assert self.phase == 'val' or self.phase == 'test'
        u_emb = transfer_dict['preds'][:,:-1]
        u_bias = transfer_dict['preds'][:,-1]
        pos_out = self.target_model.forward_with_emb({
                                    'user_emb': u_emb, 'user_bias': u_bias,
                                    'ItemID': feed_dict['item_ItemID']})
        neg_out = self.target_model.forward_with_emb({
                                    'user_emb': u_emb, 'user_bias': u_bias,
                                    'ItemID': feed_dict['negi_ItemID']})
        pos_preds = torch.sigmoid(pos_out['preds'])
        neg_preds = torch.sigmoid(neg_out['preds'])
        
        return {'pos_preds': pos_preds, 'neg_preds': neg_preds, 'pos_mask': feed_dict['item_mask'], 
                'gtp': self.gt_position[self.phase]}
    