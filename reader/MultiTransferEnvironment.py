
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

from reader.BaseReader import BaseReader
from reader.NextItemReader import sample_negative, padding_and_cut
from data.preprocess import DOMAINS
from utils import read_line_number, extract_args, init_ranking_report
from reader.ColdStartTransferEnvironment import ColdStartTransferEnvironment, get_common_user_list
    
    
#############################################################################
#                              Dataset Class                                #
#############################################################################

class MultiTransferEnvironment(ColdStartTransferEnvironment):
    
    @staticmethod
    def parse_data_args(parser):
        '''
        - data_file
        - domain_model_file
        - target
        - n_neg_val
        - n_neg_test
        '''
        parser = ColdStartTransferEnvironment.parse_data_args(parser)
        return parser
            
    def __init__(self, args):
        # implicitly call self._read_data(), self.set_target(), and self._buffer_sample()
        super(MultiTransferEnvironment, self).__init__(args)
       
    def _read_data(self, args):
        super()._read_data(args)
        self.transfer_for_train = list(self.data['train'].index)
        
    def __getitem__(self, idx):
        if self.phase == 'train':
            # domain record row_id in data['train']
            row_id = self.transfer_for_train[idx]
            # [UserID, D1_idx, ... , DN_idx]
            transfer_info = self.data['train'].iloc[row_id].to_dict()
            record = {'UserID': row_id}
        else:
            uid = self.bufferred_users[self.phase][idx]
            record, transfer_info = self.get_user_eval_info(uid)
            record['UserID'] = self.user_row[uid]
        for d in self.domains:
            record[d] = transfer_info[d]
            record['emb_' + d] = self.user_embeddings[d][record[d]]
            record['mask_' + d] = 1. if record[d] > 0 else 0.
            
        return record
        
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
        self.do_print(f"Load target domain ({target_domain}) environment")
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
        # buffer positive and negative records in target domain
        self._buffer_samples()
        # position indices of user history
        self.gt_position = {phase: torch.arange(1,self.max_hist_len[phase]+1).view(1,-1) \
                            for phase in ['val', 'test']} # [1,2,3,4,...,|H|]
        
        
    def get_target_domain_predictions(self, feed_dict, transfer_dict):
        return super().get_target_domain_predictions(feed_dict, \
                        {'preds': transfer_dict['preds']['out_emb'][self.target_domain]})
        