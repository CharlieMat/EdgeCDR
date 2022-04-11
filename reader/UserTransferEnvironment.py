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

from reader.reader_utils import *
from reader.BaseReader import BaseReader
from reader.RecDataReader import sample_negative
from utils import extract_args, calculate_ranking_metric

def get_user_vocab(vocab_path):
    item_vocab = pd.read_table(vocab_path, index_col = 1)
    value_idx = item_vocab[item_vocab['field_name'] == "UserID"][['idx']]
    value_idx = value_idx[~value_idx.index.duplicated(keep='first')].to_dict(orient = 'index')
    vocab = {str(k): vMap['idx'] for k,vMap in value_idx.items()}
    return vocab

def get_cross_domain_user(domains, data_dir):
    user_domain_ids = {}
    cross_domain_users = {}
    for domain_id, source_domain in tqdm(enumerate(domains)):
        # user set
        source_vocab = get_user_vocab(data_dir + "meta_data/" + \
                                      source_domain + "_user_fields.vocab")
        # domain-specific user id
        for uid,idx in source_vocab.items():
            if uid not in user_domain_ids:
                user_domain_ids[uid] = [0]*len(domains)
            user_domain_ids[uid][domain_id] = idx
        # cross-domain common user lists
        for target_domain in domains:
            if target_domain != source_domain:
                target_vocab = get_user_vocab(data_dir + "meta_data/" + \
                                      target_domain + "_user_fields.vocab")
                common_users = [uid for uid, idx in target_vocab.items() \
                                if uid in source_vocab]
                cross_domain_users[f"{source_domain}@{target_domain}"] = common_users
    all_user = list(user_domain_ids.keys())
    for idx,uid in enumerate(all_user):
        user_domain_ids[uid].append(idx+1)
    return cross_domain_users, user_domain_ids

def get_common_user_list(df, domains):
    CU = {}
    for source_domain in tqdm(domains):
        CU[source_domain] = {}
        subset = df[df[source_domain]>0]
        for target_domain in domains:
            CU[source_domain][target_domain] = subset[subset[target_domain]>0]['UserID'].values
    return CU
    
#############################################################################
#                              Dataset Class                                #
#############################################################################

class UserTransferEvalReader(IterableDataset):
    def __init__(self, reader, L, n_worker = 1):
        super().__init__()
        self.reader = reader
        self.L = L
        
        self.n_worker = max(n_worker, 1)
        self.worker_id = None
        
    def __iter__(self):
        for idx in tqdm(range(self.L)):
            if idx % self.n_worker == self.worker_id:
                uid = self.reader.bufferred_users[self.reader.phase][idx]
                record = self.reader.get_user_eval_info(uid)
                row_id = self.reader.user_row[uid]
                if "no_item" in record:
                    record['UniID'] = row_id + 1
                    yield record
                transfer_info = self.reader.data['train'].iloc[row_id]
                for d in self.reader.domains:
                    record[d] = transfer_info[d]
                record['UniID'] = row_id + 1
                for d in self.reader.domains:
                    record['emb_' + d] = self.reader.user_embeddings[d][record[d]]
                    record['mask_' + d] = 1. if record[d] > 0 else 0.
                yield record

class UserTransferEnvironment(BaseReader):
    
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
        print(f"\t#training samples: {len(self.training_samples)}")
        print(f"\t#user: {len(self.training_samples)}/{len(self.users)}")
        print(f"\tn_neg_val : {self.n_neg_val}")
        print(f"\tn_neg_test : {self.n_neg_test}")
        print(f"\tdata_dir : {self.data_dir}")
            
    def __init__(self, args):
        '''
        - phase: one of ["train", "val", "test"]
        - n_worker
        
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
        
        - target_domain: domain_name
        - target_model: domain-specific model object
        - training_samples: [row_id], users with target_domain and some source_domain
        
        - all_item_candidates: [idx] of target domain items
        - bufferred_positive_sample: {'val': {UserID: [iid]}, 
                                        'test': {UserID: [iid]}}
        - bufferred_users: {'val': [UserID], 'test': [UserID]}
        - bufferred_negative_sample: {'val': {UserID: [iid]}}
        '''
        self.n_neg_val = args.n_neg_val
        self.n_neg_test = args.n_neg_test
        super(UserTransferEnvironment, self).__init__(args)
        self.set_target(args)
        
    def _read_data(self, args):
        
        print("Load domain model embeddings")
        # {domain_name: domain_model_path}
        self.domain_model_path = eval(open(args.domain_model_file, 'r').readline())
        # [domain_name]
        self.domains = list(self.domain_model_path.keys())
        # {domain_name: embedding_tensor}
        self.user_embeddings = {}
        for d, log_path in self.domain_model_path.items():
            model_args = extract_args(log_path)
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
        print("Load transfer data")
        self.data = {}
        self.data_dir = args.data_file
        df = pd.read_csv(args.data_file + 'train.tsv', header = 0,
                         names = ["UserID"] + self.domains + ["All"])
        df = df[["UserID"] + self.domains]
        self.data['train'] = df
        # {source_domain: {target_domain: [uid]}}
        self.common_user_list = get_common_user_list(df, self.domains)
        # {uid: row_id}
        self.user_row = {uid: idx for idx,uid in enumerate(df['UserID'].values)}
        # [uid]
        self.users = list(self.user_row.keys())
        
    def set_target(self, args):
        target_domain = args.target
        print("Load target domain environment")
        self.target_domain = target_domain if target_domain != 'None' else self.domains[0]
        model_args = extract_args(self.domain_model_path[target_domain])
        self.target_model = torch.load(model_args.model_path, 
                                       map_location='cpu')
        self.data['val'] = pd.read_csv(self.data_dir+target_domain+"_val.tsv", 
                                       index_col = 0)
        self.data['test'] = pd.read_csv(self.data_dir+target_domain+"_test.tsv", 
                                       index_col = 0)
        df = self.data['train']
        train_subset = df[df[target_domain]>0]
        condition_str = ' | '.join([f"(train_subset['{d}']>0)" \
                                    for d in self.domains if d != target_domain])
        print(f"Target domain: {target_domain}")
        print("Filtering condition: " + condition_str)
        train_subset = train_subset[eval(condition_str)]
        self.training_samples = list(train_subset.index)
        self._buffer_negative(args)
        
    def _buffer_negative(self, args):
        '''
        - bufferred_positive_val_sample: {UserID: [ItemID]}
        - bufferred_negative_val_sample: {UserID: [ItemID]}
        - bufferred_positive_test_sample: {UserID: [ItemID]}
        - all_item_candidates: set(ItemID)
        '''
        self.all_item_candidates = self.target_model.reader.all_item_candidates
        print("Buffer validation and test samples")
        self.bufferred_positive_sample = {"val":{}, "test":{}}
        self.bufferred_users = {}
        tgt_reader = self.target_model.reader
        for key in ['val','test']:
            df = self.data[key]
            user_hist = {}
            for idx in tqdm(range(len(df))):
                row = df.iloc[idx]
                uid, iid, r, t = row['UserID'], row['ItemID'], \
                                    row['Response'], row['Timestamp']
                if uid not in user_hist:
                    user_hist[uid] = []
                user_hist[uid].append((tgt_reader.get_item_feature(iid, "ItemID")
                                       ,r,t))
            self.bufferred_positive_sample[key] = user_hist
            self.bufferred_users[key] = list(user_hist.keys())

        print("Buffer negative sample for validation set")
        self.bufferred_negative_sample = {"val": {}}
        for uid, pos_items in tqdm(self.bufferred_positive_sample['val'].items()):
            negitems = np.array(sample_negative(pos_items,
                                                self.all_item_candidates, 
                                                n_neg = max(len(pos_items), 
                                                            args.n_neg_val)))
            self.bufferred_negative_sample['val'][uid] = negitems
        

    def get_statistics(self):
        stats = {'n_user': len(self.users), 'n_domains': len(self.domains), 
                 'emb_size': [self.user_emb_size[d] for d in self.domains]}
        return stats
        
    def get_train_dataset(self):
        return self
    
    def get_eval_dataset(self):
        return UserTransferEvalReader(self, len(self.bufferred_users[self.phase]), 
                                      n_worker = self.n_worker)
    
    def __len__(self):
        if self.phase == 'train':
            return len(self.training_samples)
        else:
            return len(self.bufferred_users[self.phase])
    
    def __getitem__(self, idx):
        if self.phase == 'train':
            # domain record row_id in data['train']
            row_id = self.training_samples[idx]
            # [UserID, D1_idx, ... , DN_idx]
            record = {}
        else:
            uid = self.bufferred_users[self.phase][idx]
            record = self.get_user_eval_info(uid)
            row_id = self.user_row[uid]
            if "no_item" in record:
                record['UniID'] = row_id + 1
                return record
        transfer_info = self.data['train'].iloc[row_id].to_dict()
        record['UniID'] = row_id + 1
        for d in self.domains:
            record[d] = transfer_info[d]
            record['emb_' + d] = self.user_embeddings[d][record[d]]
            record['mask_' + d] = 1. if record[d] > 0 else 0.
            
        return record
    
    def get_user_eval_info(self, uid):
        if uid not in self.bufferred_positive_sample[self.phase]:
            return {"no_item": True}
        if self.phase == 'val':
            items, resp, Ts = zip(*self.bufferred_positive_sample[self.phase][uid])
            neg_items = self.bufferred_negative_sample['val'][uid]
        elif self.phase == 'test':
            H = self.bufferred_positive_sample['val'][uid] +\
                    self.bufferred_positive_sample['test'][uid]
            items, resp, Ts = zip(*H)
            neg_items = sample_negative(items, self.all_item_candidates, 
                                        self.n_neg_test)
        else:
            raise NotImplemented
        # construct data
        user_data = {"resp": np.array(resp),
                     "item_ItemID": np.array(items),
                     "negi_ItemID": np.array(neg_items)}
        return user_data
    
    def get_target_domain_performance(self, feed_dict, at_k_list):
        '''
        @input:
        - transfer_model
        - feed_dict: {
            ${domain_name}: (1,), # [uid]
            emb_${domain_name}: (1,dim),
            mask_${domain_name}: (1,), # [0,1]-mask for existence in domain
            All: [uid]
            item_ItemID: (1,P)
            negi_ItemID: (1,N)
            resp: (1,P)
        }
        '''
        u_emb = feed_dict['transfer_out'][:,:-1]
        u_bias = feed_dict['transfer_out'][:,-1]
        pos_out = self.target_model.forward_with_emb({
                                    'user_emb': u_emb, 
                                    'user_bias': u_bias, 
                                    'ItemID': feed_dict['item_ItemID']})
        neg_out = self.target_model.forward_with_emb({
                                    'user_emb': u_emb, 
                                    'user_bias': u_bias, 
                                    'ItemID': feed_dict['negi_ItemID']})
        pos_preds = torch.sigmoid(pos_out['preds'])
        neg_preds = torch.sigmoid(neg_out['preds'])
        return calculate_ranking_metric(pos_preds.view(-1), 
                                        neg_preds.view(-1), 
                                        at_k_list)
    