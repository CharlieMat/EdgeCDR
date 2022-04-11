import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

from model.components import init_weights
from model.general import BaseModel

################################
#      Matching Models         #
################################

class Matching_Linear(BaseModel):
    
    @staticmethod
    def parse_model_args(parser):
        parser = BaseModel.parse_model_args(parser)
        return parser
    
    def __init__(self, args, reader, device):
        self.domains = reader.domains # [domain_name]
        self.domain_dim = reader.user_emb_size # {domain_name: scalar}
        self.target_domain = reader.target_domain
        reader.target_model.device = device
        reader.target_model.to(device)
        super(Matching_Linear, self).__init__(args, reader, device)
        
    def log(self):
        self.reader.log()
        print("Model params")
        print("\tmodel_path = " + str(self.model_path))
        print("\tloss_type = " + str(self.loss_type))
        print("\tl2_coef = " + str(self.l2_coef))
        print("\tdevice = " + str(self.device))

    def _define_params(self, args, reader):
        self.mappings = {source_domain: {} for source_domain in self.domains}
        j = self.domains.index(self.target_domain)
        for i,source_domain in enumerate(self.domains):
#             for j,target_domain in enumerate(self.domains):
            mapping = nn.Linear(self.domain_dim[source_domain], 
                                self.domain_dim[self.target_domain])
            init_weights(mapping)
            self.mappings[source_domain][self.target_domain] = mapping
            self.add_module(f"mapping_{i}-{j}", mapping)
        
    def wrap_batch(self, batch):
        wrapped_batch = super().wrap_batch(batch)
        return wrapped_batch
                
    def get_forward(self, feed_dict):
        '''
        transfer from single source embedding to the target
        @input:
        - feed_dict: {
            domain_name: (B,), # [uid]
            emb_domain_name: (B,dim),
            mask_domain_name: (B,), # [0,1]-mask for existence in domain
            All: [uid]
        }
        '''
        out_emb = []
        out_weight = []
        reg = []
        for source_domain in self.domains:
            if source_domain != self.target_domain:
                mapping_output = self.get_mapped_emb(
                                        feed_dict['emb_'+source_domain], 
                                        source_domain)
                out_emb.append(mapping_output['preds']\
                               .view(1,-1,self.domain_dim[self.target_domain]))
                out_weight.append(feed_dict['mask_' + source_domain].view(1,-1,1))
                reg.append(mapping_output['reg'])
#                 print(source_domain, out_emb[-1].shape, out_weight[-1].shape)
        # output embedding
        out_emb = torch.cat(out_emb, dim = 0)
        out_weight = torch.cat(out_weight, dim = 0) + 1e-6
        domain_weight = out_weight / torch.sum(out_weight, dim = 0, keepdim = True)
        out_emb = out_emb * domain_weight
#         print('aggregate', out_emb.shape, out_weight.shape)
        # regularization
        reg_weight = torch.sum(out_weight, dim = 1).view(-1)
        reg_weight = reg_weight / torch.sum(reg_weight)
#         print('reg_weight', reg_weight)
        combined_reg = 0.
        for i,w in enumerate(reg_weight):
            combined_reg = reg[i] * w + combined_reg
#         input()
        return {'preds': torch.sum(out_emb, dim = 0), 'reg': combined_reg}
        
    def get_mapped_emb(self, emb, source_domain):
        mapping_module = self.mappings[source_domain][self.target_domain]
        return {'preds': mapping_module(emb), 
                'reg': self.get_regularization(mapping_module)}
    
    def get_loss(self, feed_dict, out_dict):
        '''
        @input:
        - feed_dict: {
            domain_name: [uid],
            emb_domain_name: (B,dim),
            mask_domain_name: (B,), # [0,1]-mask for existence in domain
            All: [uid]
        }
        - out_dict: {
            emb_pred: (B,dim)
        }
        '''
        pred = out_dict['preds'] # (B,dim)
        target_emb = feed_dict['emb_' + self.target_domain] # (B,dim)
        if self.loss_type == "rmse":
            loss = torch.sqrt(F.mse_loss(pred, target_emb))
        elif self.loss_type == "mae":
            loss = F.l1_loss(pred, target_emb)
        elif self.loss_type == "cosine":
            loss = F.cosine_embedding_loss(pred, target_emb, torch.ones(pred.shape[0]).to(self.device))
        else:
            raise NotImplemented
        loss = loss + self.l2_coef * out_dict['reg']
        return loss
    


                

