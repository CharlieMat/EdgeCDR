import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist

from model.components import init_weights, setup_dnn_layers
from model.general_fed import FederatedModel

################################
#         DUE Models           #
################################

class DUE(FederatedModel):
    
    @staticmethod
    def parse_model_args(parser):
        parser = FederatedModel.parse_model_args(parser)
        parser.add_argument('--due_dim', type=int, required=True, 
                            help='cross-domain user embedding dimension size')
        parser.add_argument('--hidden_dims', type=int, nargs='?', default=[], 
                            help='specificy a list of hidden dimension size')
        return parser
    
    def log(self):
        super().log()
        print("\tdue_dim = " + str(self.due_dim))
        print("\thidden_dims = " + str(self.hidden_dims))
        
    def __init__(self, args, reader, device):
        '''
        - due_dim
        - hidden_dims
        - domains
        - domain_dim
        - from FederatedModel:
            - device_dropout_p: overall dropout probability in [0,1)
            - individual_p: dropout probability for each individual
            - device_dropout_type: 'same' or 'max'
            - n_local_step
            - random_local_step: boolean
            - mitigation_beta: scalar
            - n_device: number of users
            - aggregation_func: 'FedAvg' or 'FedProx'
            - elastic_mu: scalar
            - from BaseModel:
                - model_path
                - loss
                - l2_coef
        '''
        self.due_dim = args.due_dim
        self.hidden_dims = args.hidden_dims
        self.domains = reader.domains # [domain_name]
        self.domain_dim = reader.user_emb_size # {domain_name: scalar}
        reader.target_model.device = device
        reader.target_model.to(device)
        
        super(DUE, self).__init__(args, reader, device)
        self.MSE = nn.MSELoss(reduction = 'none')

    def _define_params(self, args, reader):
        # DUE: decentralized user embeddings, they are assumed maintained in each local space
        self.dueEmb = nn.Embedding(len(reader.users)+1, self.due_dim)
        with torch.no_grad():
            self.dueEmb.weight.data = self.dueEmb.weight.data * 0
#         self.z_layer_norm = nn.LayerNorm(args.due_dim)
        init_weights(self.dueEmb)
        # Encoder and decoders
        self.encoders = {}
        self.decoders = {}
        for i,source_domain in enumerate(self.domains):
            enc = setup_dnn_layers(self.domain_dim[source_domain], self.hidden_dims, self.due_dim)
            dec = setup_dnn_layers(self.due_dim, self.hidden_dims, self.domain_dim[source_domain])
            self.encoders[source_domain] = enc
            self.decoders[source_domain] = dec
            self.add_module(f"enc_{i}", enc)
            self.add_module(f"dec_{i}", dec)
        
        self.domain_params = {d: {} for d in self.domains}
        for name, param in self.named_parameters():
            for i,d in enumerate(self.domains):
                if f'enc_{i}.' in name or f'dec_{i}.' in name:
                    self.domain_params[d][name] = param
                
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
        out_emb = {}
        out_z = {}
        reg = {}
        # z|E_u
        z_on_edge = self.dueEmb(feed_dict['UserID']) # (B, z_dim)
        # each domain learn its own auto-encoder
        for source_domain in self.domains:
            # z|U_{d,u}, size (B, z_dim)
            z_given_domain = self.encoders[source_domain](feed_dict['emb_' + source_domain])
#             z_given_domain = F.softmax(z_given_domain, dim = -1)
#             z_given_domain = torch.sigmoid(z_given_domain)
            out_z[source_domain] = z_given_domain.view(-1,self.due_dim)
            # U_{d,u}|E_u, size (B, domain_dim)
            emb_given_z = self.decoders[source_domain](z_on_edge)
            out_emb[source_domain] = emb_given_z.view(-1,self.domain_dim[source_domain])
            # regularization term, scalar
            reg[source_domain] = self.get_regularization(self.encoders[source_domain], 
                                                         self.decoders[source_domain])
        return {'preds': {"out_emb": out_emb, 
                          "out_z": out_z, 
                          "edge_z": z_on_edge}, 
                'reg': reg}
        
    def forward(self, feed_dict: dict) -> dict:
        '''
        Called during evaluation or prediction
        '''
        out_dict = self.get_forward(feed_dict)
        return out_dict
        
#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = Variable(std.data.new(std.size()).normal_())
#         z = eps.mul(std) + mu
#         return z
    
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
        out_emb = out_dict['preds']['out_emb'] # {domain_id: (B,dim)}
        out_z = out_dict['preds']['out_z'] # {domain_id: (B,z_dim)}
        edge_z = out_dict['preds']['edge_z'] # {domain_id: (B,z_dim)}
        target_emb = {d:feed_dict['emb_' + d] for d in self.domains} # {domain_id: (B,dim)}
        loss = 0.
        reg = torch.mean(edge_z * edge_z)
        if self.loss_type == "ae":
            for d in self.domains:
                d_mask = feed_dict['mask_' + d].view(-1,1) # (B,1)
                # encoder loss
                loss = torch.mean(self.MSE(out_z[d].view(-1,self.due_dim), 
                                           edge_z.view(-1,self.due_dim)) * d_mask) + loss
                # decoder loss
                loss = torch.mean(self.MSE(out_emb[d].view(-1,self.domain_dim[d]), 
                                          target_emb[d].view(-1,self.domain_dim[d])) * d_mask) + loss
                reg = out_dict['reg'][d] * torch.mean(d_mask) + reg
#         elif self.loss_type == "vae":
#             for d in self.domains:
#                 d_mask = torch.sum(feed_dict['mask_' + d])
#                 # encoder loss
#                 loss = torch.mean(self.MSE(out_z[d], edge_z)) * d_mask + loss
#                 mu, logvar = 
#                 # decoder loss
#                 loss = torch.sum(self.MSE(out_emb[d], target_emb[d])) * d_mask + loss
#                 KLD = - 0.5 * torch.sum(1 + logvar - pLogvar - (logvar.exp() + (mu - pMu).pow(2)) / pLogvar.exp())
#                 reg = out_dict['reg'][d] * d_mask + reg
        else:
            raise NotImplemented
        loss = loss + self.l2_coef * reg
        return loss
    

    ##################################
    #        federated control       #
    ##################################
    
    '''
    General federated learning with central node:
    model.actions_before_train() --> model.keep_cloud_params()
    for each epoch:
        model.actions_before_epoch() --> model.reset_proposal()
        for each user_batch:
            local_info = model.get_local_info(user_batch, {some training info})
            if not model.do_device_dropout():
                model.download_cloud_params()
                model.local_optimize()
                model.upload_edge_params()
        model.actions_after_epoch() --> model.mitigate_params()
    '''
    
            
    def keep_cloud_params(self):
        '''
        Keep a copy of current parameters as cloud copy. 
        '''
        print("Copy parameters to cloud space")
        self.cloud_params = {name: param.data.clone() for name, param in self.named_parameters()\
                             if 'enc' in name or 'dec' in name or 'due' in name}
            
#     def reset_proposal(self):
#         '''
#         Reset proposal buffer for later participant uploads
#         '''
#         self.param_proposal = {k: torch.zeros_like(v) for k,v in self.cloud_params.items()}
#         self.param_proposal_count = {k: torch.zeros_like(v) for k,v in self.cloud_params.items()}
    
#     def download_cloud_params(self, local_info):
#         '''
#         Copy cloud parameters into local parameters
#         - ignore the download of decentralized user encoding since it is maintained on edge
#         '''
#         with torch.no_grad():
#             for name, param in self.named_parameters():
#                 if 'enc' in name or 'dec' in name:
#                     param.data = self.cloud_params[name].clone()

    def local_optimize(self, feed_dict, local_info):
        local_response = super().local_optimize(feed_dict, local_info)
        local_info['involved_domain'] = [d for d in self.domains if torch.sum(feed_dict['mask_' + d])>0]
        return local_response
    
    def upload_edge_params(self, local_info):
        '''
        Upload edge parameters to cloud
        '''
        involved_domains = local_info['involved_domain']
        with torch.no_grad():
            for d in involved_domains:
                for name, param in self.domain_params[d].items():
                    self.param_proposal[name] += param.data.clone()
                    self.param_proposal_count[name] += 1
            for name, param in self.named_parameters():
                if 'due' in name:
                    eid = local_info['edge_id']
                    self.param_proposal[name][eid] += param.data[eid]
                    self.param_proposal_count[name][eid] += 1
                    
    def mitigate_params(self):
        '''
        Mitigate parameters and update cloud parameters
        Final mapping is the convex combination: (1 - alpha) * old_params + alpha * average_proposal
        '''
        with torch.no_grad():
            for name, param in self.cloud_params.items():
                dist.all_reduce(self.param_proposal[name], op = dist.ReduceOp.AVG)
                dist.all_reduce(self.param_proposal_count[name], op = dist.ReduceOp.AVG)
#                 if 'enc' in name or 'dec' in name:
#                 sum_grad = self.param_proposal[name]  - \
#                                 self.cloud_params[name] * self.param_proposal_count[name]
#                 self.cloud_params[name] = self.cloud_params[name] +\
#                                 self.mitigation_beta * sum_grad
                sum_grad = self.param_proposal[name] / (self.param_proposal_count[name] + 1e-9) - \
                                self.cloud_params[name]
                self.cloud_params[name] = self.cloud_params[name] +\
                                self.mitigation_beta * sum_grad
        dist.barrier()
#             for name, param in self.named_parameters():
#                 if 'due' in name:
#                     self.cloud_params[name] = param.clone()
                    
#     def actions_after_epoch(self, info):
#         self.mitigate_params() # mitigate proposals
#         if 'local_rank' in info:
#             local_rank = info['local_rank']
#             dist.barrier()
#             communicate cloud parameters between GPUs
#                 print(f"before agg (cuda:{local_rank}): {self.cloud_params['enc_1.h0.weight'][1,:5]}")
#             for name, param in self.cloud_params.items():
#                 dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
#                 print(f"after agg (cuda:{local_rank}): {self.cloud_params['enc_1.h0.weight'][1,:5]}")
#             for name, param in self.named_parameters():
#                 if 'due' in name:
#                     print(f"before agg (cuda:{local_rank}): {param.data[641049]}")
#                     dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
#                     print(f"before agg (cuda:{local_rank}): {param.data[641049]}")
                    
    
                    
                    
