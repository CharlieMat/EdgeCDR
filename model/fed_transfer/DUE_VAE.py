import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.distributed as dist

from model.components import init_weights, setup_dnn_layers
from model.general_fed import FederatedModel
from model.fed_transfer.DUE import DUE

################################
#         DUE Models           #
################################

class DUE_VAE(DUE):
    
    @staticmethod
    def parse_model_args(parser):
        parser = DUE.parse_model_args(parser)
        return parser
    
    def log(self):
        super().log()
        
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
        super(DUE_VAE, self).__init__(args, reader, device)

    def _define_params(self, args, reader):
        # DUE: decentralized user embeddings, they are assumed maintained in each local space
        self.dueEmb = nn.Embedding(len(reader.users)+1, self.due_dim * 2)
        with torch.no_grad():
            self.dueEmb.weight.data[:,:self.due_dim] = 0.
            self.dueEmb.weight.data[:,self.due_dim:] = 1.
#         self.z_layer_norm = nn.LayerNorm(args.due_dim)
        init_weights(self.dueEmb)
        # Encoder and decoders
        self.encoders = {}
        self.decoders = {}
        for i,source_domain in enumerate(self.domains):
            enc = setup_dnn_layers(self.domain_dim[source_domain], self.hidden_dims, self.due_dim * 2)
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
        z_on_edge_stats = self.dueEmb(feed_dict['UserID']).view(-1,self.due_dim*2) # (B, z_dim)
        z_on_edge_mu, z_on_edge_logvar = z_on_edge_stats[:,:self.due_dim], z_on_edge_stats[:, self.due_dim:]
        z_on_edge = self.reparametrize(z_on_edge_mu, z_on_edge_logvar)
        # each domain learn its own auto-encoder
        for source_domain in self.domains:
            # z|U_{d,u}, size (B, z_dim)
            z_given_domain = self.encoders[source_domain](feed_dict['emb_' + source_domain])
#             z_mu, z_logvar = z_stats[:,:self.due_dim], z_stats[:,self.due_dim]
#             z_given_domain = self.get_reparametrization(z_mu, z_logvar)
#             z_given_domain = F.softmax(z_given_domain, dim = -1)
#             z_given_domain = torch.sigmoid(z_given_domain)
            out_z[source_domain] = z_given_domain.view(-1,self.due_dim*2)
            # U_{d,u}|E_u, size (B, domain_dim)
            emb_given_z = self.decoders[source_domain](z_on_edge)
            out_emb[source_domain] = emb_given_z.view(-1,self.domain_dim[source_domain])
            # regularization term, scalar
            reg[source_domain] = self.get_regularization(self.encoders[source_domain], 
                                                         self.decoders[source_domain])
        return {'preds': {"out_emb": out_emb, 
                          "out_z": out_z, 
                          "edge_z": z_on_edge_stats,}, 
                'reg': reg}
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std) + mu
        return z
    
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
        out_z = out_dict['preds']['out_z'] # {domain_id: (B,z_dim * 2)}
        edge_z = out_dict['preds']['edge_z'] # (B,z_dim * 2)
        edge_mu, edge_logvar = edge_z[:,:self.due_dim], edge_z[:, self.due_dim:]
        target_emb = {d:feed_dict['emb_' + d] for d in self.domains} # {domain_id: (B,dim)}
        loss = 0.
        reg = torch.mean(edge_z * edge_z)
        if self.loss_type == "vae":
            for d in self.domains:
                d_mask = feed_dict['mask_' + d].view(-1,1) # (B,1)
                # encoder loss
                loss = torch.mean(self.MSE(out_z[d].view(-1,self.due_dim * 2), 
                                           edge_z.view(-1,self.due_dim * 2)) * d_mask) + loss
                # decoder loss
                loss = torch.mean(self.MSE(out_emb[d].view(-1,self.domain_dim[d]), 
                                          target_emb[d].view(-1,self.domain_dim[d])) * d_mask) + loss
#                 kld = - 0.5 * torch.mean(1 + logvar - pLogvar - (logvar.exp() + (mu - pMu).pow(2)) / pLogvar.exp())
                kld = - 0.5 * torch.sum((1 + edge_logvar - edge_logvar.exp() - edge_mu.pow(2)) * d_mask, dim = 1)
                loss = loss + torch.mean(kld)
                # regularization
                reg = out_dict['reg'][d] * torch.mean(d_mask) + reg
        else:
            raise NotImplemented
        loss = loss + self.l2_coef * reg
        return loss
    
                    
    
                    
                    
