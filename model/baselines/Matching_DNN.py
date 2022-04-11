# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from collections import OrderedDict

# from model.baselines import Matching_Linear

                
# class Matching_DNN(Matching_Linear):
    
#     @staticmethod
#     def parse_model_args(parser):
#         parser = Matching_Linear.parse_model_args(parser)
#         parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128], 
#                             help='specificy a list of hidden dimension size')
#         parser.add_argument('--dropout_rate', type=float, default=0.3, 
#                             help='dropout rate in deep layers')
#         parser.add_argument('--batch_norm', action='store_true', 
#                             help='add batch normalization in deep layers')
#         return parser
    
#     def __init__(self, args, reader, device):
#         super().__init__(args, reader, device)
        
#     def log(self):
#         super().log()
#         print("\thidden_dims = " + str(self.hidden_dims))
#         print("\tdropout_rate = " + str(self.dropout_rate))
#         print("\tbatch_norm = " + str(self.batch_norm))
    
#     def _define_params(self, args, reader):
#         self.hidden_dims = args.hidden_dims
#         assert len(self.hidden_dims) > 0
#         j = self.domains.index(self.target_domain)
#         self.mappings = {source_domain: {} for source_domain in self.domains}
#         for i,source_domain in enumerate(self.domains):
# #             for j,target_domain in enumerate(self.domains):
        
#             layers = [(f'h0', 
#                           nn.Linear(self.domain_dim[source_domain], 
#                                     hidden_sizes[0])),\
#                          (f'a0', nn.ReLU())]
#             init_weights(layers[0][1])
#             if args.dropout_rate > 0:
#                 layers.append((f'dropout0', 
#                                nn.Dropout(args.dropout_rate)))
#             if args.batch_norm:
#                 layers.append((f'bn0',
#                                nn.BatchNorm1d(self.hidden_dims[k-1])))
#             for k in range(1,len(self.hidden_dims)):
#                 layers.append((f'h{k}', 
#                                   nn.Linear(hidden_sizes[k-1], 
#                                             hidden_sizes[k])))
#                 init_weights(layers[-1][1])
#                 layers.append((f'a{k}', nn.ReLU()))
#                 if args.dropout_rate > 0:
#                     layers.append((f'dropout{k}', 
#                                    nn.Dropout(args.dropout_rate)))
#                 if args.batch_norm:
#                     layers.append((f'bn{k}',
#                                    nn.BatchNorm1d(self.hidden_dims[k-1])))
#             layers.append((f'h{len(hidden_sizes)+1}',
#                               nn.Linear(hidden_sizes[-1], 
#                                         self.domain_dim[self.target_domain])))
#             init_weights(layers[-1][1])
#             mapping = nn.Sequential(OrderedDict(layers))
#             self.mappings[source_domain][self.target_domain] = mapping
#             self.add_module(f'mapping_{i}-{j}', mapping)