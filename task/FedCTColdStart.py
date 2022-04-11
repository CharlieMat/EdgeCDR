import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pickle
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import utils
from task.TopK import init_ranking_report, calculate_ranking_metric, TopK
from task.ColdStartTopK import ColdStartTopK
from reader.BaseReader import worker_init_func
    
class FedCTColdStart(ColdStartTopK):
    
    @staticmethod
    def parse_task_args(parser):
        '''
        - args from TopK
            - at_k
            - n_eval_process
            - args from GeneralTask:
                - optimizer
                - epoch
                - check_epoch
                - lr
                - batch_size
                - eval_batch_size
                - with_val
                - with_test
                - val_sample_p
                - test_sample_p
                - stop_metric
                - pin_memory
        '''
        parser = TopK.parse_task_args(parser)
        parser.add_argument('--step_eval', type=int, default=-1, 
                            help='number steps between intra-epoch evaluation, -1 if ignore')
        parser.add_argument('--n_sync_per_epoch', type=int, default=0,
                            help='In-epoch synchronization')
        return parser
    
    def __init__(self, args, reader):
        self.step_eval = args.step_eval
        self.local_rank = args.local_rank
        self.is_pivot_gpu = args.local_rank == 0
        self.n_sync_per_epoch = args.n_sync_per_epoch
        self.world_size = dist.get_world_size()
        super().__init__(args, reader)
        # multi-GPU for distributed data parallel
        
    def log(self):
        super().log()
        print(f"\tn_sync_per_epoch: {self.n_sync_per_epoch}")
        
        
    def train(self, model, continuous = False):
        super().train(model, continuous)
        dist.barrier()
        
    def do_epoch(self, model, epoch_id):
        model.reader.set_phase("train")
        sampler = DistributedSampler(model.reader)
        train_loader = DataLoader(model.reader, sampler=sampler, batch_size = self.batch_size,
                                  shuffle = False, pin_memory = self.pin_memory,
                                  num_workers = self.n_worker)
        torch.cuda.empty_cache()

        model.train()
        step_loss = []
        dropout_count = 0
        pbar = tqdm(total = int(len(model.reader) / self.world_size) + 1)
        n_sync = self.n_sync_per_epoch
        sync_at_step = int(len(model.reader) / (self.world_size * (n_sync + 1) * self.batch_size))
        self.do_print(f"Sync every {sync_at_step} batches")
        for i, batch_data in enumerate(train_loader):
            gc.collect()
            feed_dict = model.wrap_batch(batch_data)
            if i == 0 and epoch_id == 1:
                self.show_batch(feed_dict)
            for j,local_uid in enumerate(feed_dict['UserID'].view(-1).detach().cpu().numpy()):
                local_info = {'epoch':epoch_id, 'lr': self.lr, 
                              'edge_id': local_uid}
                local_feed_dict = {k: v[j] for k,v in feed_dict.items()}
                local_info = model.get_local_info(local_feed_dict, local_info)

                # imitate user dropout in FL (e.g. connection lost or no response)
                if model.do_device_dropout(local_info):
                    dropout_count += 1
                    continue

                # download model parameters to personal spaces
                model.download_cloud_params(local_info)

                # local optimization
                local_response = model.local_optimize(local_feed_dict, local_info) 
                step_loss.append(local_response["loss"])
                # upload updated model parameters to the cloud of each domain
                model.upload_edge_params(local_info)
                
            if (i+1) % sync_at_step == 0 and n_sync > 0:
                dist.barrier()
                self.do_print("in-epoch sync")
                model.mitigate_params()
                model.reset_proposal()
                n_sync -= 1

            pbar.update(self.batch_size)
            if self.step_eval > 0 and i > self.step_eval:
                break
        pbar.close()
        print("Wait for end of epoch")
        dist.barrier()
#         model.download_cloud_params(None) # synchronize parameter for model saving
        print(f"#dropout device (cuda:{self.local_rank}): {dropout_count}")
        return {"loss": np.mean(step_loss), "step_loss": step_loss}

    def do_eval(self, model):
        """
        Evaluate the results for an eval dataset.
        @input:
        - model: GeneralRecModel or its extension
        
        @output:
        - resultDict: {metric_name: metric_value}
        """
        report = {}
        model.download_cloud_params(None)
        if self.is_pivot_gpu:
            print("Evaluating...")
            print("Sample p = " + str(self.eval_sample_p))
            model.eval()
            if model.loss_type == "regression": # rating prediction evaluation
    #             report = self.evaluate_regression(model)
                raise NotImplemented
            else: # ranking evaluation
                report = self.evaluate_userwise_ranking(model)
                keys, report_values = list(report.keys()), list(report.values()) 
            print("Result dict:")
        else:
            report = init_ranking_report(self.at_k_list)
            keys, report_values = list(report.keys()), [report[k] * 0. for k in report]
        dist.barrier()
        dist.broadcast_object_list(report_values, src = 0)
        report = {k: report_values[i] for i,k in enumerate(keys)}
        self.do_print(str(report))
        return report
    
    def get_after_epoch_info(self, model):
        return {'local_rank': self.local_rank}