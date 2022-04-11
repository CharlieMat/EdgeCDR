import os
import time
import argparse
import setproctitle

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utils


#################################################################################
#                              Command Interface                                #
#################################################################################

from model.baselines import *
from model.fed_transfer import *
from reader import *
from task import *

if __name__ == '__main__':
    
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--proctitle', type=str, default='Iron Man', help='process title on CLT')
    init_parser.add_argument('--model', type=str, default='MF', help='Create a model to run.')
    init_parser.add_argument('--reader', type=str, default='ColdStartTransferEnvironment', help='Reader environment.')
    init_parser.add_argument('--task', type=str, default='TopK', help='Task to run')
    initial_args, _ = init_parser.parse_known_args()
    
    modelClass = eval('{0}.{0}'.format(initial_args.model))
    taskClass = eval('{0}.{0}'.format(initial_args.task))
    readerClass = eval('{0}.{0}'.format(initial_args.reader))
#     setproctitle.setproctitle(initial_args.proctitle+"("+initial_args.model+"-"+initial_args.task+")")
    setproctitle.setproctitle(initial_args.proctitle)

    # control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=9,
                        help='random seed')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Run train")
    mode_group.add_argument("--train_and_eval", action="store_true", help="Run train")
    mode_group.add_argument("--continuous_train", action="store_true", help="Run continous train")
    mode_group.add_argument("--eval", action="store_true", help="Run eval")
    
    # GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    is_pivot = local_rank == 0
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    
    # customized args
    parser = modelClass.parse_model_args(parser)
    parser = readerClass.parse_data_args(parser)
    parser = taskClass.parse_task_args(parser)
    args, _ = parser.parse_known_args()
    args.local_rank = local_rank
    if is_pivot:
        print(initial_args)
        print(args)
    dist.barrier()
    
    # reproducibility
    utils.set_random_seed(args.seed)
    
    # reader
    print("Setup reader")
    reader = readerClass(args)
    reader.set_target_device(device)
    reader.im_gonna_print = is_pivot
    
    if is_pivot:
        print(reader.get_statistics())
    
    
    # task
    print("Setup task")
    task = taskClass(args, reader)
    task.set_device(device)
    task.im_gonna_print = is_pivot
    
    # run task
    if args.train or args.train_and_eval:
        # train model
        for i in range(task.n_round):
            dist.barrier()
            # model
            model = modelClass(args, reader, device)
            if is_pivot:
                print(f"#######################\r\n#       Round {i+1}       #\r\n#######################")
                model.log()
                task.log()
                model.show_params()
            model = model.to(device)
            task.train(model, continuous = False)
            if args.train_and_eval:
                print(f"Pivot device ({device}) evaluation")
                task.do_test(model)
            
    else:
        model = modelClass(args, reader, device)
        model.to(device)
        if args.continuous_train:
            task.train(model, continuous = True)
        task.do_test(model)
    
    
    