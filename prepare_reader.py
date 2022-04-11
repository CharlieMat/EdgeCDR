import os
import time
import argparse

import numpy as np
import torch

import utils


#################################################################################
#                              Command Interface                                #
#################################################################################

from model.baselines import *
from reader import *
from task import *

if __name__ == '__main__':
    
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--reader', type=str, default='NextItemReader', help='Create a model to run.')
    init_parser.add_argument('--reader_path', type=str, help='Reader save path')
    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)
    readerClass = eval('{0}.{0}'.format(initial_args.reader))

    # control args
    parser = argparse.ArgumentParser()
    
    # customized args
    parser = readerClass.parse_data_args(parser)
    parser.add_argument('--seed', type=int, default=9,
                        help='random seed')
    args, _ = parser.parse_known_args()
    print(args)
    
    # reproducibility
    utils.set_random_seed(args.seed)
    
    # reader
    print("Setup reader")
    reader = readerClass(args)
    print(reader.get_statistics())
    
    torch.save(reader, initial_args.reader_path)
    
    
    