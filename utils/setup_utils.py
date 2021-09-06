# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from dataloader import (
    DataCollatorForMatching, DatasetForMatching,
    MultiProcessDataLoaderForMatching, SingleProcessDataLoaderForMatching)
import logging
from transformers import BertTokenizerFast
from models.MoPQ import MoPQ
from models.TextEncoder import TextEncoder
from models.Differentiable_PQ import DPQ
from transformers import AutoConfig, AutoModel

def setup_worker(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    # Explicitly setting seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

def cleanup():
    dist.destroy_process_group()


def setup_loader(args,
                 local_rank=0,
                 end=None,
                 blocking=False,
                 multi_process=True,
                 tokenizer=None,
                 mode="train", 
                 buffer_batchs=None):
    datapath = {'train': args.train_data_path,
                'valid': args.valid_data_path,
                'test': args.test_data_path}[mode]

    batch_size = {'train': args.train_batch_size,
                  'valid': args.valid_batch_size,
                  'test': args.test_batch_size}[mode]

    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    data_collator = DataCollatorForMatching(tokenizer=tokenizer,
                                            args=args)
    dataset = DatasetForMatching(tokenizer=tokenizer,
                                 file_path=datapath)

    if multi_process:
        dataloader = MultiProcessDataLoaderForMatching(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            local_rank=local_rank,
            world_size=args.world_size,
            global_end=end,
            buffer_num=buffer_batchs,
            blocking=blocking)
    else:
        # mostly for testing
        dataloader = SingleProcessDataLoaderForMatching(
            dataset, batch_size=batch_size, collate_fn=data_collator)
    return dataloader


def setup_manager():
    mgr = mp.Manager()
    end = mgr.Value('b', False)
    end_train = mgr.Value('b', False)
    return mgr, end, end_train


def setup_model(args, local_rank):
    config = AutoConfig.from_pretrained(args.bert_model)
    config.num_hidden_layers = args.layers
    bert = AutoModel.from_pretrained(args.bert_model, config=config)
    if args.model_type == 'MoPQ':
        model = MoPQ(args, bert, key_bert=None, hidden_size=config.hidden_size, local_rank=local_rank)
    else:
        model = DPQ(args, bert, key_bert=None, hidden_size=config.hidden_size, local_rank=local_rank)
    return model