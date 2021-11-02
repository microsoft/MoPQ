# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Callable, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from filelock import FileLock
from argparse import Namespace
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import IterableDataset
import torch.distributed as dist
from transformers import BertTokenizerFast
from transformers.utils import logging
from tqdm import tqdm
from random import shuffle

from utils.utils import setuplogging

logger = logging.get_logger(__name__)


class DatasetForMatching(IterableDataset):
    def __init__(self,
                 file_path: str):
        self.data_file = open(file_path,
                              "r",
                              encoding="utf-8")

    def __iter__(self):
        for line in self.data_file:
            tokens_and_vecs = json.loads(line)
            yield tokens_and_vecs


# @dataclass
class DataCollatorForMatching:
    def __init__(self, tokenizer: BertTokenizerFast,
                 args: Namespace):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(
            self,
            samples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_id_queries = []
        attention_mask_queries = []
        input_id_keys = []
        attention_mask_keys = []
        queries_vec = []
        keys_vec = []
        for i, sample in (enumerate(samples)):
            (input_id_query,
             attention_mask_query,
             input_id_key,
             attention_mask_key,
             query_vec,
             key_vec) = self.create_training_sample(sample)

            if input_id_query is not None:
                input_id_queries.append(input_id_query)
                attention_mask_queries.append(attention_mask_query)
            else:
                queries_vec.append(query_vec)

            if input_id_key is not None:
                input_id_keys.append(input_id_key)
                attention_mask_keys.append(attention_mask_key)
            else:
                keys_vec.append(key_vec)

        if len(queries_vec) == 0:
            input_id_queries = self._tensorize_batch(
                input_id_queries, self.tokenizer.pad_token_id).long()
            attention_mask_queries = self._tensorize_batch(attention_mask_queries, 0).float()
            queries_vec = None
        else:
            # print(queries_vec)
            queries_vec = torch.FloatTensor(queries_vec)
            input_id_queries, attention_mask_queries = None, None

        if len(keys_vec) == 0:
            input_id_keys = self._tensorize_batch(input_id_keys,
                                                  self.tokenizer.pad_token_id).long()
            attention_mask_keys = self._tensorize_batch(attention_mask_keys, 0).float()
            keys_vec = None
        else:
            keys_vec = torch.FloatTensor(keys_vec)
            input_id_keys, attention_mask_keys = None, None

        return {
            "input_id_query":
            input_id_queries,
            "attention_mask_query":
            attention_mask_queries,
            "input_id_key":
            input_id_keys,
            "attention_mask_key":
            attention_mask_keys,
            "queries_vec":
            queries_vec,
            "keys_vec":
            keys_vec
        }

    def _tensorize_batch(self, examples: List[torch.Tensor],
                         padding_value) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(
            x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples,
                                batch_first=True,
                                padding_value=padding_value)

    def create_training_sample(self, sample: Dict):
        """Turn sample into train pair

        Args:
            sample (Dict): {"query_tokens":List[int], "query_vec":List[float], "key_tokens":List[int], "key_vec":List[float]}

        """
        input_id_queries, attention_mask_queries, input_id_keys, attention_mask_keys, query_vec, key_vec = None, None, None, None, None, None
        if self.args.self_supervised:
            assert 'key_tokens' in sample
            input_id_keys, attention_mask_keys = self.creat_tokens_sample(sample['key_tokens'], self.args.key_max_token)
            token_queries = self.creat_query_from_key(sample['key_tokens'])
            input_id_queries, attention_mask_queries = self.creat_tokens_sample(token_queries, self.args.query_max_token)
            return input_id_queries, attention_mask_queries, input_id_keys, attention_mask_keys, query_vec, key_vec

        if 'query_tokens' in sample:
            input_id_queries, attention_mask_queries = self.creat_tokens_sample(sample['query_tokens'],
                                                                                self.args.query_max_token)
        else:
            query_vec = self.creat_vecs_sample(sample['query_vec'])

        if 'key_tokens' in sample:
            input_id_keys, attention_mask_keys = self.creat_tokens_sample(sample['key_tokens'],
                                                                                self.args.query_max_token)
        else:
            key_vec = self.creat_vecs_sample(sample['key_vec'])

        return input_id_queries, attention_mask_queries, input_id_keys, attention_mask_keys, query_vec, key_vec

    def creat_query_from_key(self, token_keys):
        if len(token_keys) > 1:
            if self.args.augment_method == 'drop':
                token_queries = self.drop_tokens(token_keys)
            elif self.args.augment_method == 'replace':
                token_queries = self.replace_tokens(token_keys)
            elif self.args.augment_method == 'add':
                token_queries = self.add_tokens(token_keys)
            elif self.args.augment_method == 'all':
                rand = random.random()
                if rand < 0.33:
                    token_queries = self.add_tokens(token_keys)
                elif rand < 0.66:
                    token_queries = self.replace_tokens(token_keys)
                else:
                    token_queries = self.drop_tokens(token_keys)
        return token_queries

    def creat_tokens_sample(self, tokens_list, max_tokens_num):
        input_ids= torch.tensor(
                    self.tokenizer.build_inputs_with_special_tokens(
                        tokens_list[:max_tokens_num]))
        attention_mask = torch.tensor([1] * len(input_ids))
        return input_ids, attention_mask


    def creat_vecs_sample(self, vector):
        return vector


    def drop_tokens(self, input_ids):
        drop_num = 1

        l = len(input_ids)
        final_tokens = random.sample(range(l),l-drop_num)
        temp = []
        for inx, t in enumerate(input_ids):
            if inx in final_tokens:
                temp.append(t)
        return temp

    def replace_tokens(self, input_ids):
        replace_num = 1

        raplace_inx = set()
        for i in range(replace_num):
            inx = np.random.randint(0, len(input_ids)-1)
            while inx in raplace_inx:
                inx = np.random.randint(0, len(input_ids) - 1)
            raplace_inx.add(inx)
            input_ids[inx] = np.random.randint(0, self.tokenizer.vocab_size - 1)
        return input_ids

    def add_tokens(self, input_ids):
        final_tokens = random.sample(range(self.tokenizer.vocab_size), 1)
        rand = random.random()
        if rand<0.5:
            final_tokens.extend(input_ids)
            return final_tokens
        else:
            input_ids.extend(final_tokens)
            return input_ids




# @dataclass
class SingleProcessDataLoaderForMatching:
    # dataset: IterableDataset
    # batch_size: int
    # collate_fn: Callable
    # drop_last: bool = True
    def __init__(self,
                 dataset: IterableDataset,
                 batch_size: int,
                 collate_fn: Callable,
                 drop_last: bool = True,
                 debug = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last  = drop_last
        self.debug = debug

    def set_end(self, value):
        self.end = value

    def if_end(self,):
        return self.end

    def skip_sample(self, i, sample):
        return False

    def _start(self):
        self.set_end(False)
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        for batch in self._generate_batch():
            if self.if_end():
                break
            self.outputs.put(batch)
            self.aval_count += 1
        self.pool.shutdown(wait=False)
        raise

    def _generate_batch(self):
        batch = []
        for i, sample in enumerate(self.dataset):
            if self.skip_sample(i, sample): continue
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.set_end(True)

    def __iter__(self):
        if self.debug:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        if self.aval_count == 0 and self.if_end():
            raise StopIteration
        next_batch = self.outputs.get()
        self.outputs.task_done()
        self.aval_count -= 1
        return next_batch


@dataclass
class MultiProcessDataLoaderForMatching:
    dataset: IterableDataset
    batch_size: int
    collate_fn: Callable
    local_rank: int
    world_size: int
    global_end: Any
    buffer_num:int
    blocking: bool=False
    drop_last: bool = True

    def _start(self):
        self.local_end=False
        self.global_end.value = False
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        try:
            for batch in self._generate_buffer_batch():
                self.outputs.put(batch)
                self.aval_count += 1
            self.pool.shutdown(wait=False)
        except:
            import sys
            import traceback
            import logging
            error_type, error_value, error_trace = sys.exc_info()
            traceback.print_tb(error_trace)
            logging.info(error_value)
            self.pool.shutdown(wait=False)
            raise

    def _generate_buffer_batch(self):
        """buffer batches, and shuffling insider batchs to produce next.
        """
        buffer = []
        for i, sample in enumerate(self.dataset):
            if i % self.world_size != self.local_rank: continue
            buffer.append(sample)
            if len(buffer) >= self.batch_size * self.buffer_num:
                buffer, next_batch = self.buffer_shuffle(buffer)
                yield self.collate_fn(next_batch)
        else:
            if len(buffer) > 0:
                while len(buffer) >= self.batch_size:
                    buffer, next_batch = self.buffer_shuffle(buffer)
                    yield self.collate_fn(next_batch)
                if len(buffer) > 0 and not self.drop_last:
                    yield self.collate_fn(buffer)
        self.local_end=True

    def buffer_shuffle(self, batchs):
        shuffle(batchs)
        return batchs[:-self.batch_size], batchs[-self.batch_size:]

    def __iter__(self):
        if self.blocking:
            return self._generate_buffer_batch()
        self._start()
        return self

    def __next__(self):
        dist.barrier()
        while self.aval_count == 0:
            if self.local_end or self.global_end.value:
                self.global_end.value=True
                break
        dist.barrier()
        if self.global_end.value:
            raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch


#
