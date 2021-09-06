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
                 tokenizer: BertTokenizerFast,
                 file_path: str,
                 overwrite_cache=False,
                 tokenizing_batch_size=65536):
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_{}_{}".format(
                tokenizer.__class__.__name__,
                filename,
            ),
        )
        setuplogging()
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        # Input file format:
        # One query and one key per line, split by '\t'.
        with FileLock(lock_path):
            if os.path.exists(cached_features_file +
                              ".finish") and not overwrite_cache:
                self.data_file = open(cached_features_file,
                                      "r",
                                      encoding="utf-8")
            else:
                logger.info(
                    f"Creating features from dataset file at {directory}")
                batch_query, batch_key = [], []
                with open(file_path, encoding="utf-8") as f, open(
                        cached_features_file, "w", encoding="utf-8") as fout:
                    for lineNum, line in tqdm(enumerate(f)):
                        line = line.strip()
                        if not line: continue
                        try:
                            query_and_nn, key_and_nn = line.strip('\n').split(
                                '\t')
                        except ValueError:
                            logger.error("line {}: {}".format(
                                lineNum, line.replace("\t", "|")))
                            continue
                        batch_query.append(query_and_nn.strip())
                        batch_key.append(key_and_nn.strip())

                        if len(batch_query) >= tokenizing_batch_size:
                            tokenized_result_query = tokenizer.batch_encode_plus(
                                batch_query, add_special_tokens=False)
                            tokenized_result_key = tokenizer.batch_encode_plus(
                                batch_key, add_special_tokens=False)
                            for j, (tokens_query, tokens_key) in enumerate(
                                    zip(tokenized_result_query['input_ids'],
                                        tokenized_result_key['input_ids'])):
                                fout.write(json.dumps([tokens_query, tokens_key]) + '\n')
                            batch_query, batch_key = [], []

                    if len(batch_query) > 0:
                        tokenized_result_query = tokenizer.batch_encode_plus(
                            batch_query, add_special_tokens=False)
                        tokenized_result_key = tokenizer.batch_encode_plus(
                            batch_key, add_special_tokens=False)
                        for j, (tokens_query, tokens_key) in enumerate(
                                zip(tokenized_result_query['input_ids'],
                                    tokenized_result_key['input_ids'])):
                            fout.write(json.dumps([tokens_query, tokens_key]) + '\n')
                        batch_query, batch_key = [], []

                    logger.info(f"Finish creating")
                with open(cached_features_file + ".finish",
                          "w",
                          encoding="utf-8"):
                    pass
                self.data_file = open(cached_features_file,
                                      "r",
                                      encoding="utf-8")
            # os.remove(cached_features_file + ".finish")
    def __iter__(self):
        for line in self.data_file:
            tokens_title = json.loads(line)
            yield tokens_title



# @dataclass
class DataCollatorForMatching:
    def __init__(self, tokenizer: BertTokenizerFast,
                 args: Namespace):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(
            self,
            samples: List[List[List[int]]]) -> Dict[str, torch.Tensor]:
        input_id_queries = []
        attention_mask_queries = []
        input_id_keys = []
        attention_mask_keys = []
        for i, sample in (enumerate(samples)):
            (input_id_query,
             attention_mask_query,
             input_id_key,
             attention_mask_key) = self.create_training_sample(sample)

            input_id_queries.append(input_id_query)
            attention_mask_queries.append(attention_mask_query)
            input_id_keys.append(input_id_key)
            attention_mask_keys.append(attention_mask_key)


        input_id_queries = self._tensorize_batch(
            input_id_queries, self.tokenizer.pad_token_id)
        input_id_keys = self._tensorize_batch(input_id_keys,
                                              self.tokenizer.pad_token_id)
        return {
            "input_id_query":
            input_id_queries,
            "attention_masks_query":
            self._tensorize_batch(attention_mask_queries, 0),
            "input_id_key":
            input_id_keys,
            "attention_masks_key":
            self._tensorize_batch(attention_mask_keys, 0)
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

    def create_training_sample(self, sample: List[List[int]]):
        """Turn sample into train pair

        Args:
            sample (List[List[int]]): [query_seq_id, key_seq_id]

        Returns:
            input_id_queries: query_seq_id wth CLS END
        """

        max_query_token = self.args.query_max_token
        max_key_token = self.args.key_max_token

        token_queries, token_keys = sample

        if self.args.self_supervised:
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


        input_id_queries = torch.tensor(
                    self.tokenizer.build_inputs_with_special_tokens(
                        token_queries[:max_query_token]))
        input_id_keys    = torch.tensor(
                    self.tokenizer.build_inputs_with_special_tokens(
                        token_keys[:max_key_token]))

        attention_mask_queries = torch.tensor([1] * len(input_id_queries))
        attention_mask_keys = torch.tensor([1] * len(input_id_keys))

        return input_id_queries, attention_mask_queries, input_id_keys, attention_mask_keys


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
