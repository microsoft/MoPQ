# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utils.setup_utils import *
import json
from typing import Any, Callable, Dict, List, Tuple
from tqdm import tqdm
import heapq
from multiprocessing import Process,Manager,cpu_count,Pool
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_batch(texts, tokenizer, batch_size, max_l):
    tokens = []
    attention_mask = []
    vecs = []
    for i in range(len(texts)):
        content = texts[str(i)]
        if isinstance(content, str):
            temp = tokenizer(content, max_length=max_l, pad_to_max_length=True, truncation=True, add_special_tokens=False)
            tokens.append(temp['input_ids'])
            attention_mask.append(temp['attention_mask'])
        else:
            vecs.append(content)
        if len(tokens)==batch_size or len(vecs)==batch_size:
            if len(vecs) == 0:
                tokens = torch.LongTensor(tokens).cuda()
                attention_mask = torch.FloatTensor(attention_mask).cuda()
                vecs = None
            else:
                tokens = None
                attention_mask = None
                vecs = torch.FloatTensor(vecs).cuda()
            yield tokens,attention_mask, vecs
            tokens = []; attention_mask = []; vecs = []
    if len(tokens)>0 or len(vecs)>0:
        if len(vecs) == 0:
            tokens = torch.LongTensor(tokens).cuda()
            attention_mask = torch.FloatTensor(attention_mask).cuda()
            vecs = None
        else:
            tokens = None
            attention_mask = None
            vecs = torch.FloatTensor(vecs).cuda()
        yield tokens, attention_mask, vecs
        tokens = []; attention_mask = []; vecs = []


def encode(args, model=None, mode='test', model_type='MoPQ'):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    query_file = './data/{}/{}_queries.json'.format(args.dataset, mode)
    key_file = './data/{}/{}_keys.json'.format(args.dataset, mode)

    key_codes = []
    query_dtable = []
    batch_size = 2048
    with torch.no_grad():
        mode = 'soft' if model_type == 'spq' else 'ori'
        query = json.load(open(query_file))
        # stime = time.time()
        for data in tqdm(generate_batch(query, tokenizer, batch_size=batch_size, max_l=args.query_max_token),
                         total=(len(query) // batch_size)):
            input, mask, vecs = data
            dtable = model.encode(input=input, mask=mask, vecs=vecs, mode=mode)
            dtable = dtable.detach().cpu().numpy()
            query_dtable.extend(dtable)

        key = json.load(open(key_file))
        encode_time = 0
        for data in tqdm(generate_batch(key,tokenizer,batch_size=batch_size,max_l=args.key_max_token),
                         total=(len(key) // batch_size)):
            input, mask, vecs = data
            start_time = time.time()
            codes = model.encode(input=input, mask=mask, vecs=vecs, mode='hard')
            codes = codes.detach().cpu().numpy()
            key_codes.extend(codes)
            encode_time += time.time() - start_time
    return key_codes, query_dtable, encode_time



def get_result_by_dtable(dtable, codes, truth_nn, topk):
    res = [0]*len(topk)
    maxk = max(topk)

    for qi,table in enumerate(dtable):
        dists = np.sum(table[range(len(table)),codes],axis=-1)
        ans = heapq.nlargest(maxk, range(len(dists)), dists.__getitem__)

        for ki,k in enumerate(topk):
            ansk = set(ans[:k])
            res[ki] += len(set.intersection(truth_nn[qi], ansk)) / len(truth_nn[qi])
    res = np.array(res)
    return res

def recall_by_dtable(key_codes, query_dtable, topk, nn_file=None, enable_mulprocess=True, ):
    query_nn = json.load(open(nn_file))

    dtable = []
    truth_nn = []
    for k, v in query_nn.items():
        dtable.append(query_dtable[int(k)])
        truth_nn.append(set(v))

    if not enable_mulprocess:
        res = get_result_by_dtable(dtable, key_codes, truth_nn, topk)
        return res
    else:
        # mul process
        world_size = max(cpu_count() - 3,1)
        pool = Pool(processes=world_size)
        results = []
        per_process = len(truth_nn) // world_size
        for i in range(world_size):
            start = per_process * i
            end = min(len(truth_nn), per_process * (i + 1))
            per_dtable = dtable[start:end]
            per_truth_nn = truth_nn[start:end]

            result = pool.apply_async(get_result_by_dtable, args=(per_dtable, key_codes, per_truth_nn, topk))
            results.append(result)
        logging.info('Waiting for all subprocesses done...')
        pool.close()
        pool.join()
        logging.info('All subprocesses done.')

        res = np.array([0.0] * len(topk))
        for result in results:
            res += result.get()
        return res / len(truth_nn)


