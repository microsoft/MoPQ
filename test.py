# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import time
from argparse import Namespace
import torch

from utils.setup_utils import (setup_loader, setup_model)
from utils.utils import (compute_retrive_acc)
from utils.recall import recall_by_dtable, encode

def test_single_process(model: torch.nn.Module,
                        args: Namespace,
                        mode: str,
                        model_type:str):

    assert mode in {"valid", "test"}
    model.eval()

    with torch.no_grad():
        dataloader = setup_loader(args, mode=mode, multi_process=False)

        retrive_acc = [0, 0]
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                batch = {
                    k: v.cuda(non_blocking=True)
                    for k, v in batch.items() if v is not None
                }

            input_id_query = batch['input_id_query']
            attention_masks_query = batch['attention_masks_query']
            input_id_key = batch['input_id_key']
            attention_masks_key = batch['attention_masks_key']

            q = model.infer_q(
                 input_id_query, attention_masks_query)
            k = model.infer_k(
                 input_id_key, attention_masks_key)

            if model_type == 'spq':
                _, q, _ = model.get_soft_and_hard(q)
                _, _, k = model.get_soft_and_hard(k)
            elif model_type == 'encoder':
                pass
            else:
                k = model.quant(k)

            hit_num, all_num = compute_retrive_acc(q,k)
            retrive_acc[0] += hit_num.item()
            retrive_acc[1] += all_num.item()

        logging.info('Final-- qk_acc:{}'.format(retrive_acc[0] /
                                                retrive_acc[1]))
        return retrive_acc[0] / retrive_acc[1]


def test_recall(model: torch.nn.Module,
                args: Namespace,
                mode:str ='test',
                model_type: str ='MoPQ'):

    nn_file = './data/{}/{}_nn.json'.format(args.dataset, mode)
    topk = [1, 5, 10, 50, 100, 500]

    codes, dtable, encode_time = encode(args, model, mode=mode, model_type=model_type)
    logging.info('compute dtable costs:{}s---------------------------------------'.format(encode_time))
    stime = time.time()
    res = recall_by_dtable(codes,
                     dtable,
                     topk=topk,
                     enable_mulprocess=True,
                     nn_file=nn_file)
    search_time = time.time() - stime
    logging.info('search topk costs:{}s---------------------------------------'.format(search_time))
    logging.info('All Time:{}s [{} + {}]---------------------------------------'.format(encode_time+search_time, encode_time, search_time))
    return res

def test(args):
    device = torch.device("cuda")
    model = setup_model(args, local_rank=-1)
    model = model.to(device)

    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    res = test_recall(model, args, mode='test', model_type=args.model_type)
    logging.info("test_recall:{},{},{},{},{},{}".format(
        res[0], res[1], res[2], res[3],
        res[4], res[5]))