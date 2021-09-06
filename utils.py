# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def warmup_linear(args, step):
    if step <= args.warmup_step:
        return max(step, 1) / args.warmup_step
    return max(1e-4, (args.schedule_step - step) /
               (args.schedule_step - args.warmup_step))


def dummy(*args, **kwargs):
    pass

def compute_acc(scores, labels):
    #hit num
    prediction = torch.argmax(scores, dim=-1)  # N L
    hit = (prediction == labels).float()  # N　L
    hit = torch.sum(hit)

    #all num
    labels = labels.masked_fill(labels >= 0, 1)
    labels = labels.masked_fill(labels < 0, 0)
    labels = torch.sum(labels)

    return hit, labels


def compute_retrive_acc(q, k, mask_q=None, mask_k=None):
    score = torch.matmul(q, k.transpose(0, 1))  #N N
    labels = torch.arange(start=0,
                          end=score.shape[0],
                          dtype=torch.long,
                          device=score.device)  #N
    if mask_q is not None and mask_k is not None:
        mask = mask_q * mask_k
    elif mask_q is not None:
        mask = mask_q
    elif mask_k is not None:
        mask = mask_k
    else:
        mask = None

    if mask is not None:
        score = score.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))  #N N
        labels = labels.masked_fill(mask == 0, -100)

    return compute_acc(score, labels)


def setuplogging(level=logging.INFO):
    # silent transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    if (root.hasHandlers()):
        root.handlers.clear()  # otherwise logging have multi output
    root.addHandler(handler)


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")


def save_model(model, optim, args,
               epoch=None, steps=None, name=None):
    if name is None:
        ckpt_path = os.path.join(args.model_dir,
                                 f'{args.savename}-{steps*args.train_batch_size}.pt')
    else:
        ckpt_path = os.path.join(args.model_dir,
                                 f'{args.savename}-{name}.pt')
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer': optim.state_dict()
        }, ckpt_path)
    logging.info(f"Model saved to {ckpt_path}")


def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def get_barrier(args):
    import torch.distributed as dist

    def nothing():
        pass

    if args.world_size > 1:
        return dist.barrier
    return nothing

