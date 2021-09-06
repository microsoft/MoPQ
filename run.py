# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import logging
import os
import time
from pathlib import Path
import sys, traceback
from argparse import Namespace
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from parameters import parse_args
from utils.setup_utils import (cleanup, setup_worker, setup_loader, setup_manager, setup_model)
from utils.utils import (compute_retrive_acc, dump_args,
                    get_barrier, save_model, setuplogging)

from test import test_single_process, test, test_recall

def main():
    setuplogging()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()

    args.train_data_path = './data/{}/train.tsv'.format(args.dataset)
    args.valid_data_path = './data/{}/valid.tsv'.format(args.dataset)
    args.test_data_path = './data/{}/test.tsv'.format(args.dataset)
    dump_args(args)

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if 'train' in args.mode:
        mgr, end, end_train = setup_manager()
        mp.spawn(train,
                 args=(args, end, end_train),
                 nprocs=args.world_size,
                 join=True)

    elif 'test' in args.mode:
        test(args)

def train(local_rank:int,
          args: Namespace,
          end_dataloder,
          end_train
          ):
    '''
    Args:
        local_rank: the rank of GPU
        args: Namespace
        end_dataloader: mp.Manager().Value(), exit dataloader
        end_train: mp.Manager().Value(), exit training
    '''
    setuplogging()
    try:
        if args.world_size > 1:
            os.environ["RANK"] = str(local_rank)
            setup_worker(local_rank, args.world_size)
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu',
                              local_rank)
        barrier = get_barrier(args)

        model = setup_model(args, local_rank)
        model = model.to(device)

        #if use self-superived methods for unlabeled data, please load a pretrained TextEncoder and freeze it during training
        if args.load:
            checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            logging.info('load ckpt:{}'.format(args.load_ckpt_name))
        if args.fix_embedding:
            model.bert = model.bert.requires_grad_(False)
            model.last_linear = model.last_linear.requires_grad_(False)

        if args.world_size > 1:
            ddp_model = DDP(model,
                            device_ids=[local_rank],
                            output_device=local_rank,
                            find_unused_parameters=True)
            logging.info("[{}] Done copying model".format(local_rank))
        else:
            logging.info("Not using ddp model")
            ddp_model = model

        rest_param = filter(
            lambda x: id(x) not in list(map(id, model.bert.parameters())),
            ddp_model.parameters())

        optimizer = optim.Adam([{
            'params': model.bert.parameters(),
            'lr': args.pretrain_lr
        }, {
            'params': rest_param,
            'lr': args.lr}])

        loss = 0.0
        global_step = 0
        best_acc, best_count = 0.0, 0
        best_ep = 0
        best_model = copy.deepcopy(model)

        for ep in range(args.epochs):
            start_time = time.time()
            dataloader = setup_loader(args,
                                      local_rank=local_rank,
                                      end=end_dataloder,
                                      mode='train',
                                      buffer_batchs=50,
                                      blocking=False)

            ddp_model.train()
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

                batch_loss = ddp_model(input_id_query, attention_masks_query,
                                       input_id_key, attention_masks_key)

                loss += batch_loss.item()
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                global_step += 1

                if global_step % args.log_steps == 0:
                    logging.info('[{}] step:{}, train_loss: {:.5f}'.
                        format(local_rank,
                            global_step * args.train_batch_size,
                            loss / args.log_steps))
                    loss = 0.0

            logging.info("train time:{}".format(time.time() - start_time))

            if local_rank == 0 and ep >= args.start_test_epoch:
                logging.info("Star validation for epoch-{}".format(ep + 1))
                acc = test_single_process(model, args, "valid", args.model_type)
                logging.info("validation time:{}".format(time.time() - start_time))
                if acc > best_acc:
                    best_model = copy.deepcopy(model)
                    best_acc = acc
                    best_count = 0
                    save_model(best_model, optimizer, args, name='best')
                    best_ep = ep + 1
                else:
                    best_count += 1
                    if best_count >= args.early_stop_epoch:
                        start_time = time.time()
                        save_model(best_model, optimizer, args, name='best')
                        logging.info('Best_vali_ACC-- soft/ori-hard_qk_acc:{}'.format(best_acc))
                        logging.info("Star testing for best")
                        res = test_recall(best_model, args, mode='test', model_type=args.model_type)
                        logging.info("best epoch:{}".format(best_ep))
                        logging.info("test_recall:{},{},{},{},{},{}".format(
                            res[0], res[1], res[2], res[3],
                            res[4], res[5]))
                        logging.info("test time:{}".format(time.time() -
                                                           start_time))

                        end_train.value = True
            barrier()
            if end_train.value:
                break

        if local_rank == 0 and not end_train.value:
            start_time = time.time()
            save_model(best_model, optimizer, args, name='best')
            logging.info('Best_vali_ACC-- soft/ori-hard_qk_acc:{}'.format(best_acc))
            logging.info("Star testing for best")
            res = test_recall(model, args, mode='test', model_type=args.model_type)
            logging.info("best epoch:{}".format(best_ep))
            logging.info("test_recall:{},{},{},{},{},{}".format(
                res[0], res[1], res[2], res[3],
                res[4], res[5]))
            logging.info("test time:{}".format(time.time() -
                                               start_time))

        cleanup()
    except:
        error_type, error_value, error_trace = sys.exc_info()
        traceback.print_tb(error_trace)
        logging.info(error_value)


if __name__ == "__main__":
    main()
