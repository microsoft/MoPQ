# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--train_batch_size", type=int, default=500)
    parser.add_argument("--valid_batch_size", type=int, default=500)
    parser.add_argument("--test_batch_size", type=int, default=500)

    parser.add_argument("--model_dir", type=str, default='./model')  # path to save
    parser.add_argument("--enable_gpu", type=str2bool, default=True)

    parser.add_argument("--query_max_token", type=int, default=32)
    parser.add_argument("--key_max_token", type=int, default=50)

    parser.add_argument("--savename", type=str, default='model')
    parser.add_argument("--world_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=3000)
    parser.add_argument("--log_steps", type=int, default=10)

    parser.add_argument("--model_type", default="MoPQ", choices=['MoPQ', 'TextEncoder', 'DPQ', 'SPQ', 'DQN', 'DVSQ'],  type=str, required=True)
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default = 'model/savename-epoch-1.pt',
        help="choose which ckpt to load and test"
    )
    parser.add_argument("--ckpt_for_infer", default="bert-base-uncased", type=str)


    # lr schedule
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pretrain_lr", type=float, default=0.0001)

    # model
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--partition_num", type=int, default=8)
    parser.add_argument("--centroids_num", type=int, default=256)
    parser.add_argument("--output_hidden_size", type=int, default=128)
    parser.add_argument("--select_mode", type=str, default='l2', help='codeword selection function', choices=['ip','mlp','sub_mlp','l2','mix','mix_w','cosine','dpq'])
    parser.add_argument("--recall_method", type=str, default='ip')
    parser.add_argument("--load", type=str2bool, default=False)
    parser.add_argument("--start_test_epoch", type=int, default=5)
    parser.add_argument("--blocking", type=str2bool, default=True)
    parser.add_argument("--dataset", type=str, default='Mind')
    parser.add_argument("--early_stop_epoch", type=int, default=3)
    parser.add_argument("--fix_embedding", type=str2bool, default=False)
    parser.add_argument("--self_supervised", type=str2bool,
                        help='if use self-superived methods for unlabeled data, please load a pretrained TextEncoder and freeze it (fix_embedding=True) during training',
                        default=False)
    parser.add_argument("--augment_method", type=str, default='all',choices=['drop','delete','add','all'])
    parser.add_argument("--cross_device", type=str2bool, default=True)

    parser.add_argument("--input_query_vec", type=str2bool, default=False)
    parser.add_argument("--input_key_vec", type=str2bool, default=False)

    parser.add_argument("--quantization_loss", type=str,
        nargs='+',
        default=['l2'],
        choices=['dot', 'subdot', 'l2'])
    parser.add_argument("--quantization_loss_weight", type=float, default=1e-8)



    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
