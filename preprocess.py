import os
import logging
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from utils.setup_utils import setup_model
from parameters import parse_args
from utils.utils import setuplogging


def preprocess(args, tokenizing_batch_size=2048):
    tokenizer =  AutoTokenizer.from_pretrained(args.bert_model)
    args.train_data_path = './data/{}/train.tsv'.format(args.dataset)
    args.valid_data_path = './data/{}/valid.tsv'.format(args.dataset)
    args.test_data_path = './data/{}/test.tsv'.format(args.dataset)
    for file_path in [args.train_data_path, args.valid_data_path]:
        logging.info(f"processing {file_path}")

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "preprocessed_{}".format(filename),
        )

        model_for_infer = None
        if args.input_query_vec or args.input_key_veec:
            model_for_infer = setup_model(args, model_type='TextEncoder')
            model_for_infer.load_state_dict(torch.load(args.ckpt_for_infer)['model_state_dict'])
            model_for_infer.cuda()

        batch_query, batch_key = [], []
        with open(file_path, encoding="utf-8") as f, open(
                cached_features_file, "w", encoding="utf-8") as fout:
            for lineNum, line in tqdm(enumerate(f)):
                line = line.strip()
                if not line: continue
                try:
                    query, key = line.strip('\n').split( '\t')
                except ValueError:
                    logging.error("line {}: {}".format(
                        lineNum, line.replace("\t", "|")))
                    continue
                batch_query.append(query.strip())
                batch_key.append(key.strip())

                if len(batch_query) >= tokenizing_batch_size:
                    input_ids_query, input_ids_key, query_vecs, key_vecs = tokenize_and_infer(batch_query,
                                                                                              batch_key,
                                                                                              input_query_vec=args.input_query_vec,
                                                                                              input_key_vec=args.input_key_vec,
                                                                                              max_query_length=args.query_max_token,
                                                                                              max_key_length=args.key_max_token,
                                                                                              tokenizer=tokenizer,
                                                                                              model=model_for_infer)
                    save_data(input_ids_query, input_ids_key, query_vecs, key_vecs, output_f=fout)
                    batch_query, batch_key = [], []

            if len(batch_query) > 0:
                input_ids_query, input_ids_key, query_vecs, key_vecs = tokenize_and_infer(batch_query,
                                                                                          batch_key,
                                                                                          input_query_vec=args.input_query_vec,
                                                                                          input_key_vec=args.input_key_vec,
                                                                                          max_query_length=args.query_max_token,
                                                                                          max_key_length=args.key_max_token,
                                                                                          tokenizer=tokenizer,
                                                                                          model=model_for_infer)
                save_data(input_ids_query, input_ids_key, query_vecs, key_vecs, output_f=fout)
                batch_query, batch_key = [], []

            logging.info(f"Finish creating")



def tokenize_and_infer(batch_query, batch_key, input_query_vec, input_key_vec, max_query_length, max_key_length, tokenizer, model=None):
    input_ids_query, input_ids_key, q_vecs, k_vecs = None, None, None, None
    if input_query_vec:
        tokenized_result_query = tokenizer.batch_encode_plus(
            batch_query, add_special_tokens=True, padding=True, truncation=True, max_length=max_query_length)
        input_ids = torch.LongTensor(tokenized_result_query['input_ids']).cuda()
        attention_mask = torch.FloatTensor(tokenized_result_query['attention_mask']).cuda()
        q_vecs = model.infer_q(input_ids, attention_mask).detach().cpu().numpy()
    else:
        tokenized_result_query = tokenizer.batch_encode_plus(
            batch_query, add_special_tokens=False)
        input_ids_query = tokenized_result_query['input_ids']

    if input_key_vec:
        tokenized_result_key = tokenizer.batch_encode_plus(
            batch_key, add_special_tokens=True, padding=True, truncation=True, max_length=max_key_length)
        input_ids = torch.LongTensor(tokenized_result_key['input_ids']).cuda()
        attention_mask = torch.FloatTensor(tokenized_result_key['attention_mask']).cuda()
        k_vecs = model.infer_k(input_ids, attention_mask).detach().cpu().numpy()
    else:
        tokenized_result_key = tokenizer.batch_encode_plus(
            batch_key, add_special_tokens=False)
        input_ids_key =  tokenized_result_key['input_ids']

    return input_ids_query, input_ids_key, q_vecs, k_vecs

def save_data(input_ids_query, input_ids_key, query_vecs, key_vecs, output_f):
    if query_vecs is not None:
        query_name = 'query_vec'
        query_data = query_vecs
    else:
        query_name = 'query_tokens'
        query_data = input_ids_query

    if key_vecs is not None:
        key_name = 'key_vec'
        key_data = key_vecs
    else:
        key_name = 'key_tokens'
        key_data = input_ids_key

    for j, (q, k) in enumerate(
            zip(query_data, key_data)):
        output_f.write(json.dumps({query_name:[float(x) for x in q], key_name:[float(x) for x in k]}) + '\n')


if __name__ == '__main__':
    setuplogging()
    args = parse_args()
    preprocess(args, tokenizing_batch_size=2048)