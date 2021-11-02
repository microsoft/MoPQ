# Matching-oriented Product Quantization For Ad-hoc Retrieval
Repo for EMNLP 2021 paper: Matching-oriented Product Quantization For Ad-hoc Retrieval.

## Introduction
In this work, we identify the limitation of using reconstruction loss minimization in supervised PQ methods, 
and propose MCL as the new training objective, where the model can be learned to maximize the query-key matching 
probability to achieve the optimal retrieval accuracy. We further leverage DCS for contrastive sample argumentation, which ensures the  effective minimization of MCL.  

## Dataset Format

File Name | Description | Format
------------- | ------------- | --------------
train.tsv/valid.tsv/test.tsv  | pairs of query and key | two columns and splited by "\t"
test_keys.josn  | keys' text and its id | {k_id : "key_text"}
test_queries.json  | queries' text and its id | {q_id : "query_text"}
test_ann.json  | queries' positive neighbors | {q_id : [k_id, ...]}


## Preprocess
- **Jointly optimize the codebook and embeddings**  
Here are the command to for tokenization:
```
python preprocess.py --dataset Mind  --bert_model bert-base-uncased
```
This command will create two files: `./data/MIND/preprocessed_train.tsv` and `./data/MIND/preprocessed_valid.tsv`, where each line is the tokenization results of query and key:
```
{"query_tokens":List[int], "key_tokens":List[int]}
{"query_tokens":List[int], "key_tokens":List[int]}
...
```

- **Optimize the codebooks based on the fixed embeddings**  
If you want to trian the MoPQ based on the existing embeddings, you should generate the preprocessed files in the following format:
```
{"query_vec":List[float], "key_vec":List[float]}
{"query_vec":List[float], "key_vec":List[float]}
...
```
You also can storage the "query_vec" and "key_tokens" in the preprocessed files to only fix queries' embeddings. 
Besides, you need to provide the embeddings of queries/keys for testdata, i.e., `test_queries.json`/`test_keys.json`
```
{"id":List[float], ...}
```
You can download the Mind_with_emb dataset from [here](https://microsoft-my.sharepoint.com/:f:/p/t-shxiao/EvQgMhZCoHdIp3PNSOF6re4BpNfxCxVJ3MappYxwpCN3RA?e=w2bh6J).

## Train
Use the following command to train MoPQ. And it will automatically select the best model to test.
```
python run.py \
  --mode train \
  --dataset Mind \
  --bert_model bert-base-uncased \
  --model_type {dataset: Mind or Mind_with_emb} \
  --savename MoPQ_Mind \
  --cross_device True \
  --world_size {the number of your GPUs}
```
If you use more than one GPU, `--cross_device` should be True to activate the Differentiable Cross-device in-batch Sampling.  
  
## Test
You can also start the test process manually using following command:
```
python run.py \
  --mode test \
  --dataset {dataset: Mind or Mind_with_emb} \
  --model_type MoPQ \
  --load_ckpt_name ./model/MoPQ_Mind-best.pt 
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
