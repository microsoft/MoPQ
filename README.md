# Matching-oriented Product Quantization For Ad-hoc Retrieval
Repo for EMNLP 2021 paper: Matching-oriented Product Quantization For Ad-hoc Retrieval.

## Introduction
In this work, we identify the limitation of using reconstruction loss minimization in supervised PQ methods, 
and propose MCL as the new training objective, where the model can be learned to maximize the query-key matching 
probability to achieve the optimal retrieval accuracy. We further leverage DCS for contrastive sample argumentation, which ensures the  effective minimization of MCL.  


## Train
Use the following command to train on the Mind dataset. And it will automatically select the best model to test.
```
python run.py --mode train \
--dataset Mind \
--model_type MoPQ --savename MoPQ_Mind \
--cross_device True --world_size 8 
```

## Test
You can also start the test process manually using following command:
```
python run.py --mode test \
--dataset Mind \
--model_type MoPQ --load_ckpt_name ./model/MoPQ_Mind-best.pt 
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
