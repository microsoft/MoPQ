# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, query_bert, key_bert=None, hidden_size=768, output_hidden_size=128):
        super().__init__()

        self.bert = query_bert
        self.last_linear = nn.Linear(hidden_size, output_hidden_size)
        self.last_f = nn.ELU()

        self.key_bert = None
        if key_bert is not None:
            self.key_bert = key_bert
            self.key_last_linear = nn.Linear(hidden_size, output_hidden_size)

    def retrieve_loss(self, q, k):
        score = torch.matmul(q, k.transpose(0, 1))
        labels = torch.arange(start=0, end=score.shape[0],
                    dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score,labels)
        return loss

    def pooling(self, ents, masks):
        valid_len = torch.sum(masks, dim=1, keepdim=True)
        new_masks = masks[..., None]
        masks_dim = new_masks.repeat(1, 1, ents.shape[-1])
        ents = ents*masks_dim
        pool_ents = torch.sum(ents, dim=1)
        averag_ents = pool_ents/valid_len
        return averag_ents

    def infer_q(
            self,
            input_id_query,attention_mask_query):
        outputs_query = self.bert(
            input_id_query,
            attention_mask=attention_mask_query,
        )[0]
        q = self.pooling(outputs_query, attention_mask_query)
        q = self.last_f(self.last_linear(q))
        return q

    def infer_k(self,
            input_id_key, attention_mask_key):
        if self.key_bert is not None:
            outputs_key = self.key_bert(
                input_id_key,
                attention_mask=attention_mask_key,
            )[0]
            k = self.pooling(outputs_key, attention_mask_key)
            k = self.last_f(self.key_last_linear(k))
        else:
            k = self.infer_q(input_id_key, attention_mask_key)
        return k

    def forward(self,
            input_id_query,attention_mask_query,
            input_id_key, attention_mask_key):
        q = self.infer_q(input_id_query, attention_mask_query)
        k = self.infer_k(input_id_key, attention_mask_key)

        if self.cross_device:
            q = self.gather_tensor(q)
            k = self.gather_tensor(k)

        return self.retrieve_loss(q, k)


