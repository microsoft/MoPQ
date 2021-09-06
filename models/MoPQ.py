# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from models.Differentiable_PQ import *
import torch.distributed as dist

class MoPQ(DPQ):
    def __init__(self, args, query_bert, key_bert=None, hidden_size=768, local_rank=-1):
        super().__init__(args, query_bert, key_bert, hidden_size)
        self.cross_device = args.cross_device
        self.world_size = args.world_size
        self.local_rank = local_rank

    def forward(self,
                input_id_query, attention_mask_query,
                input_id_key, attention_mask_key):

        q, k, hard_k = self.infer(input_id_query, attention_mask_query,
              input_id_key, attention_mask_key)
        if self.cross_device:
            q = self.gather_tensor(q)
            k = self.gather_tensor(k)
            hard_k = self.gather_tensor(hard_k)

        loss = self.retrieve_loss(q, k)
        loss = loss + self.retrieve_loss(q, hard_k)
        return loss

    def infer(self, input_id_query, attention_mask_query,
              input_id_key, attention_mask_key):
        q = self.infer_q(input_id_query, attention_mask_query)
        k = self.infer_k(input_id_key, attention_mask_key)
        hard_k = self.quant(k)
        return q,k,hard_k

    def gather_tensor(self, vecs):
        all_tensors = [torch.empty_like(vecs) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, vecs)
        all_tensors[self.local_rank] = vecs
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors



