# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TextEncoder import TextEncoder

class DPQ(TextEncoder):
    '''
    This is the implementation for differentiable product quantization models (e.g., SPQ, DQN, DVSQ)
    '''
    def __init__(self, args, query_bert, key_bert=None, hidden_size=768):
        super().__init__(query_bert, key_bert, hidden_size, args.output_hidden_size)

        self.partition = args.partition_num
        self.centroids = args.centroids_num
        self.select_mode = args.select_mode
        self.output_hidden_size = args.output_hidden_size
        self.sub_dim = args.output_hidden_size//self.partition
        self.quantization_loss_weight = args.quantization_loss_weight
        self.quantization_loss_list = args.quantization_loss
        self.model_type = args.model_type
        self.cross_device = args.cross_device

        if self.select_mode == 'mlp':
            self.prob_w = nn.Linear(args.output_hidden_size,
                                         self.partition*self.centroids)
        elif self.select_mode == 'sub_mlp':
            self.prob_w2 = nn.Linear(args.output_hidden_size//self.partition,
                                         self.centroids)
        elif self.select_mode == 'mix_w':
            self.prob_w3 = nn.Linear(self.sub_dim*2, 1, bias=False)
        elif self.select_mode == 'dpq':
            self.matrix_k = nn.Parameter(torch.empty(self.partition, self.centroids, self.sub_dim).uniform_(-1, 1)).type(torch.FloatTensor) #P K D

        self.codebook = nn.Parameter(torch.empty(self.partition, self.centroids, self.sub_dim).uniform_(-1, 1)).type(torch.FloatTensor)  #nn.Parameter default requires_grad=True


    def select_codeword(self, vecs, softmax=True):
        if self.select_mode == 'mlp':
            vecs = F.relu(self.prob_w(vecs))
            assignment = vecs.view(vecs.size(0), self.partition, -1) # B P K

        elif self.select_mode == 'sub_mlp':
            vecs = vecs.view(vecs.size(0), self.partition, -1)  # B P D
            assignment = F.relu(self.prob_w2(vecs))  # B P K

        elif self.select_mode == 'dpq':
            #the selection is used by DPQ model
            batch_size = vecs.size(0)
            vecs = vecs.view(batch_size, self.partition, -1).unsqueeze(-1) #B P D 1
            reshape_k = self.matrix_k.unsqueeze(0).expand(batch_size, -1, -1, -1)  # B P K D
            assignment = torch.matmul(reshape_k, vecs).squeeze(-1)  # B P K

        elif self.select_mode == 'ip':
            vecs = vecs.view(vecs.size(0), self.partition, -1).unsqueeze(-1)  # B P D 1
            codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)  # B P K D
            assignment = torch.matmul(codebook, vecs).squeeze(-1)  # B P K

        elif self.select_mode == 'l2':
            vecs = vecs.view(vecs.size(0), self.partition, -1) #B P D
            codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)  # B P K D
            assignment = - torch.sum((vecs.unsqueeze(-2)-codebook)**2,-1) #B P K

        elif self.select_mode == 'mix_w':
            vecs = vecs.view(vecs.size(0), self.partition, -1)  # B P D
            codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)  # B P K D
            l2 = (vecs.unsqueeze(-2) - codebook) ** 2  # B P K D

            codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)  # B P K D
            ip = torch.mul(vecs.unsqueeze(-2), codebook)  #B P K D

            ip_l2 = torch.cat([ip,l2],-1) #B P K 2D
            assignment = self.prob_w3(ip_l2).squeeze(-1) #B P K

        elif self.select_mode == 'cosine':
            vecs = vecs.view(vecs.size(0), self.partition, -1)
            vecs = vecs.unsqueeze(-2).expand(-1,-1,self.centroids,-1) #B P K D
            codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)  # B P K D
            assignment = torch.cosine_similarity(vecs,codebook,dim=-1) #B P K

        else:
            raise NotImplementedError

        if softmax:
            assignment = F.softmax(assignment, -1)
        return assignment

    def soft_vecs(self, assignment):
        # self.codebook #P K D
        assignment = assignment.unsqueeze(2) #B P 1 K
        codebook = self.codebook.unsqueeze(0).expand(assignment.size(0),-1,-1,-1) #B P K D
        soft_vecs = torch.matmul(assignment,codebook).squeeze(2) #B P D
        soft_vecs = soft_vecs.view(assignment.size(0),-1) #B L
        return soft_vecs

    def STEstimator(self, assignment):
        index = assignment.max(dim=-1, keepdim=True)[1]
        assignment_hard = torch.zeros_like(assignment,device=assignment.device,dtype=assignment.dtype).scatter_(-1, index, 1.0)
        return assignment_hard.detach() - assignment.detach() + assignment

    def hard_vecs(self, assignment):
        assignment = self.STEstimator(assignment)   #B P K
        assignment = assignment.unsqueeze(2)  # B P 1 K
        codebook = self.codebook.unsqueeze(0).expand(assignment.size(0), -1, -1, -1)  # B P K D
        hard_vecs = torch.matmul(assignment, codebook).squeeze(2)  # B P D
        hard_vecs = hard_vecs.view(assignment.size(0), -1)  # B L
        return hard_vecs

    def get_soft_and_hard(self, vecs):
        prob = self.select_codeword(vecs)
        soft_vecs = self.soft_vecs(prob)
        hard_vecs = self.hard_vecs(prob)
        return prob,soft_vecs,hard_vecs

    def quant(self, vecs):
        assignment = self.select_codeword(vecs)
        hard_vecs = self.hard_vecs(assignment)
        return hard_vecs

    def sub_scores(self,q,k):
        batch_size = q.size(0)
        q = q.view(batch_size,self.partition,-1) #B P K
        k = k.view(batch_size,self.partition,-1) #B P K
        q = q.unsqueeze(1).expand(-1,batch_size,-1,-1) #B B P K
        k = k.unsqueeze(0).expand(batch_size,-1,-1,-1) #B B P K
        sub_score = torch.sum(torch.mul(q,k),dim=-1) #B B P
        return sub_score

    def JointCentralLoss(self,soft_vecs, hard_vecs):
        return torch.mean(torch.norm(soft_vecs-hard_vecs,dim=-1))

    def quantization_loss(self, q, k, hard_k):
        # 'dot', 'subdot', 'l2'
        sub_score = self.sub_scores(q, k)
        score = torch.sum(sub_score, dim=-1)

        sub_qscore = self.sub_scores(q, hard_k)
        qscore = torch.sum(sub_qscore, dim=-1)

        loss = 0.0
        if 'dot' in self.quantization_loss_list:
            loss = loss + self.quantization_loss_weight*torch.mean((score-qscore)**2)
        if 'subdot' in self.quantization_loss_list:
            loss = loss + self.quantization_loss_weight*torch.mean((sub_score-sub_qscore)**2)
        if 'l2' in self.quantization_loss_list:
            loss = loss + self.quantization_loss_weight*torch.mean(torch.sum((k-hard_k)**2,dim=-1))
        return loss

    def encode(self, input=None, mask=None, mode='hard', vecs=None):
        if mode == 'hard':
            if vecs is None: vecs = self.infer_k(input, mask)
            prob = self.select_codeword(vecs)
            codes = prob.max(dim=-1, keepdim=True)[1].squeeze(-1)
            return codes
        else:
            if vecs is None: vecs = self.infer_q(input, mask)
            if mode == 'soft': # This mode is used by SPQ
                prob = self.select_codeword(vecs)
                vecs = self.soft_vecs(prob)

            vecs_reshape = vecs.view(vecs.size(0), self.partition, -1)  # B P D
            codebook = self.codebook.unsqueeze(0).expand(vecs_reshape.size(0), -1, -1, -1)  # B P K D
            dtable = torch.matmul(codebook, vecs_reshape.unsqueeze(-1)).squeeze(-1)  # B P K
            return dtable

    def forward(self,
                input_id_query, attention_mask_query,
                input_id_key, attention_mask_key):
        q = self.infer_q(input_id_query, attention_mask_query)
        k = self.infer_k(input_id_key, attention_mask_key)

        if self.model_type == 'spq':
            batch = input_id_query.size(0)
            vecs = torch.cat([q,k],0)
            prob, soft_vecs, hard_vecs = self.get_soft_and_hard(vecs)

            loss = self.retrieve_loss(soft_vecs[:batch], soft_vecs[batch:])
            loss = loss + self.retrieve_loss(soft_vecs[:batch], hard_vecs[batch:])
            loss = loss + self.quantization_loss_weight*self.JointCentralLoss(soft_vecs, hard_vecs)
            return loss

        else:
            loss = self.retrieve_loss(q,k)
            hard_k = self.quant(k)
            loss = loss + self.quantization_loss(q, k, hard_k)

        return loss

