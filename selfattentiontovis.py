# from math import sqrt
 
# import torch
# import torch.nn as nn
 
# class SelfAttention(nn.Module):
#     def __init__(self, dim_q=2432, dim_k=1, dim_v=1):
#         super(SelfAttention, self).__init__()
#         self.dim_q = dim_q
#         self.dim_k = dim_k
#         self.dim_v = dim_v
 
#         #定义线性变换函数
#         self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
#         self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
#         self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
#         self._norm_fact = 1 / sqrt(dim_k)
 
#     def forward(self, x):
#         # x: batch, n, dim_q
#         #根据文本获得相应的维度
 
#         batch, n, dim_q = x.shape
#         assert dim_q == self.dim_q
 
#         q = self.linear_q(x)  # batch, n, dim_k
#         k = self.linear_k(x)  # batch, n, dim_k
#         v = self.linear_v(x)  # batch, n, dim_v
#         #q*k的转置 并*开根号后的dk
#         dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
#         #归一化获得attention的相关系数
#         dist = torch.softmax(dist, dim=-1)  # batch, n, n
#         #attention系数和v相乘，获得最终的得分
#         att = torch.bmm(dist, v)
#         return att
from math import sqrt
 
import torch
import torch.nn as nn
 
class SelfAttention(nn.Module):
    def __init__(self, dim_q=4, dim_k=64, dim_v=1):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
 
        #定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)
 
    def forward(self, x):
        # x: batch, n, dim_q
        #根据文本获得相应的维度
 
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q
 
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        #q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        #归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        #attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        return att
