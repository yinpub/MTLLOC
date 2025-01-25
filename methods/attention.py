import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class Muliple_Attention_Linear(torch.nn.Module):
    def __init__(self, n_tasks,feat_in):
        super(Muliple_Attention_Linear, self).__init__()
        self.n_tasks = n_tasks
        self.feat_in=feat_in
        for i in range(n_tasks):
            setattr(self,f'bias_{i}',nn.Parameter(torch.randn((1,feat_in)) ,requires_grad = True))
            setattr(self,f'linler_{i}',nn.Linear(feat_in, feat_in,bias=False))
        # self.drop_out = nn.Dropout()
    
    def forward(self, x):
        #x = torch.stack([self.drop_out(x + getattr(self,f'bias_{i}')) for i in range(self.n_tasks)],dim=0) 
        x = torch.stack([getattr(self,f'linler_{i}')(x) for i in range(self.n_tasks)],dim=0) 
        x = torch.softmax(x,dim=0)
        return [a for a in x]

class Causals_Attention_Linear(torch.nn.Module):
    def __init__(self, n_tasks,feat_in, feat_out ):
        super(Causals_Attention_Linear, self).__init__()
        self.n_tasks = n_tasks
        self.causal_CO = Muliple_Attention_Linear(2, feat_in) 
        self.causal_O = nn.Linear(feat_in, feat_out,bias=False)
        self.causal_C = nn.Linear(feat_in, feat_out,bias=False)
        self.bn_O = nn.BatchNorm1d(feat_out)
        self.bn_C = nn.BatchNorm1d(feat_out)
        
        self.causal_CS = Muliple_Attention_Linear(n_tasks, feat_out)
        
        # 多元因果干预算子 后续不会直接使用，转而使用它的权重
        self.causal_CS_mix = nn.Linear(n_tasks*feat_out, n_tasks*feat_out, bias=False) 
        index = torch.ones((n_tasks,n_tasks,feat_out,feat_out))
        index *= torch.eye(n_tasks,n_tasks).unsqueeze(-1).unsqueeze(-1) 
        index = index.permute(0,2,1,3).reshape((n_tasks*feat_out,n_tasks*feat_out))
        self.causal_CS_mix_index = index>0
        self.bn_CS_mix = nn.BatchNorm1d(n_tasks*feat_out)

        for i in range(n_tasks):
            setattr(self,f'linler_ci_{i}',nn.Linear(feat_out, feat_out,bias=False))
            setattr(self,f'bn_{i}',nn.BatchNorm1d(feat_out))

    def forward(self, x: Tensor):
        att_C,att_O = self.causal_CO(x)
        C_0,O_0 = x*att_C,x*att_O
        O = torch.relu(self.causal_O(O_0))
        C = torch.relu(self.causal_C(C_0))

        att_Cs = self.causal_CS(C)
        CS = []
        for i,att in enumerate(att_Cs):
            C_i = att*C
            C_i = getattr(self,f'linler_ci_{i}')(C_i)
            #C_i = getattr(self,f'bn_{i}')(C_i)
            C_i = torch.relu(C_i)
            CS.append(C_i)

        # -----------------Muliple Causals  addion ----------------------
        CS = torch.cat(CS,dim = -1)
        weight = torch.dropout(self.causal_CS_mix.weight,p=0.5,train=self.training)
        weight[self.causal_CS_mix_index] = 1.
        # bias = torch.dropout(self.causal_CS_mix.bias,p=0.5,train=self.training)
        bias = None
        CS_mix = F.linear(CS,weight=weight,bias=bias)
        CS = torch.relu(CS_mix) #self.bn_CS_mix(CS + torch.relu(CS_mix))
        
        # -----------------Muliple Causals  addion ----------------------
        CS = torch.chunk(CS,self.n_tasks,dim=-1)
        CO_S = [(CS[i].squeeze(-1) + O) for i in range(self.n_tasks)]#
        
        return CO_S
    


class Causals_Attention_patchs(torch.nn.Module):
    def __init__(self, n_tasks,feat_in, patchnum=64):
        super(Causals_Attention_patchs, self).__init__()

        self.dim=1
        self.emb=nn.Linear(feat_in,self.dim)
        
        self.n_tasks = n_tasks
        self.causal_CO = Muliple_Attention_Linear(2, patchnum) 
        # self.causal_O = nn.Linear(patchnum, self.dim,bias=False)
        # self.causal_C = nn.Linear(patchnum, self.dim,bias=False)
        # self.bn_O = nn.BatchNorm1d(self.dim)
        # self.bn_C = nn.BatchNorm1d(self.dim)

        # self.causal_CS = Muliple_Attention_Linear(n_tasks, self.dim)
        
        # # 多元因果干预算子 后续不会直接使用，转而使用它的权重
        # self.causal_CS_mix = nn.Linear(n_tasks*self.dim, n_tasks*self.dim, bias=False) 
        # index = torch.ones((n_tasks,n_tasks,self.dim,self.dim))
        # index *= torch.eye(n_tasks,n_tasks).unsqueeze(-1).unsqueeze(-1) 
        # index = index.permute(0,2,1,3).reshape((n_tasks*self.dim,n_tasks*self.dim))
        # self.causal_CS_mix_index = index>0
        # self.bn_CS_mix = nn.BatchNorm1d(n_tasks*self.dim)

        # for i in range(n_tasks):
        #     setattr(self,f'linler_ci_{i}',nn.Linear(self.dim, self.dim,bias=False))
        #     setattr(self,f'bn_{i}',nn.BatchNorm1d(self.dim))

        self.causal_O = nn.Linear(patchnum, patchnum,bias=False)
        self.causal_C = nn.Linear(patchnum, patchnum,bias=False)
        self.bn_O = nn.BatchNorm1d(patchnum)
        self.bn_C = nn.BatchNorm1d(patchnum)
        
        self.causal_CS = Muliple_Attention_Linear(n_tasks, patchnum)
        
        # 多元因果干预算子 后续不会直接使用，转而使用它的权重
        self.causal_CS_mix = nn.Linear(n_tasks*patchnum, n_tasks*patchnum, bias=False) 
        index = torch.ones((n_tasks,n_tasks,patchnum,patchnum))
        index *= torch.eye(n_tasks,n_tasks).unsqueeze(-1).unsqueeze(-1) 
        index = index.permute(0,2,1,3).reshape((n_tasks*patchnum,n_tasks*patchnum))
        self.causal_CS_mix_index = index>0
        self.bn_CS_mix = nn.BatchNorm1d(n_tasks*patchnum)

        for i in range(n_tasks):
            setattr(self,f'linler_ci_{i}',nn.Linear(patchnum, patchnum,bias=False))
            setattr(self,f'bn_{i}',nn.BatchNorm1d(patchnum))
        
        self.tem=None

    def forward(self, x: Tensor):
        
        x_=self.emb(x)       #  x_     batchsize,patch num,feature num   x:patches
        x_=x_.permute(0,2,1)
      
        att_C,att_O = self.causal_CO(x_)     
        C_0,O_0 = x_*att_C,x_*att_O
        O = torch.relu(self.causal_O(O_0))
        C = torch.relu(self.causal_C(C_0))

        att_Cs = self.causal_CS(C)
        CS = []
        for i,att in enumerate(att_Cs):
            C_i = att*C
            C_i = getattr(self,f'linler_ci_{i}')(C_i)
            #C_i = getattr(self,f'bn_{i}')(C_i)
            C_i = torch.relu(C_i)
            CS.append(C_i)

        # -----------------Muliple Causals  addion ----------------------
        CS = torch.cat(CS,dim = -1)
        weight = torch.dropout(self.causal_CS_mix.weight,p=0.5,train=self.training)     # 消融2 去掉，没有因果干预
        weight[self.causal_CS_mix_index] = 1.
        # bias = torch.dropout(self.causal_CS_mix.bias,p=0.5,train=self.training)
        bias = None
        CS_mix = F.linear(CS,weight=weight,bias=bias)
        CS = torch.relu(CS_mix) #self.bn_CS_mix(CS + torch.relu(CS_mix))
        CS = torch.dropout(CS,p=0.5,train=self.training) 
        # -----------------Muliple Causals  addion ----------------------
        CS = torch.cat(torch.chunk(CS,self.n_tasks,dim=-1),dim=1)
        CS = torch.softmax(CS,dim=1)
        CS = torch.chunk(CS,self.n_tasks,dim=1)
        #CO_S = [(CS[i].squeeze(-1) + O).permute(0,2,1)*x for i in range(self.n_tasks)]  #
       
        CO_S = []
        templist=[]
      
        for i in range(self.n_tasks):
            # 获取 CS[i]，并移除其最后一个维度（squeeze(-1)），
            # 然后加上 Os
            #temp = CS[i].squeeze(-1) + O   # temp (2,2,1)   CS[i].squeeze(-1) (2,1)    O (2,1,1)
            
            temp = CS[i] + O   # temp (2,2,1)   CS[i].squeeze(-1) (2,1)    O (2,1,1) # 消融1 去掉O，没有全局非因果信息
                                                               
            # 对 temp 进行 permute，改变维度顺序 (0, 2, 1)，
            # 这意味着将第 2 维和第 1 维交换位置
            temp = temp.permute(0, 2, 1)      
            templist.append(temp)

            # 将 temp 与 x 相乘
            temp = temp * x    # ()           (temp= 2,1,2    x=2,64,5184)
            # 将结果添加到 CO_S 列表中
            CO_S.append(temp)
        
        self.tem=templist
        return CO_S