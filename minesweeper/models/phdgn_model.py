import torch
from models.phdgn_utils import PortHamiltonianConv
import torch
from typing import Optional
from torch.nn import Module, Linear, ModuleList, Sequential, Dropout, LayerNorm
from collections import OrderedDict


class PHDGN_Model(Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_dim,
                 num_layers,
                 epsilon,
                 activ_fun='tanh',
                 p_conv_mode: str = 'naive',
                 q_conv_mode: str = 'naive',
                 doubled_dim: bool = True,
                 final_state: str = 'pq',
                 alpha: float = 0.,
                 beta: float = 0.,
                 dampening_mode: Optional[str] = None,
                 external_mode : Optional[str] = None,
                 dtype=torch.float32,
                 node_level_task=False,
                 train_weights: bool = True, 
                 weight_sharing: bool = True,
                 bias: bool = True) -> None:
        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.activ_fun = activ_fun
        self.p_conv_mode = p_conv_mode
        self.q_conv_mode = q_conv_mode
        self.doubled_dim = doubled_dim
        self.final_state = final_state
        self.alpha = alpha
        self.beta = beta
        self.dampening_mode = dampening_mode
        self.external_mode = external_mode
        self.dtype = dtype
        self.node_level_task = node_level_task
        self.train_weights = train_weights
        self.weight_sharing = weight_sharing
        self.bias = bias

        self.emb = Linear(self.input_dim, self.hidden_dim)
        self.nhid = self.hidden_dim*2 if self.doubled_dim else self.hidden_dim

        self.convs = ModuleList()
        for _ in range(1 if self.weight_sharing else self.num_layers):
            self.convs.append(PortHamiltonianConv(
                in_channels=self.nhid,
                num_iters=self.num_layers if self.weight_sharing else 1, 
                epsilon=epsilon,
                activ_fun=activ_fun,
                p_conv_mode=p_conv_mode, 
                q_conv_mode=q_conv_mode,bias=bias, 
                beta=beta, 
                alpha=alpha, 
                dampening_mode=dampening_mode, 
                external_mode=external_mode, 
                dtype=dtype
            ))
            
        if self.final_state != 'pq':
            self.nhid = self.nhid // 2

        if not train_weights:
            #for param in self.enc.parameters():
            #    param.requires_grad = False
            for param in self.conv.parameters():
                param.requires_grad = False

        self.dropout = Dropout(p=0.2)
        self.norma = LayerNorm(self.nhid)
        self.readout = Sequential(
            Linear(self.nhid, self.nhid//2),
            torch.nn.GELU(),
            Linear(self.nhid//2, self.nhid//2),
            torch.nn.GELU(),
            Linear(self.nhid//2, output_dim)
        )

    def forward(self,data,edge_index) -> torch.Tensor:
        x = data.x

        h = self.emb(x) if self.emb else x
        h = self.dropout(h)
        h = torch.nn.GELU()(h)
        
        if self.doubled_dim:
            h = torch.cat([h,h],dim=1)

        for conv in self.convs:
            h = conv(h, edge_index)
        
        if self.final_state == 'p': # taking p
            h = h[:,:self.nhid]
        elif self.final_state == 'q': # taking q
            h = h[:,self.nhid:]
        else: # self.final_state == 'pq'
            pass # x contains both p and q already

        h = self.norma(h)  
        h = self.readout(h)
        return h
    

    
