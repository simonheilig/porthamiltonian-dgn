import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

#from custom_graphgym.layer.gatedgcn_layer import GatedGCNLayer
#from custom_graphgym.layer.gine_conv_layer import GINEConvLayer
#from custom_graphgym.layer.gcnii_conv_layer import GCN2ConvLayer
#from custom_graphgym.layer.mlp_layer import MLPLayer
from custom_graphgym.layer.phdgn_conv_layer import PortHamiltonianConv


class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."
        
        dim_in = 2*dim_in if cfg.gnn.doubled_dim else dim_in

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        self.model_type = cfg.gnn.layer_type
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            if cfg.gnn.weight_sharing:
                layers.append(conv_model(in_channels=dim_in,
                                        epsilon = cfg.gnn.t_end/cfg.gnn.iterations,
                                        p_conv_mode = cfg.gnn.conv_p,
                                        q_conv_mode = cfg.gnn.conv_q,
                                        num_iters = cfg.gnn.iterations,
                                        activ_fun = "tanh",
                                        alpha=cfg.gnn.port_alpha,
                                        beta = cfg.gnn.port_beta,
                                        external_mode=cfg.gnn.port_external,
                                        dampening_mode=cfg.gnn.port_dampening))
            else:
               for _ in range(cfg.gnn.iterations):
                    layers.append(conv_model(in_channels=dim_in,
                                            epsilon = cfg.gnn.t_end/cfg.gnn.iterations,
                                            p_conv_mode = cfg.gnn.conv_p,
                                            q_conv_mode = cfg.gnn.conv_q,
                                            num_iters = 1,
                                            activ_fun = "tanh",
                                            alpha=cfg.gnn.port_alpha,
                                            beta = cfg.gnn.port_beta,
                                            external_mode=cfg.gnn.port_external,
                                            dampening_mode=cfg.gnn.port_dampening)) 
                    
        self.gnn_layers = torch.nn.Sequential(*layers)

        self.GNNHead = register.head_dict[cfg.gnn.head]

        if cfg.gnn.final_state != 'pq':
            dim_in = dim_in // 2

        self.dim_in_dec = dim_in
        self.post_mp = self.GNNHead(dim_in=self.dim_in_dec, dim_out=dim_out)

    def build_conv_model(self, model_type):
        #if model_type == 'gatedgcnconv':
        #    return GatedGCNLayer
        #elif model_type == 'gineconv':
        #    return GINEConvLayer
        #if model_type == 'gcniiconv':
        #    return GCN2ConvLayer
        #elif model_type == 'mlp':
        #    return MLPLayer
        if model_type == "porthamiltonianconv":
            return PortHamiltonianConv
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            if type(module) == self.GNNHead:
                if cfg.gnn.final_state == 'p':
                    batch.x = batch.x[:,:self.dim_in_dec]
                elif cfg.gnn.final_state == 'q':
                    batch.x = batch.x[:,self.dim_in_dec:]
                #else final_state=='pq'
                    #pass since x = [p,q]

            if type(module) == torch.nn.Sequential and cfg.gnn.doubled_dim:
                batch.x = torch.cat([batch.x,batch.x],dim=1)

            if self.model_type == 'gcniiconv':
                batch.x0 = batch.x # gcniiconv needs x0 for each layer

            batch = module(batch)

        return batch


register_network('custom_gnn', CustomGNN)
