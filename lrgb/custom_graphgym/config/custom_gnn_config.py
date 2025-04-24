from torch_geometric.graphgym.register import register_config


def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.weight_sharing = True
    cfg.gnn.doubled_dim = False #p,q splitting or doubling
    cfg.gnn.final_state = 'pq'
    cfg.gnn.t_end = 1.0 #integration time
    cfg.gnn.iterations = 10 #internal layers
    cfg.gnn.conv_p = 'naive'# conv mode for p part
    cfg.gnn.conv_q = 'naive'# conv mode for q part
    cfg.gnn.port_alpha = 0.0
    cfg.gnn.port_beta = 0.0
    cfg.gnn.port_dampening = ''
    cfg.gnn.port_external = ''
    cfg.gnn.l2_norm = False
    cfg.gnn.batchnorm = False

register_config('custom_gnn', custom_gnn_cfg)
