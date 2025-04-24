from models import *

def config_PHGNN(num_features, num_classes, driving_forces=True):
    for h in [30,20,10]:
        for l in [30,20,10,5,1]:
            for t_end in [0.1,1,2,3,4]:
                for p_conv in ['naive','gcn']:
                    for q_conv in ['naive','gcn']:
                        for pq_readout in ['p','q','pq']:
                            for alpha in ([-1.0,0.0,1.0] if driving_forces else [0.0]):
                                for beta in ([-1.0,0.0,1.0] if driving_forces else [0.0]):
                                    for dampening_mode in (['param', 'param+', 'MLP4ReLU', 'DGNReLU'] if driving_forces else [None]):
                                        for external_mode in (['MLP4Sin', 'DGNtanh'] if driving_forces else [None]):
                                            yield {
                                                'model': {
                                                    'input_dim': num_features,
                                                    'output_dim': num_classes,
                                                    'hidden_dim': h,
                                                    'num_layers':l,
                                                    'epsilon': t_end/l,
                                                    'p_conv_mode': p_conv,
                                                    'q_conv_mode': q_conv,
                                                    'final_state': pq_readout,
                                                    'alpha': alpha,
                                                    'beta': beta,
                                                    'dampening_mode': dampening_mode,
                                                    'external_mode': external_mode,
                                                },
                                                'optim': {
                                                    'lr': 0.003,
                                                    'weight_decay': 1e-6
                                                }
                                            }


c1 = lambda num_features, num_classes: config_PHGNN(num_features, num_classes, driving_forces=False)
c2 = lambda num_features, num_classes: config_PHGNN(num_features, num_classes)

CONFIGS = {
    'PHGNN_conservative_GraphProp':(c1,PHDGN_GraphProp),
    'PHGNN_GraphProp':(c2,PHDGN_GraphProp),
}


