from models import *                       

def config_PHGNN(num_features, num_classes):
    for hidden in [512,256]:
        for l in [8,5,1]:
            for e in [1e-1,1.0]:
                for p_conv in ['naive','gcn']:
                    for q_conv in ['naive','gcn']:
                        for pq_readout in ['q']:
                            for alpha in [1.0,0.0]:
                                    for beta in [1.0,0.0]:
                                        for dampening_mode in (['param', 'MLP4ReLU', 'DGNReLU'] if alpha !=0 else [None]):
                                            for external_mode in (['MLP4Sin', 'DGNtanh'] if beta !=0 else [None]):
                                                yield {
                                                    'model': {
                                                        'input_dim': num_features,
                                                        'output_dim': num_classes,
                                                        'hidden_dim': hidden,
                                                        'num_layers':l,
                                                        'epsilon': e,
                                                        'p_conv_mode': p_conv,
                                                        'q_conv_mode': q_conv,
                                                        'final_state': pq_readout,
                                                        'alpha':alpha,
                                                        'beta': beta,
                                                        'dampening_mode':dampening_mode,
                                                        'external_mode':external_mode,
                                                    },
                                                    'optim': {
                                                        'lr': 3e-5,
                                                        'wd': 0.0
                                                    }
                }                                                    


c1 = lambda num_features, num_classes: config_PHGNN(num_features, num_classes)

CONFIGS = {
    'PHGNN_grid':(c1,PHDGN_Model),
 }

