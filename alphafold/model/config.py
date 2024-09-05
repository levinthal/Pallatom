"""Model config."""

import copy
import ml_collections

def model_config() -> ml_collections.ConfigDict:
    """Get the ConfigDict of the model."""   
    cfg = copy.deepcopy(CONFIG)
    return cfg

CONFIG = ml_collections.ConfigDict({
    'model': {
        'pallatom': {
            'max_relative_feature': 32,
            'single_channel': 256,
            'pair_channel': 128,
            'atom_channel': 128,
            'atompair_channel': 16,
            'token_channel': 256,
            'r_cannel': 3,
            'num_letters': 20,
            'plm_bins': 22,
            'plm_first_break': 0.0,
            'plm_last_break': 10.0,
            
            "r3_edmp":{
                "s_min": 0.001,
                "s_max": 50.0,
                "sigma_data": 16.0,
                "psigma_mean": -1.2,
                "psigma_std": 1.5,
                "rho": 7.0,
            },
            'atom_transformer':{
                'dropout_rate': 0.0,
                'gating': True,
                'num_head': 4,
                'orientation': 'per_row',
                'shared_dropout': True,
                'use_q_bias':True},
            
            'outer_product_mean': {
                'chunk_size': 128,
                'dropout_rate': 0.0,
                'num_outer_channel': 32,
                'orientation': 'per_row',
                'shared_dropout': True
            },            
            
            'row_attention_with_pair_bias': {
                    'dropout_rate': 0.15,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'use_q_bias':False,
                },
            'row_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
            'triangle_block':{
                'triangle_attention_starting_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'use_q_bias':False
                },
                'triangle_attention_ending_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_column',
                    'shared_dropout': True,
                    'use_q_bias':False
                },
                'triangle_multiplication_outgoing': {
                    'dropout_rate': 0.25,
                    'equation': 'ikc,jkc->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'fuse_projection_weights': True,
                },
                'triangle_multiplication_incoming': {
                    'dropout_rate': 0.25,
                    'equation': 'kjc,kic->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'fuse_projection_weights': True,
                },
                'pair_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                },
            },
            
            'template': {
                'pair_channel': 128,
                'attention': {
                    'gating': False,
                    'key_dim': 64,
                    'num_head': 4,
                    'value_dim': 64
                },
                'dgram_features': {
                    'min_bin': 3.25,
                    'max_bin': 50.75,
                    'num_bins': 39
                },
                'template_pair_stack': {
                    'num_block': 2,
                    'triangle_attention_starting_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'key_dim': 64,
                        'num_head': 4,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'value_dim': 64,
                        'use_q_bias':False
                    },
                    'triangle_attention_ending_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'key_dim': 64,
                        'num_head': 4,
                        'orientation': 'per_column',
                        'shared_dropout': True,
                        'value_dim': 64,
                        'use_q_bias':False
                    },
                    'triangle_multiplication_outgoing': {
                        'dropout_rate': 0.25,
                        'equation': 'ikc,jkc->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'fuse_projection_weights': False,
                    },
                    'triangle_multiplication_incoming': {
                        'dropout_rate': 0.25,
                        'equation': 'kjc,kic->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'fuse_projection_weights': False,
                    },
                    'pair_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 2,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    }
                },
                'max_templates': 1,
                'subbatch_size': 128,
                'use_template_unit_vector': False,
            },
            'mpnn': {
                'num_letters': 20, # out_pred
                'vocab': 20, # input_vocab_size
                'node_features': 128,
                'edge_features': 128,
                'hidden_dim': 128,
                'num_encoder_layers': 3,
                'num_decoder_layers': 3,
                'augment_eps': 0.0,
                'k_neighbors': 48,
                'dropout': 0.0
            },
            'seqhead':{
                'num_letters': 20
            }
        },
        
        'global_config': {
            'bfloat16': False,
            'bfloat16_output': False,
            'deterministic': False,
            'subbatch_size': 4,
            'use_remat': False,
            'zero_init': True,
            'eval_dropout': False,
        },
        
        'heads': {
            'distogram': {
                'first_break': 2.3125,
                'last_break': 21.6875,
                'num_bins': 64,
                'weight': 0.3
            },
            'structure_module': {
                'num_layer': 1,
                'fape': {
                    'clamp_distance': 10.0,
                    'clamp_type': 'relu',
                    'loss_unit_distance': 10.0
                },
                'angle_norm_weight': 0.01,
                'chi_weight': 0.5,
                'clash_overlap_tolerance': 1.5,
                'compute_in_graph_metrics': True,
                'dropout': 0.1,
                'num_channel': 384,
                'num_head': 12,
                'num_layer_in_transition': 3,
                'num_point_qk': 4,
                'num_point_v': 8,
                'num_scalar_qk': 16,
                'num_scalar_v': 16,
                'position_scale': 10.0,
                'sidechain': {
                    'atom_clamp_distance': 10.0,
                    'num_channel': 128,
                    'num_residual_block': 2,
                    'weight_frac': 0.5,
                    'length_scale': 10.,
                },
                'structural_violation_loss_weight': 1.0,
                'violation_tolerance_factor': 12.0,
                'weight': 0.0
            },
        },
    },
})
