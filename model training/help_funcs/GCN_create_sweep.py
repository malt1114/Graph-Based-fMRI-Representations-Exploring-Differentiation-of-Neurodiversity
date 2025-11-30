import wandb
from help_funcs.data_func import get_node_features, load_dataset



# Initialize the sweep
sweep_config = {
    #Make gridsearch
    "method": "grid",
    #Set goal of the model
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "abs_edge_weights": {"values": [False, True]},
        "num_of_classes": {"value": None},
        "num_of_layers": {"value": 2},
        "random_seed": {"value": 42},
        "feature_set": {"value": ['var_bin', 'mean_bin']},
        "edge_feature_set": {"value": ['corr_var_bin_21','corr_mean_bin_21',
                                        'corr_var_bin_42','corr_mean_bin_42',
                                        'corr_var_bin_86','corr_mean_bin_86']},
        "num_of_features": {"value": None},
        "num_epochs": {"value": 500},
        "loss_func": {"value": None},
        "batch_size": {"values": [16, 32]},
        "optimizer": {"values": ["adam", 'sgd']},
        "learning_rate": {"values": [0.001, 0.002, 0.005, 0.01]},
        "hidden_channels_1": {"values": [4, 8, 16, 32, 64, 128]},
        "pool": {"values": ["global_mean_pool", "global_max_pool"]},
        "layer_norm": {"values": ["graph", "node"]},
        "edge_weight_thres": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.4]},
        "activation": {"values": ['relu', 'softmax']},
    }
}

if __name__ =="__main__":
    # ---- Create multi sweep ----
    
    #Set loss function + number of classes
    sweep_config['parameters']["loss_func"]['value'] = 'NLL_Loss'
    sweep_config['parameters']["num_of_classes"]['value'] = 4

    #Get number of features + print their names
    selected_feature = sweep_config['parameters']["feature_set"]['value']
    sweep_config['parameters']['num_of_features']['value'] = len(get_node_features(selected_feature))

    #Load data
    # train_data = load_dataset(dataset = 'train', 
    #                           num_of_classes = sweep_config['parameters']["num_of_classes"]['value'],
    #                           feature_names = selected_feature,
    #                           edge_names = sweep_config['parameters']["edge_feature_set"]['value'],
    #                           edge_w_thres = 0.0,
    #                           drop_strategy = None,
    #                           edge_w_abs = False,
    #                           GAT = False
    #                           )
    
    # val_data = load_dataset(dataset = 'val', 
    #                         num_of_classes = sweep_config['parameters']["num_of_classes"]['value'],
    #                         feature_names = selected_feature,
    #                         edge_names = sweep_config['parameters']["edge_feature_set"]['value'],
    #                         edge_w_thres = 0.0,
    #                         drop_strategy = None,
    #                         edge_w_abs = False,
    #                         GAT = False)

    #Make sweep
    sweep_id = wandb.sweep(sweep_config, project="GCN_Multi_Final")
    print("ID for multiclass:", sweep_id, flush=True)
    
    # ---- Create binary sweep ----
    

    # #Set loss function + number of classes
    # sweep_config['parameters']["loss_func"]['value'] = 'BCE'
    # sweep_config['parameters']["num_of_classes"]['value'] = 2

    # #Get number of features + print their names
    # selected_feature = sweep_config['parameters']["feature_set"]['value']
    # sweep_config['parameters']['num_of_features']['value'] = len(get_node_features(selected_feature))

    # #Load data
    # train_data = load_dataset(dataset = 'train', 
    #                           num_of_classes = sweep_config['parameters']["num_of_classes"]['value'],
    #                           feature_names = selected_feature,
    #                           edge_names = sweep_config['parameters']["edge_feature_set"]['value'],
    #                           edge_w_thres = 0.0,
    #                           drop_strategy = None,
    #                           edge_w_abs = False,
    #                           GAT = False
    #                           )
    
    # val_data = load_dataset(dataset = 'val', 
    #                         num_of_classes = sweep_config['parameters']["num_of_classes"]['value'],
    #                         feature_names = selected_feature,
    #                         edge_names = sweep_config['parameters']["edge_feature_set"]['value'],
    #                         edge_w_thres = 0.0,
    #                         drop_strategy = None,
    #                         edge_w_abs = False,
    #                         GAT = False
    #                         )

    # #Make sweep
    sweep_id = wandb.sweep(sweep_config, project="GCN_Binary_Final")
    print("ID for binary:", sweep_id, flush=True)
