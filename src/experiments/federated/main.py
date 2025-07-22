import os
import sys
import yaml
from pathlib import Path
from experiments.federated import Simulator

def get_hyperparameters():
    """
    Fetches the hyperparameters from the docker compose config file
    :return: the experiment name and the hyperparameters (as a dictionary name -> values)
    """
    hyperparams = os.environ['LEARNING_HYPERPARAMETERS']
    hyperparams = yaml.safe_load(hyperparams)
    experiment_name, hyperparams = list(hyperparams.items())[0]
    return experiment_name.lower(), hyperparams

if __name__ == '__main__':

    algorithms = ['fedavg']
    # data_folder = 'data/METR-LA'
    data_folder = 'data/PEMS04'
    dataset_name = 'PEMS04.csv'
    max_seed = 1
    results_folder = 'results-PEMS04'
    batch_size = 64
    local_epochs = 2
    global_rounds = 15
    horizon = 1
    window_size = 20
 
    data_output_directory = Path(results_folder)
    data_output_directory.mkdir(parents=True, exist_ok=True)

    # experiment_name, hyperparams = get_hyperparameters()
    # clusters_subset  = hyperparams['cluster'][0]
    # if clusters_subset == 0:
    #     clusters_subset = 'all'
    
    clusters_subsets = ['all', 1, 2, 3]

    # clusters_subset = 1
    for seed in range(max_seed):
        for cluster_subset in clusters_subsets:
            for algorithm in algorithms:
                print(f'Starting simulation for algorithm {algorithm}, seed {seed}, clusters {cluster_subset}')
                Path(f'{results_folder}/clusters-{cluster_subset}').mkdir(parents=True, exist_ok=True)
                sim = Simulator(algorithm, data_folder, dataset_name, seed, results_folder, batch_size, local_epochs, window_size, horizon, cluster_subset)
                sim.start(global_rounds)