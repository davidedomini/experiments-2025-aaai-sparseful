from pathlib import Path
from experiments.federated import Simulator

if __name__ == '__main__':

    algorithms = ['fedavg'] #fedprox, scaffold
    data_folder = 'data/METR-LA'
    max_seed = 4
    results_folder = 'results'
    batch_size = 64
    local_epochs = 2
    global_rounds = 15
    horizon = 1
    window_size = 20

    data_output_directory = Path(results_folder)
    data_output_directory.mkdir(parents=True, exist_ok=True)

    for seed in range(max_seed):
        for algorithm in algorithms:
            sim = Simulator(algorithm, data_folder, seed, results_folder, batch_size, local_epochs, window_size, horizon)
            sim.start(global_rounds)