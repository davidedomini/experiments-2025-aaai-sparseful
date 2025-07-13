from ctypes import windll

from experiments.federated import Simulator

if __name__ == '__main__':

    algorithms = ['fedavg'] #fedprox, scaffold
    data_folder = 'data/METR-LA'
    max_seed = 1
    results_folder = 'results'
    batch_size = 64
    local_epochs = 2
    global_rounds = 10
    horizon = 1
    window_size = 20

    for seed in range(max_seed):
        for algorithm in algorithms:
            for round in range(global_rounds):
                sim = Simulator(algorithm, data_folder, seed, results_folder, batch_size, local_epochs)
                sim.start()