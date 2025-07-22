import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.cm as cm
from matplotlib import pyplot as plt

def plot(data_centralized, data_oracle, metrics, chart_folder):
    viridis = cm.get_cmap('viridis', 2)
    colors = [viridis(0), viridis(1)]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(data_centralized[metric], label='Centralized', color=colors[0])
        plt.plot(data_oracle[metric], label='Oracle', color=colors[1])
        # plt.title(f'Confronto su {metric}')
        plt.xlabel('Time')
        plt.ylabel(metric)
        # if not 'R2' in metric:
        #     plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{chart_folder}/{metric}.pdf')


if __name__ == '__main__':
    

    chart_folder = 'charts/PEMS04'
    Path(chart_folder).mkdir(parents=True, exist_ok=True)

    data_centralized = pd.read_csv('results-PEMS04/clusters-all/algorithm_fedavg_seed0.csv')

    data_oracle = [pd.read_csv(f'results-PEMS04/clusters-{i}/algorithm_fedavg_seed0.csv') for i in range(1, 4)]
    data_oracle = pd.concat(data_oracle).groupby(level=0).mean()

    #data_oracle = pd.read_csv(f'results-PEMS04/clusters-2/algorithm_fedavg_seed0.csv')

    metrics = ['TrainingLoss', 'ValidationLoss', 'ValidationR2', 'ValidationMAE', 'ValidationMAPE']

    plot(data_centralized, data_oracle, metrics, chart_folder)