import torch
import random
import numpy as np
import pandas as pd
from experiments.federated.server.FedAvgServer import FedAvgServer
from experiments.federated.client.FedAvgClient import FedAvgClient

class Simulator:

    def __init__(self, algorithm, data_folder, seed, results_folder, batch_size, epochs, window_size, horizon):
        self.seed = seed
        self.epochs = epochs
        self.horizon = horizon
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.window_size = window_size
        self.data_folder = data_folder
        self.results_folder = results_folder

        self.simulation_data = pd.DataFrame(columns=['Round', 'TrainingLoss', 'ValidationLoss', 'ValidationR2']) # TODO add other metrics? MAE, MAPE
        self.train_data, self.val_data, self.test_data = self.load_data()
        self.clients = self.initialize_clients()
        self.server = self.initialize_server()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def start(self):
        raise Exception("Not implemented")

    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def initialize_clients(self):
        ids = list(self.train_data.columns)
        clients = {
            client_id :
            FedAvgClient(client_id, self.train_data[[client_id]], self.train_data[[client_id]], self.train_data[[client_id]], self.batch_size, self.epochs, self.window_size, self.horizon)
            for client_id in ids
        }
        return clients

    def initialize_server(self):
        return FedAvgServer()

    def evaluate_clients(self, validation = True):
        losses = []
        r2s = []
        for client in self.clients.values():
            loss, r2 = client.evaluate_model(validation=validation)
            losses.append(loss)
            r2s.append(r2)
        return sum(losses) / len(losses), sum(r2s) / len(r2s)

    def notify_clients(self):
        for client in self.clients.values():
            if self.algorithm == 'fedavg' or self.algorithm == 'fedprox':
                client.notify_updates(self.server.model)
            elif self.algorithm == 'scaffold':
                client.notify_updates(self.server.model, self.server.control_state)

    def notify_server(self):
        client_data = {}
        for client_id, client in self.clients.items():
            if self.algorithm == 'fedavg' or self.algorithm == 'fedprox':
                client_data[client_id] = client.model
            elif self.algorithm == 'scaffold':
                client_data[client_id] = {'model': client.model, 'client_control_state': client.client_control_state}
        self.server.receive_client_update(client_data)

    def load_data(self):
        train_data = pd.read_csv(f'{self.data_folder}/reduced_METR-LA-train.csv')
        validation_data = pd.read_csv(f'{self.data_folder}/reduced_METR-LA-val.csv')
        test_data = pd.read_csv(f'{self.data_folder}/reduced_METR-LA-test.csv')
        return train_data, validation_data, test_data
