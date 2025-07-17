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
        self.results_path = f'{results_folder}/algorithm_{algorithm}_seed{seed}'

        self.simulation_data = pd.DataFrame(columns=['Round', 'TrainingLoss', 'ValidationLoss', 'ValidationR2']) # TODO add other metrics? MAE, MAPE
        self.train_data, self.val_data, self.test_data = self.load_data()
        self.clients = self.initialize_clients()
        self.server = self.initialize_server()
        self.device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def start(self, global_rounds):
        for r in range(global_rounds):
            print(f'Starting global round {r}')
            self.notify_clients()
            training_loss = self.clients_update()
            self.notify_server()
            self.server_update()
            self.notify_clients()
            validation_loss, validation_r2 = self.evaluate_clients()
            self.export_data(r, training_loss, validation_loss, validation_r2)
            print(f'Training loss: {training_loss} | Validation loss: {validation_loss} | Validation R2: {validation_r2}')
            print('----------------------------------------------------------------------------------------------------------------------')
        self.evaluate_clients(False)
        self.save_data()


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
            FedAvgClient(client_id, self.train_data[[client_id]], self.val_data[[client_id]], self.test_data[[client_id]], self.batch_size, self.epochs, self.window_size, self.horizon)
            for client_id in ids
        }
        return clients

    def initialize_server(self):
        if self.algorithm == 'fedavg':
            return FedAvgServer()
        else:
            raise Exception('Unknown algorithm! Please check :)')

    def evaluate_clients(self, validation = True):
        losses = []
        r2s = []
        for client in self.clients.values():
            loss, r2 = client.evaluate_model(validation=validation)
            losses.append(loss)
            r2s.append(r2)

        loss = sum(losses) / len(losses)
        r2 = sum(r2s) / len(r2s)

        if not validation:
            data = pd.DataFrame({'Loss': [loss], 'R2': [r2]})
            data.to_csv(f'{self.results_path}-test.csv', index=False)

        return loss, r2

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

    def filter_data(self, df):
        clusters = pd.read_csv(f'{self.data_folder}/clusters-zone1.csv')
        ids = [int(i) for i in set(clusters.values.ravel())]
        return df[ids]


    def load_data(self):

        data = pd.read_hdf(f'{self.data_folder}/METR-LA.h5')
        data.columns = data.columns.astype(int)

        n_rows = len(data)
        train_end = int(0.7 * n_rows)
        val_end = int(0.85 * n_rows)

        data = self.filter_data(data)

        train_data = data.iloc[:train_end].reset_index(drop=True)
        validation_data = data.iloc[train_end:val_end].reset_index(drop=True)
        test_data = data.iloc[val_end:].reset_index(drop=True)

        # train_data = pd.read_csv(f'{self.data_folder}/reduced_METR-LA-train.csv')
        # validation_data = pd.read_csv(f'{self.data_folder}/reduced_METR-LA-val.csv')
        # test_data = pd.read_csv(f'{self.data_folder}/reduced_METR-LA-test.csv')


        return train_data, validation_data, test_data

    def clients_update(self):
        training_losses = []
        for client_id, client in self.clients.items():
            loss, _, _ = client.train() # TODO maybe can delete this multiple return
            training_losses.append(loss)
        return sum(training_losses) / len(training_losses)

    def server_update(self):
        self.server.aggregate()

    def export_data(self, round, training_loss, validation_loss, validation_r2):
        self.simulation_data = self.simulation_data._append(
            {'Round': round, 'TrainingLoss': training_loss, 'ValidationLoss': validation_loss,
             'ValidationR2': validation_r2},
            ignore_index=True
        )

    def save_data(self):
        self.simulation_data.to_csv(f'{self.results_path}.csv', index=False)