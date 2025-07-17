import copy
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from experiments.common import train_model, TimeSeriesDataset, evaluate_model
from experiments.federated.utils.FedUtils import initialize_model
from torch.utils.data import DataLoader


class FedAvgClient:

    def __init__(self, mid, train_data, validation_data, test_data, batch_size, epochs, window_size, horizon):
        self.mid = mid
        self.lr = 0.001
        self.epochs = epochs
        self.horizon = horizon
        self.weight_decay=1e-4
        self.batch_size = batch_size
        self.training_set = train_data
        self.window_size = window_size
        self.validation_set = validation_data
        self.test_set = test_data
        self.preprocess_data()
        self.device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = initialize_model().to(self.device)

        print(f'Client {self.mid} -- Training Data {len(self.training_set)}')
        print(f'Client {self.mid} -- Validation Data {len(self.validation_set)}')
        print(f'Client {self.mid} -- Test Data {len(self.test_set)}')
        print('----------------------------------------------------------------------------------------------------------------------')

    def train(self):
        train_dataloader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=True)
        train_losses, validation_losses, validation_r2s = train_model(self._model, train_dataloader, val_dataloader, self.epochs, self.lr, self.device)
        return sum(train_losses) / len(train_losses), sum(validation_losses) / len(validation_losses), sum(validation_r2s) / len(validation_r2s)

    def evaluate_model(self, validation = True):
        if validation:
            dataloader = DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
        loss, r2 = evaluate_model(self._model, dataloader)
        return loss, r2

    def notify_updates(self, global_model):
        self._model.load_state_dict(copy.deepcopy(global_model.state_dict()))

    @property
    def model(self):
        return self._model

    def preprocess_data(self):
        scaler = StandardScaler()
        scaler.fit(self.training_set.values)

        self.training_set = pd.DataFrame(scaler.transform(self.training_set.values),
                                       index=self.training_set.index,
                                       columns=self.training_set.columns)

        self.validation_set = pd.DataFrame(scaler.transform(self.validation_set.values),
                                     index=self.validation_set.index,
                                     columns=self.validation_set.columns)

        self.test_set  = pd.DataFrame(scaler.transform(self.test_set.values),
                                      index=self.test_set.index,
                                      columns=self.test_set.columns)

        self.training_set = TimeSeriesDataset(self.training_set, window_size=self.window_size, horizon=self.horizon)
        self.validation_set = TimeSeriesDataset(self.validation_set, window_size=self.window_size, horizon=self.horizon)
        self.test_set = TimeSeriesDataset(self.test_set, window_size=self.window_size, horizon=self.horizon)