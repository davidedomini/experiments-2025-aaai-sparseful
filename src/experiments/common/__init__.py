import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_size: int, horizon: int = 1):
        """
        Args:
            data: DataFrame [time, sensors]
            window_size: input sequence length
            horizon: future steps to predict (default 1
        """
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + self.window_size + self.horizon - 1]
        return x, y


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # predici tutti i sensori

    def forward(self, x):
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        y_pred = self.fc(last_hidden)
        return y_pred


def train_model(model, train_dataloader, validation_dataloader, num_epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    validation_losses = []
    validation_r2s = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)
        val_loss, val_r2 = evaluate_model(model, validation_dataloader)
        scheduler.step()
        avg_loss = epoch_loss / len(train_dataloader.dataset)
        train_losses.append(avg_loss)
        validation_losses.append(val_loss)
        validation_r2s.append(val_r2)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    return train_losses, validation_losses, validation_r2s


def evaluate_model(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    avg_loss = total_loss / len(dataloader.dataset)

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()

    r2 = r2_score(y_true, y_pred)

    return avg_loss, r2
