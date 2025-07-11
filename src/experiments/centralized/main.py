import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from experiments.common import TimeSeriesDataset, SimpleRNN, train_model, evaluate_model, get_device

if __name__ == "__main__":

    device = 'cpu'#get_device()

    WINDOW_SIZE = 30
    HORIZON = 1
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_EPOCHS = 20

    df_train = pd.read_csv('data/METR-LA/reduced_METR-LA-train.csv')
    df_val = pd.read_csv('data/METR-LA/reduced_METR-LA-val.csv')
    df_test = pd.read_csv('data/METR-LA/reduced_METR-LA-test.csv')

    scaler = StandardScaler()

    scaler.fit(df_train.values)

    df_train_scaled = pd.DataFrame(scaler.transform(df_train.values),
                               index=df_train.index,
                               columns=df_train.columns)

    df_val_scaled = pd.DataFrame(scaler.transform(df_val.values),
                                  index=df_val.index,
                                  columns=df_val.columns)

    df_test_scaled = pd.DataFrame(scaler.transform(df_test.values),
                              index=df_test.index,
                              columns=df_test.columns)

    dataset = TimeSeriesDataset(df_train_scaled, window_size=WINDOW_SIZE, horizon=HORIZON)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TimeSeriesDataset(df_val_scaled, window_size=WINDOW_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TimeSeriesDataset(df_test_scaled, window_size=WINDOW_SIZE, horizon=HORIZON)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleRNN(input_size=df_train_scaled.shape[1], hidden_size=HIDDEN_SIZE, num_layers=4)
    print(model)
    train_losses, val_losses, val_r2s = train_model(model, dataloader, val_dataloader, num_epochs=NUM_EPOCHS, device=device)

    print('-------------- TRAIN --------------')
    print(train_losses)
    print(val_losses)
    print(val_r2s)

    print('-------------- TEST --------------')
    test_loss, r2 = evaluate_model(model, test_loader, device=device)
    print(f'Test R2: {r2:.6f}')
    print(f"Test Loss (MSE) su sensori non visti: {test_loss:.6f}")

