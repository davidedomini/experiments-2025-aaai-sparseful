import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from common import TimeSeriesDataset, SimpleRNN, train_model, evaluate_model

if __name__ == "__main__":

    WINDOW_SIZE = 30
    HORIZON = 1
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_EPOCHS = 20

    df = pd.read_csv('data/METR-LA/reduced_METR-LA.csv')
    df_test = pd.read_csv('data/METR-LA/reduced_METR-LA-test.csv') 

    scaler = StandardScaler()

    scaler.fit(df.values)

    df_train_scaled = pd.DataFrame(scaler.transform(df.values),
                               index=df.index,
                               columns=df.columns)

    df_test_scaled = pd.DataFrame(scaler.transform(df_test.values),
                              index=df_test.index,
                              columns=df_test.columns)

    dataset = TimeSeriesDataset(df_train_scaled, window_size=WINDOW_SIZE, horizon=HORIZON)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TimeSeriesDataset(df_test_scaled, window_size=WINDOW_SIZE, horizon=HORIZON)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleRNN(input_size=df_train_scaled.shape[1], hidden_size=HIDDEN_SIZE, num_layers=4)
    print(model)
    train_model(model, dataloader, num_epochs=NUM_EPOCHS)

    test_loss, r2 = evaluate_model(model, test_loader)
    print(f'Test R2: {r2:.6f}')
    print(f"Test Loss (MSE) su sensori non visti: {test_loss:.6f}")

