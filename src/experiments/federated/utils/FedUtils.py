from experiments.common import SimpleRNN

def initialize_model(hidden_size = 128, layers=4):
    model = SimpleRNN(input_size=1, hidden_size=hidden_size, num_layers=layers)
    return model