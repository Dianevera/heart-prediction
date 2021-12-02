from torch import nn, zeros
import torch
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, output_dim, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.output_dim = output_dim #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm

        self.fc = nn.Linear(hidden_size, output_dim) #fully connected last layer
        self.sig = nn.Sigmoid()


    def forward(self,x):
        x = torch.unsqueeze(x,0)
        h_0 = zeros(self.num_layers, x.size(0), self.hidden_size) #hidden state
        c_0 = zeros(self.num_layers, x.size(0), self.hidden_size) #internal state
        # Propagate input through LSTM
        output, _ = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = self.fc(output) #Final Output
        out = self.sig(out)
        return out.squeeze()