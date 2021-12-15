from torch import nn
import torch

class Hidden_layer(nn.Module):
    def __init__(self, input_size, output_size, relu=True):
        super(Hidden_layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = torch.nn.Linear(self.input_size, self.output_size)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x
    
def create_hidden_layers(sizes):
    length = len(sizes)
    hiddens = []
    
    for i in range(length - 1):
        if i == length - 2:
            hidden = Hidden_layer(sizes[i], sizes[i + 1], relu=False)
        else:
            hidden = Hidden_layer(sizes[i], sizes[i + 1])
        hiddens.append(hidden)
        
    return nn.ModuleList(hiddens)

class MLP(nn.Module):
        def __init__(self, sizes):#, output_size, hidden_size = 100, input_size = 36):
            super(MLP, self).__init__()
            
            self.hiddens = create_hidden_layers(sizes)
            self.sigmoid = torch.nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x):
            for hidden in self.hiddens:
                x = hidden(x)
                
            out = self.softmax(x)
            return out