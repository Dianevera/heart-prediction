from torch import nn
import torch

class Hidden_layer(nn.Module):
    def __init__(self, input_size, output_size, relu=True):
        """
            Create the Hidden_layer object.

                    Parameters:
                            input_size (int): The input size of the layer
                            output_size (int): The output size of the layer
                            relu (bool): If True we add a ReLu layer at the end
                            sizes ([int]): List of the sizes of the hidden layers 
        """
        super(Hidden_layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = torch.nn.Linear(self.input_size, self.output_size)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        """
            Computer the output tensor.

                    Parameters:
                            x (Tensor): The input tensor
                            
                    Returns:
                            x (Tensor): The output tensor
        """
        x = self.linear(x)
        x = self.relu(x)
        return x
    
def create_hidden_layers(sizes):
    """
    Create the hidden layers.

            Parameters:
                    sizes ([int]): List of the sizes of the hidden layers 

            Returns:
                    (ModuleList): The hidden layers
    """
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
        def __init__(self, sizes):
            """
                Create the MLP object.

                        Parameters:
                                sizes (int): The sizes for the hidden layers
            """
            super(MLP, self).__init__()
            
            self.hiddens = create_hidden_layers(sizes)
            self.sigmoid = torch.nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x):
            """
                Computer the output tensor.

                        Parameters:
                                x (Tensor): The input tensor

                        Returns:
                                x (Tensor): The output tensor
            """
            for hidden in self.hiddens:         
                x = hidden(x)
            out = self.softmax(x)
            return out